import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from prefix_attn import PrefixTreeCPP


@dataclass(frozen=True)
class RLRecord:
    pid: int
    seq_lens: List[int]
    block_tables: List[List[int]]
    num_blocks_total: Optional[int] = None


def iter_concatenated_json_objects(text: str) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from a text that contains multiple concatenated JSON blobs.

    This file format often looks like:
        (WorkerDict pid=...) [tag]\n { ...json... }\n (WorkerDict pid=...) ... { ... }

    We find the next '{' and use json.JSONDecoder().raw_decode from there.
    """
    dec = json.JSONDecoder()
    i = 0
    n = len(text)
    while True:
        start = text.find("{", i)
        if start < 0:
            return
        try:
            obj, consumed = dec.raw_decode(text[start:])
        except json.JSONDecodeError:
            # Skip this '{' and keep searching.
            i = start + 1
            continue
        yield obj
        i = start + consumed
        if i >= n:
            return


def load_rl_records(path: str) -> List[RLRecord]:
    raw = open(path, "r", encoding="utf-8", errors="ignore").read()
    records: List[RLRecord] = []
    for obj in iter_concatenated_json_objects(raw):
        if not isinstance(obj, dict):
            continue
        if "seq_lens" not in obj or "block_tables" not in obj:
            continue
        records.append(
            RLRecord(
                pid=int(obj.get("pid", -1)),
                seq_lens=[int(x) for x in obj["seq_lens"]],
                block_tables=[[int(y) for y in row] for row in obj["block_tables"]],
                num_blocks_total=(int(obj["num_blocks_total"]) if "num_blocks_total" in obj else None),
            )
        )
    return records


def infer_block_size(record: RLRecord, candidates: List[int] = [8, 16, 32, 64, 128]) -> Optional[int]:
    """Infer block_size by checking where padding zeros begin.

    Heuristic: for a correct block_size, each row should have zeros after the last real block id.
    We use: expected_blocks = ceil(seq_len / block_size), and require row[expected_blocks:] are all 0.

    NOTE: this assumes the producer padded with 0 and that blocks after the real range are 0.
    """
    if not record.seq_lens or not record.block_tables:
        return None

    max_blocks = max(len(r) for r in record.block_tables)

    def ok_for(bs: int) -> bool:
        for seqlen, row in zip(record.seq_lens, record.block_tables):
            need = int((seqlen + bs - 1) // bs)
            if need > len(row):
                return False
            tail = row[need:]
            if any(v != 0 for v in tail):
                return False
        return True

    for bs in candidates:
        if bs <= 0:
            continue
        if ok_for(bs):
            return bs

    # If strict check fails, still prefer 32 if it "fits" (need <= row_len) for all rows.
    for bs in [32, 16, 64]:
        if all(int((seqlen + bs - 1) // bs) <= len(row) for seqlen, row in zip(record.seq_lens, record.block_tables)):
            return bs

    return None


def to_int32_table(block_tables: List[List[int]]) -> torch.Tensor:
    max_len = max((len(r) for r in block_tables), default=0)
    padded = [r + [0] * (max_len - len(r)) for r in block_tables]
    return torch.tensor(padded, dtype=torch.int32, device="cpu")


def _measure_seconds(fn, iterations: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    times: List[float] = []
    for _ in range(iterations):
        t0 = time.time()
        fn()
        t1 = time.time()
        times.append(t1 - t0)
    return sum(times) / max(1, len(times))


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _append_kernel_perf_json(path: str, entry: Dict[str, Any]) -> None:
    # Keep the same convention as benchmark_kernel.py: JSON array stored in a single file.
    data: List[Any] = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    data = json.loads(content)
        except Exception:
            data = []
    if not isinstance(data, list):
        data = []
    data.append(entry)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def summarize_kernel_info(ki) -> Dict[str, Any]:
    # kernel_info is a C++ ref; fields may be tensors or lists.
    out: Dict[str, Any] = {}
    out["MNWs"] = list(getattr(ki, "MNWs", []))
    out["max_split_per_seq"] = int(getattr(ki, "max_split_per_seq", -1))
    out["max_seqs_in_CTA"] = int(getattr(ki, "max_seqs_in_CTA", -1))
    out["max_blocks_in_CTA"] = int(getattr(ki, "max_blocks_in_CTA", -1))

    nsps = getattr(ki, "num_split_per_seq", None)
    if isinstance(nsps, torch.Tensor):
        out["num_split_per_seq"] = nsps.cpu().tolist()
    elif nsps is not None:
        out["num_split_per_seq"] = list(nsps)

    # Bucket-level stats (lightweight)
    kvs = getattr(ki, "kv_in_CTAs", None)
    if isinstance(kvs, list) and kvs and isinstance(kvs[0], torch.Tensor):
        out["kv_in_CTAs_stats"] = [
            {
                "num_ctas": int(t.numel()),
                "min": int(t.min().item()) if t.numel() else 0,
                "max": int(t.max().item()) if t.numel() else 0,
                "sum": int(t.sum().item()) if t.numel() else 0,
            }
            for t in kvs
        ]
    return out


def _tensor_or_list_to_list(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.cpu().tolist()
    if isinstance(x, list):
        return [_tensor_or_list_to_list(v) for v in x]
    return x


def dump_kernel_info_full(ki, max_ctas: int = 0) -> Dict[str, Any]:
    """Dump kernel_info into JSON-serializable dict.

    max_ctas:
      - 0 means dump all CTAs.
      - >0 means truncate per-bucket CTA dimension to at most max_ctas.
    """
    out: Dict[str, Any] = {}
    for k in [
        "MNWs",
        "max_split_per_seq",
        "max_seqs_in_CTA",
        "max_blocks_in_CTA",
        "num_split_per_seq",
        "q_tables",
        "block_tables",
        "num_seqs_per_CTAs",
        "CTA_ranks",
        "kv_in_CTAs",
    ]:
        if hasattr(ki, k):
            out[k] = _tensor_or_list_to_list(getattr(ki, k))

    if max_ctas and max_ctas > 0:
        # Truncate bucket-wise CTA dimension (axis 0) for large dumps.
        for field in ["q_tables", "block_tables", "num_seqs_per_CTAs", "CTA_ranks", "kv_in_CTAs"]:
            v = out.get(field)
            if not isinstance(v, list):
                continue
            new_v = []
            for bucket in v:
                if isinstance(bucket, list):
                    new_v.append(bucket[:max_ctas])
                else:
                    new_v.append(bucket)
            out[field] = new_v

    return out


def build_block_trie(seq_lens: List[int], block_tables: List[List[int]], block_size: int) -> Dict[str, Any]:
    """Reconstruct a trie from per-seq block id paths.

    This mirrors what build_radix_tree conceptually builds: shared prefixes of block-id sequences.
    Returns a JSON-serializable structure with nodes/edges and summary stats.
    """
    # Node 0 is root.
    parents: List[int] = [-1]
    block_id: List[int] = [-1]
    depth: List[int] = [0]
    count: List[int] = [0]
    children: List[Dict[int, int]] = [dict()]

    def new_node(p: int, b: int, d: int) -> int:
        parents.append(p)
        block_id.append(b)
        depth.append(d)
        count.append(0)
        children.append(dict())
        return len(parents) - 1

    max_need = 0
    for seqlen, row in zip(seq_lens, block_tables):
        need = int((seqlen + block_size - 1) // block_size)
        max_need = max(max_need, need)
        node = 0
        count[node] += 1
        # Only use the real prefix part; ignore padded tail.
        for j in range(min(need, len(row))):
            b = int(row[j])
            nxt = children[node].get(b)
            if nxt is None:
                nxt = new_node(node, b, depth[node] + 1)
                children[node][b] = nxt
            node = nxt
            count[node] += 1

    num_nodes = len(parents)
    num_edges = num_nodes - 1
    max_depth = max(depth) if depth else 0
    out_degree = [len(c) for c in children]
    leaves = sum(1 for i in range(num_nodes) if i != 0 and out_degree[i] == 0)

    # Build a compact edge list.
    edges: List[Tuple[int, int]] = []
    for p, cmap in enumerate(children):
        for _, ch in cmap.items():
            edges.append((p, ch))

    nodes = [
        {
            "id": i,
            "parent": parents[i],
            "block_id": block_id[i],
            "depth": depth[i],
            "count": count[i],
            "out_degree": out_degree[i],
        }
        for i in range(num_nodes)
    ]

    return {
        "stats": {
            "num_seqs": len(seq_lens),
            "block_size": block_size,
            "max_blocks_needed": max_need,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "max_depth": max_depth,
            "num_leaves": leaves,
            "avg_out_degree": (sum(out_degree) / num_nodes) if num_nodes else 0.0,
        },
        "nodes": nodes,
        "edges": edges,
    }


def main():
    parser = argparse.ArgumentParser(description="Run PAT schedule from RL_testcase.json-style block_tables.")
    parser.add_argument("--path", type=str, default="RL_testcase.json")
    parser.add_argument("--list", action="store_true", help="List all records and exit")
    parser.add_argument("--index", type=int, default=0, help="Pick record by index in file")
    parser.add_argument("--pid", type=int, default=None, help="Pick record by pid (first match)")

    parser.add_argument("--block_size", type=int, default=0, help="Block size; 0 means infer")
    parser.add_argument("--nheads_q", type=int, default=32)
    parser.add_argument("--nheads_kv", type=int, default=8)
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Timing iterations for baseline/SOTA scheduling (includes build + schedule)",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs for timing")

    # Output integration (match run_kernel_bench.sh conventions)
    parser.add_argument(
        "--kernel_output_file",
        type=str,
        default=None,
        help="Append a schedule-only entry into kernel_perf.json (JSON array)",
    )
    parser.add_argument(
        "--schedule_output_file",
        type=str,
        default=None,
        help="Append a schedule entry into schedule_perf.json (JSONL)",
    )
    parser.add_argument(
        "--schedule_log_file",
        type=str,
        default=None,
        help="Append human-readable logs into schedule.log",
    )
    parser.add_argument(
        "--dump_json",
        type=str,
        default=None,
        help="Write a one-line JSON summary to this path (JSONL)",
    )

    # Debug outputs (do NOT change PrefixTreeCPP interface)
    parser.add_argument(
        "--dump_tree_json",
        type=str,
        default=None,
        help="Dump reconstructed block trie (nodes/edges/stats) into a JSON file",
    )
    parser.add_argument(
        "--dump_kernel_info_json",
        type=str,
        default=None,
        help="Dump baseline/SOTA kernel_info into a JSON file",
    )
    parser.add_argument(
        "--dump_kernel_info_max_ctas",
        type=int,
        default=0,
        help="If >0, truncate per-bucket CTA dimension when dumping kernel_info",
    )

    args = parser.parse_args()

    if args.nheads_q % args.nheads_kv != 0:
        raise SystemExit("nheads_q must be divisible by nheads_kv")

    # Resolve path robustly: support calling from repo root or from benchmark/.
    path = args.path
    if not os.path.exists(path):
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, args.path),
            os.path.join(here, "RL_testcase.json"),
            os.path.join(os.path.dirname(here), "test", "RL_testcase.json"),
            os.path.join(os.path.dirname(here), "benchmark", "RL_testcase.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break

    records = load_rl_records(path)
    if not records:
        raise SystemExit(f"No records found in {path}")

    if args.list:
        for i, r in enumerate(records):
            row_len = max((len(x) for x in r.block_tables), default=0)
            print(
                f"[{i:03d}] pid={r.pid} batch={len(r.seq_lens)} max_blocks={row_len} num_blocks_total={r.num_blocks_total}"
            )
        return

    if args.pid is not None:
        rec = next((r for r in records if r.pid == args.pid), None)
        if rec is None:
            raise SystemExit(f"pid={args.pid} not found")
    else:
        if args.index < 0 or args.index >= len(records):
            raise SystemExit(f"index out of range: {args.index} (0..{len(records)-1})")
        rec = records[args.index]

    block_size = args.block_size
    if block_size <= 0:
        inferred = infer_block_size(rec)
        if inferred is None:
            raise SystemExit("Failed to infer block_size; please pass --block_size")
        block_size = inferred

    seq_lens = rec.seq_lens
    table = to_int32_table(rec.block_tables)

    HRatio = args.nheads_q // args.nheads_kv
    kvHead = args.nheads_kv

    case_name = f"RL(index={args.index},pid={rec.pid})"

    def run_baseline_once() -> None:
        tree = PrefixTreeCPP(block_size)
        tree.build_radix_tree(seq_lens, table)
        tree.pack_schedule(None, HRatio, kvHead, False)

    def run_sota_once() -> None:
        tree = PrefixTreeCPP(block_size)
        tree.build_radix_tree(seq_lens, table)
        tree.pack_schedule_sota(None, HRatio, kvHead)

    time_baseline = _measure_seconds(run_baseline_once, iterations=args.iterations, warmup=args.warmup)
    time_sota = _measure_seconds(run_sota_once, iterations=args.iterations, warmup=args.warmup)

    # One run for kernel_info summary (not in timed loop)
    tree_base = PrefixTreeCPP(block_size)
    tree_base.build_radix_tree(seq_lens, table)
    tree_base.pack_schedule(None, HRatio, kvHead, False)
    ki_base = tree_base.kernel_info

    tree_sota = PrefixTreeCPP(block_size)
    tree_sota.build_radix_tree(seq_lens, table)
    tree_sota.pack_schedule_sota(None, HRatio, kvHead)
    ki_sota = tree_sota.kernel_info

    # Lightweight comparison: num_split_per_seq only (cheap + stable)
    comp_pass = True
    try:
        a = ki_base.num_split_per_seq
        b = ki_sota.num_split_per_seq
        if isinstance(a, torch.Tensor):
            a = a.cpu().tolist()
        if isinstance(b, torch.Tensor):
            b = b.cpu().tolist()
        comp_pass = list(a) == list(b)
    except Exception:
        comp_pass = False

    result = {
        "name": case_name,
        "batch_size": len(seq_lens),
        "block_size": block_size,
        "HRatio": HRatio,
        "kvHead": kvHead,
        "time_baseline": time_baseline,
        "time_sota": time_sota,
        "comparison_pass": comp_pass,
    }

    summary = {
        "path": path,
        "pid": rec.pid,
        "index": args.index,
        "seq_lens_count": len(seq_lens),
        "block_size": block_size,
        "nheads_q": args.nheads_q,
        "nheads_kv": args.nheads_kv,
        "result": result,
        "kernel_info_baseline": summarize_kernel_info(ki_base),
        "kernel_info_sota": summarize_kernel_info(ki_sota),
    }

    if args.dump_tree_json:
        tree_dump = build_block_trie(seq_lens, rec.block_tables, block_size)
        os.makedirs(os.path.dirname(args.dump_tree_json) or ".", exist_ok=True)
        with open(args.dump_tree_json, "w", encoding="utf-8") as f:
            json.dump(tree_dump, f, indent=2)

    if args.dump_kernel_info_json:
        ki_dump = {
            "case": {
                "name": case_name,
                "pid": rec.pid,
                "index": args.index,
                "nheads_q": args.nheads_q,
                "nheads_kv": args.nheads_kv,
                "HRatio": HRatio,
                "kvHead": kvHead,
                "block_size": block_size,
            },
            "baseline": dump_kernel_info_full(ki_base, max_ctas=args.dump_kernel_info_max_ctas),
            "sota": dump_kernel_info_full(ki_sota, max_ctas=args.dump_kernel_info_max_ctas),
        }
        os.makedirs(os.path.dirname(args.dump_kernel_info_json) or ".", exist_ok=True)
        with open(args.dump_kernel_info_json, "w", encoding="utf-8") as f:
            json.dump(ki_dump, f, indent=2)

    # Console output (useful when redirecting to schedule.log)
    sep = "#" * 70
    print(f"\n{sep}")
    print(f"  TEST CASE: {case_name} (RL block_tables -> PrefixTreeCPP)")
    print(f"  batch_size={len(seq_lens)}, block_size={block_size}")
    print(f"  HRatio={HRatio}, kvHead={kvHead}")
    print(f"  Performance: baseline={time_baseline:.6f}s, SOTA={time_sota:.6f}s (avg over {args.iterations} runs)")
    print(f"  Compare(num_split_per_seq): {comp_pass}")
    print("  kernel_info (SOTA summary):")
    print(f"    MNWs={summary['kernel_info_sota'].get('MNWs', [])}")
    print(f"    max_split_per_seq={summary['kernel_info_sota'].get('max_split_per_seq')}")
    print(f"    max_seqs_in_CTA={summary['kernel_info_sota'].get('max_seqs_in_CTA')}")
    print(f"    max_blocks_in_CTA={summary['kernel_info_sota'].get('max_blocks_in_CTA')}")

    # Optional integrated outputs
    if args.schedule_log_file:
        os.makedirs(os.path.dirname(args.schedule_log_file) or ".", exist_ok=True)
        with open(args.schedule_log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{sep}\n")
            f.write(f"  TEST CASE: {case_name} (RL block_tables -> PrefixTreeCPP)\n")
            f.write(f"  pid={rec.pid} index={args.index}\n")
            f.write(f"  batch_size={len(seq_lens)}, block_size={block_size}\n")
            f.write(f"  HRatio={HRatio}, kvHead={kvHead}\n")
            f.write(
                f"  Performance: baseline={time_baseline:.6f}s, SOTA={time_sota:.6f}s (avg over {args.iterations} runs)\n"
            )
            f.write(f"  Compare(num_split_per_seq): {comp_pass}\n")

    if args.schedule_output_file:
        # Match schedule_test.py output format (JSONL entries)
        schedule_entry = {
            "tree": case_name,
            "nheads_q": args.nheads_q,
            "nheads_kv": args.nheads_kv,
            "block_size": block_size,
            "result": result,
            "pid": rec.pid,
            "index": args.index,
            "source": "RL_testcase",
        }
        _append_jsonl(args.schedule_output_file, schedule_entry)

    if args.kernel_output_file:
        # Keep kernel_perf.json as a JSON array; append a schedule-only entry.
        kernel_entry = {
            "tree": case_name,
            "nheads_q": args.nheads_q,
            "nheads_kv": args.nheads_kv,
            "head_dim": 128,
            "block_size": block_size,
            "latencies": {
                "pat_schedule_baseline_ms": time_baseline * 1000.0,
                "pat_schedule_sota_ms": time_sota * 1000.0,
            },
            "correctness": {},
            "status": "success",
            "error": None,
            "source": "RL_testcase",
            "pid": rec.pid,
            "index": args.index,
        }
        _append_kernel_perf_json(args.kernel_output_file, kernel_entry)

    if args.dump_json:
        _append_jsonl(args.dump_json, summary)


if __name__ == "__main__":
    main()
