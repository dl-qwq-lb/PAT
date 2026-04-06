import sys  
import os  
import math  
from math import ceil  
from typing import List, Dict, Any  
import numpy as np  
import torch  
import argparse
import time
  
from prefix_attn import generate_tree_seqs 
from prefix_attn.prefix_tree import PrefixTree  
from prefix_attn.data_class import KernelInfo, create_seq_group  
from prefix_attn.utils import generate_random_prefix_seqs  
from prefix_attn import PrefixTreeCPP
  
  
# utils part══════════════════════════════════════════════════════════════════════   
def get_sm_count(device=0):
    """返回指定 GPU 设备的 SM 数量"""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        return props.multi_processor_count
    else:
        return None
      
def pad_block_table(block_table: List[List[int]],  
                    padding_value: int = 0,  
                    dtype=torch.int32) -> torch.Tensor:  
    max_len = max(len(row) for row in block_table) if block_table else 0  
    padded = [row + [padding_value] * (max_len - len(row)) for row in block_table]  
    return torch.tensor(padded, dtype=dtype)  
  
  
def unpad_block_table(block_table: List[List[int]],  
                      seq_lens: List[int],  
                      block_size: int) -> List[List[int]]:  
    return [block_table[i][:(seq_lens[i] + block_size - 1) // block_size]  
            for i in range(len(seq_lens))]  
  
  
def seqgroup_to_block_table(seq_group, block_size: int) -> List[List[int]]:  
    table = []  
    for seq in seq_group.sequences:  
        n = math.ceil(seq.seqlen / block_size)  
        table.append(seq.block_ids[:n])  
    return table  
  
  
# run schedule method══════════════════════════════════════════════════════════════════════   
  
def run_cpp_schedule(seq_lens: List[int],  
                     block_table: List[List[int]],  
                     block_size: int,  
                     HRatio: int = 1,  
                     kvHead: int = 8,  
                     MNWs=None):  

    padded_tensor = pad_block_table(block_table)          # CPU int32 Tensor  
    tree = PrefixTreeCPP(block_size)  
    tree.build_radix_tree(seq_lens, padded_tensor)        # 建基数树  
    tree.pack_schedule(MNWs, HRatio, kvHead)              # 调度，结果写入 _internal_info  
    return tree, tree.kernel_info                          # kernel_info 是 C++ KernelInfo 引用  


def run_cpp_schedule_sota(seq_lens: List[int],
                          block_table: List[List[int]],
                          block_size: int,
                          HRatio: int = 1,
                          kvHead: int = 8,
                          MNWs=None):

    padded_tensor = pad_block_table(block_table)          # CPU int32 Tensor
    tree = PrefixTreeCPP(block_size)
    tree.build_radix_tree(seq_lens, padded_tensor)        # 建基数树
    tree.pack_schedule_sota(MNWs, HRatio, kvHead)  # 使用 SOTA 调度链路
    return tree, tree.kernel_info

def print_cpp_kernel_info(ki, label: str = "C++ kernel_info"):  

    sep = "=" * 65  
    print(f"\n{sep}\n  {label}\n{sep}")  
    try:  
        print(f"  MNWs              = {ki.MNWs}")  
        print(f"  max_split_per_seq = {ki.max_split_per_seq}")  
        print(f"  max_seqs_in_CTA   = {ki.max_seqs_in_CTA}")  
        print(f"  max_blocks_in_CTA = {ki.max_blocks_in_CTA}")  
        print(f"  num_split_per_seq = {ki.num_split_per_seq.tolist()}")  
  
        for k, mnw in enumerate(ki.MNWs):  
            print(f"\n  ── Kernel bucket[{k}]  MNW = {mnw} ──")  
            # q_tables[k]: shape (num_CTAs, max_seqs_in_bucket)  
            print(f"    q_tables         = {ki.q_tables[k].tolist()}")  
            # block_tables[k]: shape (num_CTAs, max_blocks_in_bucket)  
            print(f"    block_tables     = {ki.block_tables[k].tolist()}")  
            # 以下均为 shape (num_CTAs,)  
            print(f"    num_seqs_per_CTA = {ki.num_seqs_per_CTAs[k].tolist()}")  
            print(f"    CTA_ranks        = {ki.CTA_ranks[k].tolist()}")  
            print(f"    kv_in_CTAs       = {ki.kv_in_CTAs[k].tolist()}")  
    except AttributeError as e:  
        print(f"  [WARN] C++ KernelInfo 字段未暴露给 Python: {e}")   
  
def _to_sorted_list(v):
    if isinstance(v, torch.Tensor):
        return sorted(v.tolist())
    return sorted(v)
  
def measure_performance(func, *args, iterations=10):  
    for _ in range(5):  
        func(*args)  
  
    times = []  
    for _ in range(iterations):  
        start = time.time()  
        func(*args)  
        end = time.time()  
        times.append(end - start)  
  
    avg_time = sum(times) / len(times)  
    return avg_time


def compare_schedules(ki_cpp, ki_sota: KernelInfo, label: str = "") -> bool:  

    sep = "-" * 65  
    print(f"\n{sep}")  
    print(f"  [DIFF] Compare: {label}")  
    print(sep)  
  
    all_pass = True  
  
    try:  
        # 1. num_split_per_seq  
        cpp_nsps = ki_cpp.num_split_per_seq.tolist()  
        sota_nsps = (ki_sota.num_split_per_seq.tolist()  
                   if isinstance(ki_sota.num_split_per_seq, torch.Tensor)  
                   else ki_sota.num_split_per_seq)  
        match = cpp_nsps == sota_nsps  
        status = "✅ PASS" if match else "❌ DIFF"  
        print(f"  {status}  num_split_per_seq")  
        if not match:  
            print(f"         C++ = {cpp_nsps}")  
            print(f"         sota  = {sota_nsps}")  
            all_pass = False
  
        # 2. MNWs nums 
        n_cpp = len(ki_cpp.MNWs)  
        n_sota  = len(ki_sota.MNWs)  
        if n_cpp != n_sota:  
            print(f"  ❌ DIFF  num_kernel_buckets: baseline={n_cpp}, sota={n_sota}")  
            all_pass = False  
        else:  
            print(f"  ✅ PASS  num_kernel_buckets = {n_cpp}")  
  
        # 3. 对每个 bucket 逐字段比较  
        for k in range(min(n_cpp, n_sota)):  
            mnw_cpp = ki_cpp.MNWs[k]  
            mnw_sota  = ki_sota.MNWs[k]  
  
            cpp_kv  = _to_sorted_list(ki_cpp.kv_in_CTAs[k])  
            sota_kv   = _to_sorted_list(ki_sota.kv_in_CTAs[k])  
            cpp_rk  = _to_sorted_list(ki_cpp.CTA_ranks[k])  
            sota_rk   = _to_sorted_list(ki_sota.CTA_ranks[k])  
            cpp_ns  = _to_sorted_list(ki_cpp.num_seqs_per_CTAs[k])  
            sota_ns   = _to_sorted_list(ki_sota.num_seqs_per_CTAs[k])  
  
            kv_ok = cpp_kv == sota_kv  
            rk_ok = cpp_rk == sota_rk  
            ns_ok = cpp_ns == sota_ns  
            bucket_ok = kv_ok and rk_ok and ns_ok  
  
            status = "✅ PASS" if bucket_ok else "❌ DIFF"  
            print(f"\n  {status}  Bucket[{k}] MNW C++={mnw_cpp} / sota={mnw_sota}")  
  
            # if not kv_ok:  
            #     print(f"         kv_in_CTAs  C++={cpp_kv}")  
            #     print(f"                     sota ={sota_kv}")  
            # if not rk_ok:  
            #     print(f"         CTA_ranks   C++={cpp_rk}")  
            #     print(f"                     sota ={sota_rk}")  
            # if not ns_ok:  
            #     print(f"         num_seqs    C++={cpp_ns}")  
            #     print(f"                     sota ={sota_ns}")  
  
            all_pass &= bucket_ok  
  
    except AttributeError as e:  
        print(f"  [SKIP] C++ KernelInfo 字段未暴露，无法比较: {e}")   
  
    result_str = "✅ ALL PASS" if all_pass else "❌ DIFFERENCES FOUND"  
    print(f"\n  Overall [{label}]: {result_str}")  
    return all_pass    
  
# test engine══════════════════════════════════════════════════════════════════════  

def run_single_test_cpp_compare(name: str,
                                seq_lens: List[int],
                                block_table: List[List[int]],
                                block_size: int,
                                HRatio: int,
                                kvHead: int,
                                MNWs=None) -> Dict[str, Any]:
    print(f"\n{'#'*70}")
    print(f"  TEST CASE: {name} (C++ balancePack vs balancePackSota)")
    print(f"  batch_size={len(seq_lens)}, block_size={block_size}")
    print(f"  HRatio={HRatio}, kvHead={kvHead}")

    # 测量 baseline 性能
    def run_baseline():
        tree = PrefixTreeCPP(block_size)
        padded_tensor = pad_block_table(block_table)
        tree.build_radix_tree(seq_lens, padded_tensor)
        tree.pack_schedule(MNWs, HRatio, kvHead, False)  # use_sota=False
        return tree.kernel_info

    time_base = measure_performance(run_baseline)
    ki_base = run_baseline()

    # 测量 SOTA 性能
    def run_sota():
        tree = PrefixTreeCPP(block_size)
        padded_tensor = pad_block_table(block_table)
        tree.build_radix_tree(seq_lens, padded_tensor)
        tree.pack_schedule_sota(MNWs, HRatio, kvHead)
        return tree.kernel_info

    time_sota = measure_performance(run_sota)
    ki_sota = run_sota()

    print(f"  Performance: baseline={time_base:.6f}s, SOTA={time_sota:.6f}s (avg over 10 runs)")
    print_cpp_kernel_info(ki_base, label=f"C++ KernelInfo [baseline {name}]")
    print_cpp_kernel_info(ki_sota, label=f"C++ KernelInfo [SOTA {name}]")

    comp = compare_schedules(ki_base, ki_sota, label=f"{name} C++ baseline vs SOTA")
    print(f"  test {name} result: {comp}")

    return {
        "name": name,
        "batch_size": len(seq_lens),
        "block_size": block_size,
        "HRatio": HRatio,
        "kvHead": kvHead,
        "time_baseline": time_base,
        "time_sota": time_sota,
        "comparison_pass": comp
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run schedule tests for balancePack comparison.")
    parser.add_argument("--tree", type=str, help="Tree string for test case (e.g., '1,10_4096,416')")
    parser.add_argument("--nheads_q", type=int, default=32, help="Number of Q heads")
    parser.add_argument("--nheads_kv", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--block_size", type=int, default=32, help="Block size")
    parser.add_argument("--output_file", type=str, help="Output file for results")

    args = parser.parse_args()

    seq_group, _ = generate_tree_seqs(args.tree, args.block_size)
    seq_lens = [seq.seqlen for seq in seq_group.sequences]
    block_table = seqgroup_to_block_table(seq_group, args.block_size)
    HRatio = args.nheads_q // args.nheads_kv
    kvHead = args.nheads_kv

    result = run_single_test_cpp_compare(f"tree_{args.tree}",
                                            seq_lens,
                                            block_table,
                                            args.block_size,
                                            HRatio,
                                            kvHead)

    if args.output_file:
        import json
        output_data = {
            "tree": args.tree,
            "nheads_q": args.nheads_q,
            "nheads_kv": args.nheads_kv,
            "block_size": args.block_size,
            "result": result
        }
        with open(args.output_file, 'a') as f:
            json.dump(output_data, f)
            f.write('\n')
