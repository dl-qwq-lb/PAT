import gc
import math
import time
import torch
import argparse
import json
import os
from typing import List

from vllm.vllm_flash_attn import flash_attn_with_kvcache as vllm_flash_attn_with_kvcache

from FlashInferAttn import get_decode_wrapper
from RelayAttention import schedule_single_sys, RelayAttention, RelayAttentionPlus
from prefix_attn import (
    SeqGroup,
    generate_random_kv_cache,
    PrefixTree,
    PrefixTreeCPP,
    prefix_attn_with_kvcache,
    generate_tree_seqs,
)
from prefix_attn.data_class import KernelInfo
from prefix_attn.data_class import create_seq_group
from FastTree import (
    FastTreeParams,
    fasttree_decode,
    fasttree_preparation,
    generate_tree,
    qkv_preparation,
)


def load_json_allow_prefix(path: str) -> dict:
    """兼容形如 '(WorkerDict ...) { ... }' 的前缀/拼接日志；从首个'{'起只解析第一个完整 JSON 对象。"""
    with open(path, "r") as f:
        text = f.read()
    start = text.find("{")
    if start == -1:
        raise ValueError(f"Invalid JSON content in {path}: cannot find '{{'")
    decoder = json.JSONDecoder()
    obj, _end = decoder.raw_decode(text[start:])
    return obj

def Timer(func, iter):
    import numpy as np

    # warm up
    for i in range(10):
        func()

    latencies = []
    for _ in range(iter):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        func()
        end_event.record()

        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        latencies.append(elapsed_time_ms)

    return np.median(latencies)


def _compare_and_report(
    baseline_name: str,
    baseline_output: torch.Tensor,
    ref_output: torch.Tensor,
    max_error: float,
):
    max_diff = (baseline_output - ref_output).abs().max().item()
    mean_diff = (baseline_output - ref_output).abs().mean().item()

    status = "PASSED"
    try:
        assert max_diff <= max_error
    except AssertionError:
        status = "FAILED"

    return {"status": status, "max_diff": max_diff, "mean_diff": mean_diff}


def benchmark(
    seq_group: SeqGroup,
    num_blocks: int,
    nheads_q: int,
    nheads_kv: int,
    head_dim: int,
    block_size: int,
    n_repeats: int,
    dtype: torch.dtype,
    device: str = "cuda:0",
    seed: int = 0,
    baselines: List[str] = None,
):
    torch.cuda.set_device(device)
    torch.random.manual_seed(seed)
    assert head_dim in [64, 128], "head_dim should be 64 or 128 currently"
    assert nheads_q % nheads_kv == 0
    scale = 1.0 / math.sqrt(head_dim)
    max_error = 1e-2

    all_available_baselines = ["vllm-fa", "pat", "flashinfer"]
    run_baselines = all_available_baselines if "all" in baselines else baselines

    q = torch.randn(len(seq_group), 1, nheads_q, head_dim, device=device, dtype=dtype)
    (
        seqlens,
        k_cache,
        v_cache,
        kv_padding_mask,
        block_table,
        k_cache_paged,
        v_cache_paged,
    ) = generate_random_kv_cache(
        seq_group=seq_group,
        num_blocks=num_blocks,
        block_size=block_size,
        nheads_kv=nheads_kv,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        seed=seed,
    )

    results = {}
    latencies = {}
    correctness_report = {}

    if "flashinfer" in run_baselines:

        def flashinfer_forward(wrapper, q, K_Cache, V_Cache):
            o = wrapper.run(q, (K_Cache, V_Cache), return_lse=True)
            return o

        decode_wrapper = get_decode_wrapper(
            seq_group.seqlens,
            block_table,
            block_size,
            nheads_q,
            nheads_kv,
            head_dim,
            scale,
            device,
        )

        def flashinfer_func():
            res = flashinfer_forward(
                decode_wrapper, q.squeeze(1), k_cache_paged, v_cache_paged
            )[0].unsqueeze(1)
            return res

        latencies["flashinfer"] = Timer(flashinfer_func, n_repeats)
        out_flashinfer = flashinfer_func()
        results["flashinfer"] = out_flashinfer  # Shape: (batch, 1, n_q, h_dim)

    if "vllm-fa" in run_baselines:
        if vllm_flash_attn_with_kvcache is not None:

            def vllm_fa_func():
                _ = vllm_flash_attn_with_kvcache(
                    q,
                    k_cache_paged,
                    v_cache_paged,
                    cache_seqlens=seqlens,
                    block_table=block_table,
                    causal=True,
                    num_splits=0,
                )

            latencies["vllm-fa"] = Timer(vllm_fa_func, n_repeats)
            out_vllm_fa = vllm_flash_attn_with_kvcache(
                q,
                k_cache_paged,
                v_cache_paged,
                cache_seqlens=seqlens,
                block_table=block_table,
                causal=True,
                num_splits=0,
            )
            results["vllm-fa"] = out_vllm_fa

    if "ra++" in run_baselines:
        seqlen4ra, block_table4ra = schedule_single_sys(
            block_table=block_table,
            seq_lens=seq_group.seqlens,
            block_size=block_size,
            device=device,
        )

        def ra_func():
            _ = RelayAttentionPlus(
                q,
                k_cache_paged,
                v_cache_paged,
                seqlens=seqlen4ra,
                block_tables=block_table4ra,
            )

        latencies["ra++"] = Timer(ra_func, n_repeats)
        out_ra = RelayAttentionPlus(
            q,
            k_cache_paged,
            v_cache_paged,
            seqlens=seqlen4ra,
            block_tables=block_table4ra,
        )
        results["ra++"] = out_ra.unsqueeze(1)  # Shape: (batch, 1, n_q, h_dim)

    if "ra" in run_baselines:
        sys_len, seqlen4ra = schedule_single_sys(
            block_table=block_table,
            seq_lens=seq_group.seqlens,
            block_size=block_size,
            device=device,
            only_sys=True,
        )
        k_cache_ra = [
            k_cache[:, sys_len:, :, :],  # usr
            k_cache[:, :sys_len, :, :],
        ]  # sys
        v_cache_ra = [
            v_cache[:, sys_len:, :, :],  # usr
            v_cache[:, :sys_len, :, :],
        ]  # sys

        def ra_func():
            _ = RelayAttention(
                q,
                k_cache_ra,
                v_cache_ra,
                seqlens=seqlen4ra,
            )

        latencies["ra"] = Timer(ra_func, n_repeats)
        out_ra = RelayAttention(
            q,
            k_cache_ra,
            v_cache_ra,
            seqlens=seqlen4ra,
        )
        results["ra"] = out_ra.unsqueeze(1)  # Shape: (batch, 1, n_q, h_dim)

    if "pat" in run_baselines:
        # 在同一次 benchmark 调用中，先跑 baseline 调度再跑 SOTA 调度，统一输入与环境，方便直接对比。
        MNWs = None
        table = block_table.cpu()
        seq_lens = seq_group.seqlens

        # 1) baseline 调度：use_sota = False
        tree_base = PrefixTreeCPP(block_size)
        tree_base.build_radix_tree(seq_lens, table)
        tree_base.pack_schedule(MNWs, nheads_q // nheads_kv, nheads_kv, False)
        tree_base.kernel_info.to_gpu(torch.device(device))

        out_pat_base = torch.empty_like(q, device=device, dtype=dtype)

        def pat_baseline_func():
            prefix_attn_with_kvcache(
                q=q,
                k_cache_paged=k_cache_paged,
                v_cache_paged=v_cache_paged,
                tree=tree_base,
                softmax_scale=scale,
                out=out_pat_base,
            )

        latencies["pat_baseline"] = Timer(pat_baseline_func, n_repeats)
        pat_baseline_func()
        results["pat_baseline"] = out_pat_base

        # 2) SOTA 调度：use_sota = True，CTA 上限在 C++ balancePackSota 内部根据 SM 数自动推断
        tree_sota = PrefixTreeCPP(block_size)
        tree_sota.build_radix_tree(seq_lens, table)
        tree_sota.pack_schedule(MNWs, nheads_q // nheads_kv, nheads_kv, True)
        tree_sota.kernel_info.to_gpu(torch.device(device))

        out_pat_sota = torch.empty_like(q, device=device, dtype=dtype)

        def pat_sota_func():
            prefix_attn_with_kvcache(
                q=q,
                k_cache_paged=k_cache_paged,
                v_cache_paged=v_cache_paged,
                tree=tree_sota,
                softmax_scale=scale,
                out=out_pat_sota,
            )

        latencies["pat_sota"] = Timer(pat_sota_func, n_repeats)
        pat_sota_func()
        results["pat_sota"] = out_pat_sota

    if "vllm-fa" in baselines and "vllm-fa" in results:
        # 对所有已计算结果（除 vllm-fa 自身）做 correctness 对比，包括 pat_baseline 与 pat_sota
        for method, out in results.items():
            if method == "vllm-fa":
                continue
            correctness_report[method] = _compare_and_report(
                baseline_name=method,
                baseline_output=out,
                ref_output=results["vllm-fa"],
                max_error=max_error,
            )

    return latencies, correctness_report


def ft_benchmark(
    tree: str,
    nheads_q: int,
    nheads_kv: int,
    head_dim: int,
    n_repeats: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda:0",
    baselines: list = None,
):
    torch.cuda.set_device(device)
    nodes, contexts = tree.split("_")
    layer_sizes = list(map(int, nodes.split(",")))
    context_sizes = list(map(int, contexts.split(",")))

    tree_info = generate_tree(layer_sizes, context_sizes)

    Q, K_cache, V_cache, cache_seqlens, K_tree_tensor, V_tree_tensor, KV_ptrs = (
        qkv_preparation(tree_info, nheads_q, nheads_kv, head_dim, device, dtype)
    )
    latencies = {}
    correctness_report = {}
    out = None
    out_flash = None

    if nheads_q // nheads_kv == 1 or nheads_q // nheads_kv == 4:
        # FastTree will get wrong answer or runtime error when there are multi sys in level 0.
        # which is a limitation of FastTree.
        if int(tree[0]) == 1 and tree[1] == ",":
            batch_size = Q.shape[0]
            Q_TILE_SIZE_PER_PHASE = None
            if nheads_q // nheads_kv == 1:
                Q_TILE_SIZE_PER_PHASE = [64, 16]
            elif nheads_q // nheads_kv == 4:
                Q_TILE_SIZE_PER_PHASE = [16, 4]
            elif nheads_q // nheads_kv == 16:
                Q_TILE_SIZE_PER_PHASE = [4, 1]

            KV_TILE_SIZE_PER_PHASE = [32, 32]
            KV_SPLIT_SIZES = [1024, 128]
            para_threshs1 = [132, 528]
            para_threshs2 = [132, 132]
            params = FastTreeParams()
            alpha = 0.66
            beta = 0.33
            gamma = 0.1
            params.set_values(alpha, beta, gamma)
            params.set_q_tile_sizes(Q_TILE_SIZE_PER_PHASE)
            params.set_kv_tile_sizes(KV_TILE_SIZE_PER_PHASE)
            params.set_kv_group_num(nheads_q // nheads_kv)

            fasttree_aux, node_assignment = fasttree_preparation(
                tree_info,
                KV_ptrs,
                batch_size,
                nheads_q,
                nheads_kv,
                head_dim,
                KV_SPLIT_SIZES,
                para_threshs1,
                para_threshs2,
                params,
                device,
            )
            sm_scale = 1.0 / (head_dim**0.5)
            max_error = 1e-2
            out = torch.empty(
                batch_size, nheads_q, head_dim, device=device, dtype=dtype
            )

            def fasttree_func():
                fasttree_decode(
                    Q,
                    K_tree_tensor,
                    V_tree_tensor,
                    out,
                    *fasttree_aux,
                    Q_TILE_SIZE_PER_PHASE,
                    KV_TILE_SIZE_PER_PHASE,
                    sm_scale,
                )

            # FastTree can only test in this way to get right latencies
            ms = Timer(fasttree_func, n_repeats)
            latencies["FastTree"] = ms

    if "fa" in baselines:

        def flash_func():
            vllm_flash_attn_with_kvcache(
                Q.unsqueeze(1),
                K_cache,
                V_cache,
                cache_seqlens=cache_seqlens,
                causal=True,
                num_splits=0,  # 0: best split;  1: do not split kv
            )

        latencies["fa"] = Timer(flash_func, n_repeats)

        # it's slow, cause it can't utilize L2 cache well
        out_flash = vllm_flash_attn_with_kvcache(
            Q.unsqueeze(1),
            K_cache,
            V_cache,
            cache_seqlens=cache_seqlens,
            causal=True,
            num_splits=0,  # 0: best split;  1: do not split kv
        )
        out_flash = out_flash.squeeze(1)

    try:
        if out is not None and out_flash is not None:
            correctness_report["FastTree"] = _compare_and_report(
                "FastTree", out, out_flash, 0.01
            )
    except Exception as e:
        correctness_report["FastTree"] = {"status": "ERROR", "message": str(e)}

    return latencies, correctness_report


def cascade_benchmark(
    tree: str,
    nheads_q: int,
    nheads_kv: int,
    head_dim: int,
    n_repeats: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda:0",
):
    import ctypes

    try:
        import flashinfer
    except ImportError:
        raise ImportError(
            "Please install flashinfer to use MultiLevelCascadeAttentionWrapper."
        )

    _cudart = ctypes.CDLL("libcudart.so")

    def build_bfs_kv_indices(node_pages_per_level, node_num_per_level):
        """
        Build BFS-based KV indices.
        """
        num_levels = len(node_pages_per_level)
        nodes = []
        for node_num in node_num_per_level:
            nodes.append([None] * node_num)

        current_index = [0]

        def dfs(level, pos):
            start = current_index[0]
            length = node_pages_per_level[level]
            end = start + length
            slice_for_node = list(range(start, end))
            nodes[level][pos] = slice_for_node
            current_index[0] = end

            if level < num_levels - 1:
                stride = node_num_per_level[level + 1] // node_num_per_level[level]
                for i in range(stride):
                    dfs(level + 1, pos * stride + i)

        dfs(0, 0)

        result = []
        for lvl in range(num_levels):
            concat_for_level = []
            for pos in range(node_num_per_level[lvl]):
                concat_for_level.extend(nodes[lvl][pos])
            result.append(
                torch.tensor(concat_for_level, dtype=torch.int32, device="cuda:0")
            )

        return result

    node_num_per_level, node_seqlen_per_level = tree.split("_")
    # Parse levels
    node_num_per_level = list(map(int, node_num_per_level.split(",")))
    node_seqlen_per_level = list(map(int, node_seqlen_per_level.split(",")))

    # Fixed hyperparameters
    num_layers = 1
    page_size = 1

    node_pages_per_level = [ns // page_size for ns in node_seqlen_per_level]
    pages_per_level = [
        node_pages * node_num
        for node_pages, node_num in zip(node_pages_per_level, node_num_per_level)
    ]
    total_num_pages = sum(pages_per_level)
    batch_size = node_num_per_level[-1]
    num_levels = len(node_num_per_level)

    # Workspace
    workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.uint8, device=device)

    wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
        num_levels, workspace_buffer, "NHD"
    )

    # Build KV page indices
    kv_page_indices_arr = build_bfs_kv_indices(node_pages_per_level, node_num_per_level)

    kv_page_indptr_arr = []
    for node_pages, node_num in zip(node_pages_per_level, node_num_per_level):
        kv_page_indptr_arr.append(
            torch.arange(node_num + 1, device=device, dtype=torch.int32) * node_pages
        )

    kv_last_page_len_arr = []
    for node_num in node_num_per_level:
        kv_last_page_len_arr.append(
            torch.full((node_num,), page_size, dtype=torch.int32, device=device)
        )

    kv_cache_at_layer = [
        torch.randn(
            total_num_pages,
            2,
            page_size,
            nheads_kv,
            head_dim,
            dtype=dtype,
            device=device,
        )
        for _ in range(num_layers)
    ]

    # Qo index pointers
    qo_indptr_arr = []
    for node_num in node_num_per_level:
        qo_indptr_arr.append(
            torch.linspace(
                0, batch_size, node_num + 1, dtype=torch.int32, device=device
            )
        )

    wrapper.plan(
        qo_indptr_arr,
        kv_page_indptr_arr,
        kv_page_indices_arr,
        kv_last_page_len_arr,
        nheads_q,
        nheads_kv,
        head_dim,
        page_size,
    )

    q_arr = [
        torch.randn(batch_size, nheads_q, head_dim, dtype=dtype, device=device)
        for _ in range(num_layers)
    ]

    latencies = {}

    def cascade_func():
        wrapper.run(q_arr[0], kv_cache_at_layer[0])

    latencies["cascade"] = Timer(cascade_func, n_repeats)

    return latencies, {}


def DeFT_benchmark(
    tree: str,
    nheads_q: int,
    nheads_kv: int,
    head_dim: int,
    n_repeats: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda:0",
    baselines: list = None,
):
    from DeFT import DeFT_preparation, DeFT_decode

    torch.cuda.set_device(device)
    nodes, contexts = tree.split("_")
    layer_sizes = list(map(int, nodes.split(",")))
    context_sizes = list(map(int, contexts.split(",")))

    tree_info = generate_tree(layer_sizes, context_sizes)

    Q, K_cache, V_cache, cache_seqlens, K_tree_tensor, V_tree_tensor, KV_ptrs = (
        qkv_preparation(tree_info, nheads_q, nheads_kv, head_dim, device, dtype)
    )
    batch_size = Q.shape[0]

    subtree_len = 128
    mask_len = 64
    # Use the tile config in the official DeFT repo
    Q_TILE_SIZE = 32
    KV_TILE_SIZE = 16

    DeFT_aux = DeFT_preparation(
        tree_info, K_cache, subtree_len, mask_len, nheads_q, head_dim
    )
    out = torch.empty(batch_size, nheads_q, head_dim, device=device, dtype=dtype)
    sm_scale = 1.0 / (head_dim**0.5)
    max_error = 1e-2

    results = {}
    latencies = {}
    correctness_report = {}

    def DeFT_func():
        DeFT_decode(
            Q,
            K_cache.view(-1, K_cache.shape[-2], K_cache.shape[-1]),
            V_cache.view(-1, V_cache.shape[-2], V_cache.shape[-1]),
            out,
            *DeFT_aux,
            Q_TILE_SIZE,
            KV_TILE_SIZE,
            sm_scale,
            mask_len,
        )

    latencies["DeFT"] = Timer(DeFT_func, n_repeats)
    results["DeFT"] = out

    Q = Q.unsqueeze(1)
    out_flash = vllm_flash_attn_with_kvcache(
        Q,
        K_cache,
        V_cache,
        cache_seqlens=cache_seqlens,
        causal=True,
        num_splits=0,  # 0: best split;  1: do not split kv
    )
    correctness_report["DeFT"] = _compare_and_report(
        "DeFT", out, out_flash.squeeze(1), max_error
    )

    return latencies, correctness_report

# 原有版本，这里做了增强，增加了异常捕获和日志记录功能，以便在benchmark过程中即使出现错误也能得到有用的反馈，并且不会中断整个测试流程。
# def run_benchmark(
#     tree: str,
#     nheads_q: int,
#     nheads_kv: int,
#     output_path: str,
#     head_dim: int = 128,
#     block_size: int = 32,
#     n_repeats: int = 20,
#     dtype: torch.dtype = torch.float16,
#     device: str = "cuda:0",
#     seed: int = 0,
# ):
#     seq_group, num_blocks = generate_tree_seqs(tree, block_size)
#     baseline = [
#         "vllm-fa",
#         "pat",
#         "flashinfer",
#     ]
#     if int(tree[0]) == 1 and tree[1] == ",":
#         baseline.extend(["ra", "ra++", "cascade", "deft"])

#     lats_std, corr_std = benchmark(
#         seq_group,
#         num_blocks,
#         nheads_q,
#         nheads_kv,
#         head_dim,
#         block_size,
#         n_repeats,
#         dtype,
#         device,
#         seed,
#         baselines=baseline,
#     )

#     lats_ft, corr_ft = {}, {}
#     if nheads_q // nheads_kv in [1, 4, 16]:
#         lats_ft, corr_ft = ft_benchmark(
#             tree,
#             nheads_q,
#             nheads_kv,
#             head_dim,
#             n_repeats,
#             dtype,
#             device,
#             baselines=["fa"],
#         )

#     lats_deft, corr_deft = {}, {}
#     lats_cascade, corr_cascade = {}, {}

#     if int(tree[0]) == 1 and tree[1] == ",":
#         lats_deft, corr_deft = DeFT_benchmark(
#             tree, nheads_q, nheads_kv, head_dim, n_repeats, dtype, device
#         )
#         lats_cascade, corr_cascade = cascade_benchmark(
#             tree, nheads_q, nheads_kv, head_dim, n_repeats, dtype, device
#         )

#     result_entry = {
#         "tree": tree,
#         "nheads_q": nheads_q,
#         "nheads_kv": nheads_kv,
#         "head_dim": head_dim,
#         "block_size": block_size,
#         "latencies": {**lats_std, **lats_ft, **lats_deft, **lats_cascade},
#         "correctness": {**corr_std, **corr_ft, **corr_deft, **corr_cascade},
#     }
#     # print(result_entry)

#     data = []
#     if os.path.exists(output_path):
#         try:
#             with open(output_path, "r") as f:
#                 content = f.read()
#                 if content.strip():
#                     data = json.loads(content)
#         except:
#             data = []

#     data.append(result_entry)

#     with open(output_path, "w") as f:
#         json.dump(data, f, indent=4)

#     print(f"Done: {tree} | {nheads_q}/{nheads_kv}")

def run_benchmark(
    tree: str,
    nheads_q: int,
    nheads_kv: int,
    output_path: str,
    rl_testcase_json: str = None,
    # pat_schedule: str = "baseline", # balance debug
    head_dim: int = 128,
    block_size: int = 32,
    n_repeats: int = 20,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda:0",
    seed: int = 0,
):
    import traceback
    import pdb
    
    # 初始化结果字典，确保异常时也能写入基本信息
    result_entry = {
        "tree": tree,
        "nheads_q": nheads_q,
        "nheads_kv": nheads_kv,
        "head_dim": head_dim,
        "block_size": block_size,
        "latencies": {},
        "correctness": {},
        "status": "failed",
        "error": None,
    }
    
    try:
        if rl_testcase_json is not None:
            rl = load_json_allow_prefix(rl_testcase_json)

            seq_lens = list(map(int, rl["seq_lens"]))
            block_tables_padded = rl["block_tables"]
            # RL_testcase.json 的 block_tables 通常被 padding 到同一长度；按 seq_lens 截断恢复真实 blockinfo
            block_tables = [
                row[: (seq_lens[i] + block_size - 1) // block_size]
                for i, row in enumerate(block_tables_padded)
            ]

            seq_group = create_seq_group(block_tables=block_tables, seq_lens=seq_lens, block_size=block_size)
            num_blocks = int(rl.get("num_blocks_total", 0))
            if num_blocks <= 0:
                max_bid = max((max(r) for r in block_tables if len(r) > 0), default=0)
                num_blocks = int(max_bid) + 1
            else:
                max_bid = max((max(r) for r in block_tables if len(r) > 0), default=0)
                num_blocks = max(num_blocks, int(max_bid) + 1)

            # RL 用例默认只跑 pat（包含 pat_baseline/pat_sota），避免 vllm-fa reference 带来的额外显存/耗时风险。
            # 输出结构仍与 kernel_perf.json 既有格式一致：latencies 里有 pat_baseline/pat_sota，correctness 为空字典。
            baseline = ["pat"]
        else:
            seq_group, num_blocks = generate_tree_seqs(tree, block_size)
            baseline = [
                "vllm-fa",
                "pat",
                "flashinfer",
            ]
            if int(tree[0]) == 1 and tree[1] == ",":
                baseline.extend(["ra", "ra++", "cascade", "deft"])
 
        lats_std, corr_std = benchmark(
            seq_group,
            num_blocks,
            nheads_q,
            nheads_kv,
            head_dim,
            block_size,
            n_repeats,
            dtype,
            device,
            seed,
            baselines=baseline,
        )
 
        lats_ft, corr_ft = {}, {}
        # FastTree/DeFT/cascade 仅适用于 synthetic tree string（例如 '1,10_4096,416'）。
        if rl_testcase_json is None and nheads_q // nheads_kv in [1, 4, 16]:
            lats_ft, corr_ft = ft_benchmark(
                tree,
                nheads_q,
                nheads_kv,
                head_dim,
                n_repeats,
                dtype,
                device,
                baselines=["fa"],
            )
 
        lats_deft, corr_deft = {}, {}
        lats_cascade, corr_cascade = {}, {}
 
        if rl_testcase_json is None and int(tree[0]) == 1 and tree[1] == ",":
            lats_deft, corr_deft = DeFT_benchmark(
                tree, nheads_q, nheads_kv, head_dim, n_repeats, dtype, device
            )
            lats_cascade, corr_cascade = cascade_benchmark(
                tree, nheads_q, nheads_kv, head_dim, n_repeats, dtype, device
            )
 
        result_entry = {
            "tree": tree,
            "nheads_q": nheads_q,
            "nheads_kv": nheads_kv,
            "head_dim": head_dim,
            "block_size": block_size,
            "latencies": {**lats_std, **lats_ft, **lats_deft, **lats_cascade},
            "correctness": {**corr_std, **corr_ft, **corr_deft, **corr_cascade},
            "status": "success",
            "error": None,
        }
        # print(result_entry)
        
    except Exception as e:
        # 捕获异常信息
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        result_entry["error"] = error_info
        result_entry["status"] = "failed"
        
        # 打印错误信息
        print(f"\n{'='*60}")
        print(f"ERROR in benchmark: {tree} | {nheads_q}/{nheads_kv}")
        print(f"{'='*60}")
        print(f"Exception Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print(f"\nTraceback:")
        print(traceback.format_exc())
        print(f"{'='*60}\n")
        
        # 取消下面注释可以在出错时打断点
        # pdb.set_trace()
        
    finally:
        # 无论成功或失败，都写入json文件
        data = []
        if os.path.exists(output_path):
            try:
                with open(output_path, "r") as f:
                    content = f.read()
                    if content.strip():
                        data = json.loads(content)
            except:
                data = []
 
        data.append(result_entry)
 
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
 
        status_icon = "✓" if result_entry["status"] == "success" else "✗"
        print(f"Done: {tree} | {nheads_q}/{nheads_kv} [{status_icon}]")
        if result_entry["status"] == "failed":
            print(f"  Error: {result_entry['error']['type']}: {result_entry['error']['message']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=str, default=None,
                        help="Tree string for synthetic test case. Ignored if --rl_testcase_json is set.")
    parser.add_argument("--rl_testcase_json", type=str, default=None,
                        help="Path to RL_testcase.json containing {pid, seq_lens, block_tables, num_blocks_total}.")
    parser.add_argument("--nheads_q", type=int)
    parser.add_argument("--nheads_kv", type=int)
    parser.add_argument("--output_file", type=str)
    # balance debug
    # parser.add_argument("--pat_schedule", type=str, default="baseline",
    #                    choices=["baseline", "sota"],
    #                    help="PAT scheduling strategy: baseline (Python) or sota (C++ balancePackSota)")

    args = parser.parse_args()

    head_dim = 128
    block_size = 32
    n_repeats = 20
    dtype = torch.float16
    device = "cuda:1"
    seed = int(time.time())

    tree_name = args.tree
    if args.rl_testcase_json is not None and tree_name is None:
        rl = load_json_allow_prefix(args.rl_testcase_json)
        pid = rl.get("pid", "unknown")
        tree_name = f"rl_pid_{pid}"

    if tree_name is None:
        raise ValueError("Either --tree or --rl_testcase_json must be provided.")

    run_benchmark(
        tree_name,
        args.nheads_q,
        args.nheads_kv,
        args.output_file,
        args.rl_testcase_json,
        # args.pat_schedule, # balance debug
        head_dim,
        block_size,
        n_repeats,
        dtype,
        device,
        seed,
    )

