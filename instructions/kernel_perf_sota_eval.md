# kernel_perf_sota vs kernel_perf 分析与建议

## 0. 对比背景

- baseline 文件: `benchmark/kernel_perf.json`
- sota 文件: `benchmark/kernel_perf_sota.json`
- 公共条目数: 80（相同的 tree、nheads_q、nheads_kv、head_dim、block_size 组合）
- 在公共条目上统计 PAT 延迟：
  - baseline 平均 PAT 延迟: 0.9274 ms
  - sota 平均 PAT 延迟:    0.9569 ms
  - 即 **整体上 SOTA 版本的 PAT 稍慢于 baseline**（约多 3% 左右，具体 case 有涨有跌）。
- 在 sota 文件中，PAT correctness 出现 2 次 FAILED：
  - tree = `1,10_4096,416`, (nheads_q=16, nheads_kv=8, head_dim=128, block_size=32)
  - tree = `1,10_4096,416`, (nheads_q=32, nheads_kv=8, head_dim=128, block_size=32)

下面按你提的四个问题分别分析。

---

## 1. 为何调整 benchmark_kernel 后，几乎所有算法 latencies 都有波动？

现象：
- 对比 kernel_perf 与 kernel_perf_sota，同一条目下不仅 PAT，有时连 flashinfer / vllm-fa / ra / FastTree / DeFT / cascade 的延迟也有涨有跌。

原因分析：

1）**随机输入 + CUDA 运行噪声导致的自然波动**
- benchmark_kernel 中每次运行都会：
  - 使用 `generate_random_kv_cache` 和随机 `q`（依赖 `seed`，但也有 `time.time()` 这类非固定种子路径）；
  - 在 GPU 上多次重复测量，取 median；
  - GPU 上还有其他线程 / driver 任务、L2/L1 cache 热度、时钟动态调整等噪声。
- 即使完全不改任何调度，只要重新跑，所有算法的 latency 都会在一个小范围内抖动，这是预期现象。

2）**你改动了 PAT 调度路径，引入了不同的 kernel 行为模式**
- baseline kernel_perf 使用的是 Python PrefixTree + `balancePack` 调度。
- sota kernel_perf_sota 把 PAT 部分改成 C++ PrefixTreeCPP + `pack_schedule_sota` 调度，然后仍然在同一个 benchmark 框架下测所有算法。
- 当 PAT 调度发生变化时：
  - PAT 本身的访问模式（block-table 切分方式、CTA 分布）变化，L2/L1 cache 命中模式会影响随后其他算法的测量；
  - 不同算法的运行顺序和“预热效果”会导致整体的 cache/SM 状态不同，进而影响其他方法的 median latency；
- 换言之：**改变一个算法的调度，也会通过 cache / 竞争等间接影响其它算法的时间分布**。

3）**运行顺序 & 复用同一进程的 side-effect**
- benchmark_kernel 中各算法是在同一 Python 进程 / 同一 GPU context 下依次运行的：flashinfer → vllm-fa → ra++ → ra → pat → ...
- 如果某个算法在 SOTA 版本下变“更重”，占用更多带宽或 cache，接下来算法的 warmup 和 timing 也会被改变。

综合来看：
- ① 的“几乎所有算法 latency 都有波动”，并不说明这些算法被你“功能性地改坏”，而是：
  - 随机输入 + GPU 噪声 + 调度改动 → 整体系统状态变化 → 全局时间都有小幅变化；
- 真正要看的，是**趋势**：对 PAT 本身，SOTA 平均上稍慢；其他算法只要 correctness 依旧 PASSED，轻微波动可以视作噪声。

---

## 2. 为何 SOTA 文件中 PAT 性能“非最优”更多，整体性能低于 baseline？

从宏观统计：
- baseline: avg PAT latency ≈ 0.9274 ms
- sota:     avg PAT latency ≈ 0.9569 ms
- 说明在这批配置上，SOTA 调度 **整体略慢**。
- 再结合已有 schedule_perf 分析：SOTA 在很多 tree 上并不是稳定优于 baseline，有的 tree 甚至明显变慢。

原因更具体地落在 SOTA 调度策略上：

1）**SOTA 的 KV 分块更“激进”，牺牲了部分前缀复用结构**
- baseline 的 `balancePack` 更保守：
  - 仅对 KV 很长的 box 做有限的 block 级切分；
  - 重点是保证 CTA 数量够用，适当平衡，而不是严格全局负载均衡。
- SOTA 的 `balancePackSota` 则：
  - 按全局/分组的 L_kv 目标，把长 KV 冒泡出来后多次切分；
  - 每个切分块都携带完整的 q_table，从而可以强行“平衡” KV 工作量。
- 在很多 tree 上，这种切分：
  - 打破了非常长的公共前缀段，让一个大前缀被拆成多个小窗口；
  - 提高了 `split_per_seq`，增加了不同 CTAs 间对同一序列的读写/调度开销；
  - 使得 N 方向上的 KV 长度在 cache 利用上变得更“碎片化”，有时反而不如 baseline 的“中等不平衡但 locality 好”。

2）**全局 L_kv 估计没有对“pathology tree”特殊保护**
- 例如 `1,10_4096,416` 这类有长 sys 前缀 + 短 user 尾部的 tree：
  - baseline 会保留一个大公共前缀块，PAT 的设计就是吃这类结构红利；
  - SOTA 的全局 L_kv 如果约等于中等长度，就会强迫对这个长前缀进行多段切分，破坏 prefix-attn 的设计优势。
- 这种情况下：
  - 虽然 LPT 风格调度在“理论负载均衡”上好看，但**对 prefix-tree 这种“结构性共享”的工作负载不够友好**，导致实际耗时变大。

3）**部分配置下 PAT 本来就不是最优（相对 flashinfer/ra++ 等）**
- 在 baseline 中，已存在若干 tree/head 组合，PAT 不是最优方法，SOTA 并不能把这些 case 全部“救活”；
- SOTA 对一些 case 有提升（如你在 test.py 看到的单条 case），但在全局 80 条数据上：
  - 有的 tree 变快，有的 tree 变慢，整体平均下来是略慢；
  - “PAT 非最优 case”在 SOTA 统计下自然会更多一些。

小结：
- SOTA 调度当前版本更多是“通用 LPT + 全局 L_kv”，没有把 prefix-tree 数据结构的特性（特别是长前缀）当成一等公民来保护。
- 这导致一部分场景可以收益，另一部分场景反而损失更多，宏观平均后 PAT 的性能比 baseline 略低。

---

## 3. 为何 SOTA 文件开头有两次 PAT correctness=FAILED？重组策略是否有问题？

数据：
- 两次 FAILED 样本：
  - tree = `1,10_4096,416`, (nheads_q=16, nheads_kv=8, head_dim=128, block_size=32)
  - tree = `1,10_4096,416`, (nheads_q=32, nheads_kv=8, head_dim=128, block_size=32)
- 错误量级：
  - max_diff ≈ 0.049
  - mean_diff ≈ 0.0036
- 阈值 `max_error = 1e-2`，因此 correctness 判定 FAILED。

这说明：
- 在这两个特定配置上：
  - SOTA 调度 "+" PAT kernel 的输出，与基准（vllm-fa）存在 ~5e-2 级别的最大偏差，已经超过我们容忍的 1e-2 阈值；
- 这与之前在 test/test.py 中 **同一 tree 但使用 PrefixTreeCPP + baseline/SOTA 对比时，两者完全一致** 并不矛盾，因为：
  - test.py 里我们比较的是 “baseline 调度 vs SOTA 调度” 的 **同一套 PAT kernel 输出**；
  - benchmark_kernel 里对 correctness 是以 vllm-fa 为参考，对比的是 PAT vs vllm-fa；
  - 两端测试的 seed / 输入构造方式也略有不同（比如 generate_random_kv_cache 的种子、随机 q 的初始化时间等）。

更关键的问题是：
- SOTA 的重组策略是否可能引入：
  1. 某些 CTA 的 KV 范围（block_table 子区间）与 baseline 不完全一致；
  2. `CTA_rank` / `split_per_seq` 的使用方式是否和 kernel 的合并逻辑 100% 对齐；
- 若存在以下任何一条，都会导致数值相对 vllm-fa 出现更大的“调度依赖误差”：
  - 切分时 `kv_in_CTA` 计算与真实可见 KV 长度不严格一致 → softmax 归一化范围出现偏差；
  - 某些 CTAs 的 prefix 段缺失了部分 block（block_table 子数组截取 index 计算有 off-by-one）；
  - `CTA_rank` 的分配导致合并顺序与期望的不一致（例如同一序列被切成 3 段，但 rank 重复或跳号）。

结合目前信息，比较合理的判断是：
- **重组策略本身存在“边界 case”实现不完全正确**：
  - 大部分 tree、heads 下 SOTA 与 baseline 一致，说明主流程没问题；
  - 特定 tree (`1,10_4096,416` + 某些 head 组合) 出现 PASSED→FAILED，说明这个 tree 的 block pattern+L_kv 组合触发了 SOTA 算法的“拐角分支”（例如某个>1 分块 + threshold 约束同时生效）。

改进建议（针对 correctness）：
1）**先对 FAILED 样本跑“baseline vs SOTA vs vllm-fa 的三方数值对比”**
   - 在 test/test.py 基础上，专门针对上述两条配置：
     - 记录 baseline-PAT vs vllm-fa 的 diff；
     - 记录 SOTA-PAT vs vllm-fa 的 diff；
   - 如果 baseline vs vllm-fa 是 PASSED，而 SOTA vs vllm-fa FAILED，则基本可以确认问题出在 SOTA 调度。

2）**在 C++ PrefixTree 的 `balancePackSota` 中，对以下点加 assert/日志：**
   - 每个 PackedBox 的 block 范围是否严格覆盖了原始树中对应前缀的所有 block（不重不漏）；
   - 对同一 qid 的所有 box：
     - `CTA_rank` 是否从 0 开始连续编号；
     - 各段的 `kv_in_CTA` 累加后是否等于该 qid 实际应该看到的 KV 长度；
   - 对 SOTA 分割前后的 KernelInfo：比较 `num_split_per_seq` 和 block_tables 的覆盖度。

3）**短期 workaround：**
   - 在 benchmark_kernel 中，如果 tree=`1,10_4096,416` 且 HRatio=nheads_q//nheads_kv 属于已知失败的组合，可以临时回退到 baseline 调度，保证 correctness；
   - 或者在 SOTA 内部做一个“前缀保护策略”：当公共前缀长度极长且 HRatio 特定时，不对这类节点做 KV 切分。


三、保护超长前缀的详细方案
3.1 如何定义“超长前缀”
可以从两个维度衡量：

绝对长度：共享前缀的 token 数（或 block 数）超过某个阈值（例如 512 tokens 或 16 blocks）。

相对占比：共享前缀长度占该请求总 KV 长度的比例超过一定阈值（如 50%），且该组查询数量较多。

建议采用组合条件：

共享前缀的 block 数 ≥ min_shared_blocks（如 16）；

且该组查询数量 ≥ min_seqs_in_group（如 4）；

且共享前缀长度占总 KV 长度比例 ≥ min_share_ratio（如 0.3）。

这些阈值可通过实验调优，例如以 1,10_4096,416 为基准，找到使性能提升且 correctness 稳定的值。

3.2 在调度中保护超长前缀
方案 A：在 _tree_heuristics 阶段避免拆分
当递归处理节点时，如果该节点的 length（共享前缀长度）超过阈值，且 s_value（共享次数）较大，则：

将整个节点作为一个整体生成 PackedBox，不进行后续的 split 或 merge 操作（即强制保留为大盒子）。

这相当于在调度树的更高层就锁定共享前缀，防止后续 balancePackSota 拆散它。

方案 B：在 balancePackSota 中对特定盒子禁用拆分
在拆分循环前，对盒子进行标记：

cpp
bool is_long_prefix = (block_table_ptr->size() >= min_shared_blocks) &&
                      (q_table.size() >= min_seqs_in_group);
拆分时，如果 is_long_prefix 为 true，则跳过该盒子的拆分（即使其 KV 长度超过 L_kv）。

但这样可能导致负载不均，因此需要配合其他平衡手段（如允许该盒子单独占用多个 CTA，但内部不分拆）。

方案 C：分层拆分——仅拆分后缀部分
在生成盒子时，将“共享前缀”与“独有后缀”分离（如 _tree_heuristics 中已实现 merged_blocks 和 shared_blocks 的区分）。

balancePackSota 只对独有后缀部分进行拆分，而共享前缀部分始终保持为一个整体，不切分。

这要求 kernel 支持一个 CTA 处理“共享前缀+部分后缀”的混合结构（当前 PAT kernel 可能已支持，因为 block_table_ptr 可以是任意连续块列表，包含共享块和后缀块）。拆分时只需将后缀部分切分，共享前缀部分保留完整。

推荐方案 C，因为它最符合 PAT 设计初衷，且能自然避免破坏共享前缀。具体实现时，需要在 _tree_heuristics 中明确标记哪些块是共享的，哪些是独有的，然后在 balancePackSota 拆分时，只拆分独有部分，共享部分保留原样。

3.3 调整拆分粒度 L_kv 的计算
对于包含长共享前缀的盒子，可调整其 L_kv 的上限，使其更大，减少拆分次数。例如：

cpp
int effective_L_kv = L_kv;
if (is_long_prefix) {
    effective_L_kv = std::max(L_kv, long_prefix_min_chunk_size);
}
其中 long_prefix_min_chunk_size 可以设为共享前缀的长度，或者一个较大的常数（如 1024）。

3.4 修正 correctness 问题
单元测试：针对 FAILED 的 case，编写 C++ 测试，对比 balancePackSota 输出的 KernelInfo 与预期 KernelInfo（baseline 调度产生的），确保：

每个 query 的所有 KV 片段（按 split_per_seq 顺序）的 block_table 拼接后等于原始 KV 块列表。

每个片段的 CTA_rank 从 0 开始连续，且无重复。

边界检查：在拆分循环中添加断言：

cpp
assert(start_blk >= 0 && end_blk <= total_blocks);
assert(sub.kv_in_CTA <= chunk_cap_tokens);
assert(sub.block_table_ptr->size() == end_blk - start_blk);
验证合并顺序：在 kernel 外部模拟合并逻辑，确保最终输出与 vllm-fa 一致（可用 Python 脚本测试）。

四、其他配套优化
4.1 成本模型调优
当前 cost = 1 * qn + (1/HRatio) * kv_len 过于简化。建议：

通过微基准测试拟合 alpha 和 beta，例如测量不同 (T_q, l_kv) 组合的 kernel 执行时间，用线性回归得到。

引入共享前缀的折扣因子：对于共享部分，成本应降低（因为只加载一次）。可以在计算盒子成本时，对共享前缀部分应用折扣 γ（如 0.5），对独有后缀部分正常计算。

4.2 保留多 query 打包
当前 SOTA 由于 _tree_heuristics 中 mm=1，导致每个盒子只有一个 query。可以调整 mm 为动态值（如 4、8 或 16），使得共享前缀的多个 query 仍能打包在一起，减少 CTA 数量和合并开销。但需要注意：

多 query 打包时，每个 query 的独有后缀可能不同，需要确保 kernel 能正确处理（PAT kernel 目前要求所有 query 共享完全相同的 block_table_ptr，因此只能打包那些独有后缀也相同的 query）。

可以在 _tree_heuristics 中，对于叶子节点（没有子节点），按 mm 打包；对于非叶子节点，如果子节点都是叶子且独有后缀相同，也可以打包。

4.3 自适应选择调度策略
根据工作负载特征，动态决定使用 baseline 调度还是 SOTA 调度。例如：

如果共享前缀比例高（如 > 70%）且总请求数少，使用 baseline 调度（保留大盒子）。

如果共享前缀比例低或请求数多，使用 SOTA 调度（精细拆分）。
这需要在 Python 端实现一个简单的决策器，在 pack_schedule 中调用。