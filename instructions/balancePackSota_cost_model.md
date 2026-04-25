# balancePackSota 成本模型详解与重构建议

本文档聚焦 C++ 侧 `PrefixTree::balancePackSota(...)` 的“成本模型”分支（即 `base_cta >= cta_limit` 时的按 box 选择 split_k 的逻辑），解释它由哪些组分构成、每个组分在估计什么、为什么这样估计，以及目前的局限与一份可落地的重构方案。

> 适用范围：这里讨论的是 **调度阶段** 的“是否进一步切分超长 prefix box”决策模型，而不是 kernel 本身的数学正确性。

---

## 1. 该模型在什么条件下触发？

`balancePackSota` 入口会先计算：

- `base_cta`: `_tree_heuristics` 产出的初始 CTA 数（`PackedBox` 数量）。
- `threshold_cnt` / `cta_limit`: 由 `kvHead` 推导的经验阈值（kvHead 越大阈值越小）。

当 `base_cta >= cta_limit` 时，代码进入“cap 分支”（我这里称为 **@cap**）：

- 目的：在 CTA 数已经“不少”的前提下，**只对明显拖尾的超长 box 做有限的再切分**，避免全局 L_kv 方案带来大量碎片化与 schedule CPU overhead。

同时还有一个 fast-skip：当 `base_cta` 远大于阈值且调度已较均衡时会直接返回（避免 schedule-only benchmark 下的稳定慢化）。

---

## 2. cost model 的基本抽象

对每个候选 box（一个 CTA 对应的 `PackedBox`），定义一个粗糙的“原始成本”：

- `q = |q_table|`：该 CTA 内要处理的 query 数（乘上 HRatio 后等价于 group q 的规模）。
- `H = HRatio`：Q/KV head ratio（例如 nheads_q/nheads_kv）。
- `KV = kv_in_CTA`：该 CTA 处理的 KV token 数。

模型把 CTA 运行时粗略表示为：

- `original_cost(q, H, KV) = cost_a * q * H + cost_b * KV`

直觉：
- `q*H` 对应“Q 侧开销”（例如按 seq 数加载、softmax 相关、对每个 q 的固定工作）。
- `KV` 对应“KV 侧扫描/访存/计算量”。

该 cost 不是精确的 kernel 时间模型，更像是一个可比较的 proxy，用于：
- 判断“哪个 CTA 是 bottleneck（最大 cost）”；
- 判断“把一个 CTA 切成 k 份后，bottleneck 能否下降到值得切”。

---

## 3. split 候选与约束

对一个 box，考虑将其按 block 均匀切成 `k` 份：

- `split_box_even_blocks(src, k, out)`：按 block_table 均分；每份 `kv_in_CTA` 根据 block offset 重新计算。

硬约束（防止过碎）：
- `max_split_n`：最多切多少份（当前实现里是 16）。
- `min_blocks_per_piece`：每份至少多少 blocks（当前实现里是 4）。
- `min_kv_tokens_per_piece`：每份至少多少 token（用 `max(min_blocks_per_piece*block_size, batch_min_kv)` 估计）。
- `split_per_seq <= 32`：每条序列最多 split 32（避免 gather/writeback 管线过重）。

---

## 4. 成本模型的组分（net = cost - benefit）

对一个候选 (box, k)，定义：

- `added_cta = k - 1`
- `kv_piece = ceil(KV / k)`

模型计算一个净增成本 `net`，并选取使 `net` 最小且为负的 k（`net < 0` 认为值得 split）。

下面逐项说明 `net` 的组成（对应代码注释里的 (1)~(7)）：

### (1) Q duplication：`C_q_dup`

- 公式：`added_cta * cost_a * q * H`
- 含义：把一个 CTA 切成 k 份后，Q 侧固定工作会被重复 k 次。
- 为什么要算：对“小 KV 但 q 很大”的 CTA，盲目切 KV 只会重复 Q 工作，通常亏。

### (2) KV tile padding waste：`C_tile_pad`

- 关键中间量：
  - `BN = infer_BN(q, KV)`：根据 MNW bucket 规则推一个 N tile（16/32/64/128）。
  - `tiles_before = ceil(KV / BN)`
  - `tiles_after  = k * ceil(kv_piece / BN)`
  - `tile_delta   = max(0, tiles_after - tiles_before)`
- 公式：`cost_b * BN * tile_delta`
- 含义：切分后可能增加 tile 对齐浪费（padding waste），导致 KV 扫描的实际 tile 数变多。
- 为什么要算：KV kernel 很多时候按 tile 处理，切分会破坏“刚好整除”的情况。

### (3) Pipeline warmup：`C_warmup`

- 公式：`added_cta * (cost_startup_factor * cost_b * BN_new)`
- 含义：更多 CTA 会带来更多启动/流水 warmup 开销（非常粗糙）。
- 为什么要算：切分虽然能降低 tail，但可能让 wave 中 CTA 更碎、启动更频繁。

### (4) Wave scheduling：`C_wave`

- 思路：假设 GPU 同时可并行执行的“有效 CTA 数”与 SM 数相关。
- 中间量：
  - `denom_sm = N_SM (fallback to cta_limit)`
  - `total_before = base_cta * kvHead`
  - `total_after  = (base_cta + added_cta) * kvHead`
  - `waves_before = ceil(total_before / denom_sm)`
  - `waves_after  = ceil(total_after / denom_sm)`
  - `delta_waves  = max(0, waves_after - waves_before)`
- 公式：`C_wave = delta_waves * new_bottleneck`
  - `new_bottleneck = max(max_cost_others, split_cta_cost)`
- 含义：如果切分导致 wave 数增加，则总 makespan 可能反而上升。
- 为什么要算：当 CTA 数已经很多，继续增加 CTA 可能只是在增加 wave，没收益。

### (5) Gather kernel：`C_gather`

- `C_gather_incremental = q * kvHead * H * added_cta * cost_gather_per_split`
- `C_gather_launch = cost_kernel_launch`（仅在从“完全不 split”变成 split 时触发一次）
- 含义：切分通常需要 gather/merge 输出（或者至少有额外写回/合并逻辑），这里用线性项估。
- 为什么要算：split 的系统成本不止 attention kernel 本身。
- 曾支持 env 调参（现已固化/移除）：
  - `PAT_GATHER_PER_SPLIT_COST`
  - `PAT_GATHER_KERNEL_BASE_COST`（已移除，固定为 0）

### (6) Page table resolve：`C_page`

- 公式：`tile_delta * (cost_page_factor * cost_b * BN_new)`
- 含义：tile 数上升可能增加页表/索引解析成本（非常经验化）。

### (7) Benefit：`Benefit`

- 只对 bottleneck CTA 计算：`Benefit = old_cost - split_cta_cost`
- 含义：切分降低了最长 CTA 的 cost，从而降低 makespan。
- 为什么只对 bottleneck：模型近似假设总时间由最长 CTA 决定（tail latency 观点）。

最终：

- `net = C_q_dup + C_tile_pad + C_warmup + C_wave + C_gather + C_page - Benefit`
- 选择 `best_k` 使 net 最小，且 `net < 0` 才执行 split。

---

## 5. 这个模型为什么“这样估”？（设计动机）

它本质是在平衡两类力量：

1) **切分的收益**：把一个超长 KV 前缀 CTA 分摊到多个 CTA，降低 tail。
2) **切分的代价**：
   - Q 工作重复
   - tile 对齐/碎片化导致的额外 KV tile
   - CTA 数增加带来的 wave 增加
   - gather/merge 额外开销

采用线性/分段的近似，是为了：
- 计算快（schedule 端可在 CPU 上快速评估多种 k）
- 不依赖 kernel 内部复杂 profile
- 能用少量 env knob 快速调参

---

## 6. 主要局限与需要改进的地方

### 6.1 单位不统一、权重缺乏校准
`cost_a/cost_b/cost_startup_factor/cost_page_factor` 全是经验系数，没有与真实 GPU 时间拟合。

建议：
- 用少量离线 profiling 数据（不同 q/H/KV, 不同 split k）做回归，得到更稳健的系数；
- 或至少把各项打印出来（debug）方便校准。

### 6.2 BN(tile N) 推断过粗
`infer_BN` 只是复用了 MNW bucket 的启发式，和实际 kernel tile 可能不一致。

建议：
- 直接从 kernel traits（例如 head_dim、实际 kernel variant 的 N tile）导出 BN；
- 或把 BN 作为 schedule 输出桶的属性，而不是重新推断。

### 6.3 wave 模型过于简化
用 `total_cta * kvHead / N_SM` 推 wave 过粗：
- 没考虑 occupancy（寄存器/共享内存限制）
- 没考虑不同 CTA 代价差异导致的 wave tail

建议：
- 改成“分层”或“排序”模型：按 cost 排序模拟 wave 填充（LPT/greedy bin packing），估算 makespan。

### 6.4 gather 成本的形式不够贴近实现
真实 gather 可能：
- 只在 split 后需要，且与 split 层数/输出布局相关
- 不是严格线性

建议：
- 把 gather 模型与实际实现对齐：例如是否按 seq 做 reduction、是否需要额外 kernel。

### 6.5 只对 bottleneck CTA 计算 Benefit
很多情况下 split 可能改善的是“第二长/第三长”导致的整体更均衡（尤其当最大 CTA 有多个并列）。

建议：
- Benefit 变为对 makespan 的减少：`Benefit = makespan_before - makespan_after`（需要更强 wave 模拟）。

---

## 7. 一份可落地的重构方案（建议）

目标：让成本模型 **可维护、可校准、可解释**，并减少 cap 分支与全局 L_kv 分支的割裂。

### 7.1 代码结构建议
引入一个独立的成本模型结构体（伪代码）：

```cpp
struct SplitCostModel {
  // knobs (from env)
  double cost_a;
  double cost_b;
  double gather_per_split;
  double gather_launch;
  int N_SM;

  // compute primitive costs
  double box_cost(int q, int H, int KV) const;
  int infer_BN(int q, int H, int KV) const;

  // evaluate split option
  double eval_split_net(const Box& b, int k, const Context& ctx) const;

  // optionally: provide detailed breakdown for debug
  Breakdown explain_split(const Box& b, int k, const Context& ctx) const;
};
```

并把 `balancePackSota` 里的：
- 输入抽取（q/H/KV、blocks、split_per_seq limit）
- 候选生成（k 的范围、min piece 约束）
- 选择 best_k

拆成清晰的子函数，减少“巨函数 + lambda”形态。

### 7.2 把 makespan 估计从 max_cost 升级为 wave 模拟
实现一个轻量的 wave 模拟：

- 输入：每个 CTA 的 estimated cost（切分后要更新）
- 过程：用 `N_SM` 个槽做 greedy 分配（类似多机调度），返回完成时间（makespan）

这样 Benefit 就可以定义成：

- `Benefit = makespan_before - makespan_after`

更贴近真实。

### 7.3 增加 debug/trace 输出接口（不默认开启）
建议新增 env：
- `PAT_SOTA_COST_TRACE=1` 时输出：每个候选 box 的 old_cost、best_k、各项分解、net。

这样你在 RL 案例上就能快速看到“为什么没切/为什么切成 4 而不是 8”。

### 7.4 引入“校准模式”
- 提供一个小脚本（Python 或 C++）自动生成不同 (q,KV,k) 的 microcase，跑 kernel 取真实 ms，然后拟合 cost_a/cost_b/gather。

---

## 8. 快速检查清单（你调模型时建议先看）

- 是否误用 block_size 导致 padding 0 被读成真实 block？（本仓库 RLtest 已做严格校验）
- 该 case 是否落在 fast-skip 分支导致根本没进入 split 评估？
- split_per_seq 是否已经接近 32 导致无法再 split？
- min_blocks_per_piece / min_kv_tokens_per_piece 是否过大导致无法切？

---

如需把该成本模型进一步“定量化”，下一步建议是：在 RLtest 批量跑的同时，把 `best_k` 的各项 breakdown 输出到 JSONL，再结合 compare.svg 对照分析与调参。
