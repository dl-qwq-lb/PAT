## PAT balancePackSota: 固定拆分开销移除可行性与统一全局贪心路线

本文分析当前实现中“拆分固定开销(fixed split overhead)”机制是否可以整体移除, 以及是否能将 `base_cta` 与 `cta_limit` 的两分支合并为单一路径, 统一使用全局贪心(global greedy)策略。

文档写作风格参考 `flashinfer.md`: 先给出目标与符号, 再给出可比较的代价函数与算法, 最后给出工程路线与验证清单。

---

### 1. 结论摘要

1) 若“整体移除固定拆分开销”指 **完全不考虑拆分带来的额外系统成本**(Q duplication + gather/merge 等), 结论是: **不建议, 且大概率会回归当前收益**。

- 原因: 当前的贪心打分本质在最小化一个 runtime proxy。拆分会降低 KV 侧的单 CTA 上界成本, 但会引入不可忽略的额外代价, 这些代价在一些区间会压过 makespan 收益。
- 直接删掉 overhead 项会系统性倾向“过度拆分”, 典型症状是: `cur_max_split` 更快靠近 32、CTA 数明显膨胀、并行度并未等比例提升、甚至 gather 成为主导。

2) 若“移除固定开销机制”指 **去掉显式的 overhead 计算代码块, 但仍然在模型里隐式计入拆分代价**, 结论是: **可行, 且能显著简化代码**。

- 做法: 把 per-added-CTA 的 overhead 从“收益上扣除”改为“并入拆分后 CTA 的等效成本”。这样可以把目标写成统一的 makespan 变化比较, 同时保留抑制过拆分的效果。

3) “合并 base_cta 与 cta_limit 两分支, 统一全局贪心”在工程上 **可行**, 但需要明确一个关键细节: 

- 统一贪心必须同时支持两种资源语义:
  - `base_cta < cta_limit`: 存在硬预算, 不能超过 `cta_limit`。
  - `base_cta >= cta_limit`: 没有预算, 但仍需用 `max_actions / max_split_n / min_piece` 约束防止极端拆分。

这可以用“同一套 greedy 引擎 + 不同参数”实现, 从代码结构角度已经是单一路径。

4) 你提出的特殊规则: 当 `base_cta < cta_limit` 时“不计多一轮 makespan 开销”, 建议精确定义为:

- 在预算区间 `N in [base_cta, cta_limit]` 内, 计算 makespan 时固定 waves, 不因 N 增加而上升。
- 当 `N > cta_limit`(理论上预算模式不会发生)或在 `base_cta >= cta_limit` 的非预算模式, waves 使用正常定义。

该规则只在 `cta_limit > denom` 且 N 增加可能导致 `ceil(N/denom)` 跳变时才影响决策。在 A100/H100 等设备上, 由于 `denom = cta_cap(SM数)` 通常远大于 `cta_limit`(6/13/27/54), 该规则多数情况下等价于“不改变结果”。

---

### 2. 现状建模与符号

当前 `balancePackSota` 的核心对象为 PackedBox, 抽象成:

- 对每个 box b:
  - q(b): `|q_table|` (CTA 内 query 数)
  - kv(b): `kv_in_CTA` (CTA 内 KV token 数, 取 `max(0, kv)`)
  - blocks(b): `|block_table|` (KV blocks 数)
- HR = max(1, HRatio)
- N = 当前 CTA 数 = |B|
- denom = max(1, (cta_cap > 0 ? cta_cap : cta_limit))

单 CTA 代理成本(实现中的 `_bp_cost_model`)为:

  cost(q, kv; HR) = a*q*HR + b*kv + c*q*kv*HR

其中 a=0.1, b=0.1, c=2.0。

当前 makespan 代理为:

  waves(N) = ceil(N / denom)
  T_hat(B) = waves(|B|) * max_{b in B} cost(q(b), kv(b); HR)

---

### 3. 为什么“完全移除固定开销”会不稳

把一个 box b 拆成 k 份, 新增 added = k-1 个 CTA。拆分会带来两类真实代价:

- Q duplication: 原本 Q 侧只做 1 次, 现在重复 k 次。
- gather/merge 相关: 至少存在更复杂的写回、索引映射或额外 kernel(视实现), 通常与 q*HR 成正相关。

当前实现用一个简化的 per-added-CTA overhead 近似:

  overhead_per_added_cta(b) = (gamma_qdup + gamma_gqh) * (q(b)*HR)

其中 gamma_qdup = 0.25*a, gamma_gqh = 0.10*a。

如果把这一项整体移除, 贪心在很多 batch 上会更偏向选择更大的 k, 因为:

- 当 denom 很大(常见: denom=SM数), waves 往往不变, makespan_benefit 主要来自 max_cost 下降。
- cost 的主导项是 c*q*kv*HR, 对于较大的 kv, 把 kv 分摊能显著降低每份的 cost。
- 没有任何机制约束 “重复做 q 侧工作 + 聚合开销” 的增长, 只能依赖 min_piece 与 max_split_n, 很容易把 split 推到约束边界。

结果上, 你可能会观察到:

- schedule 输出的 CTA 数更大、split_per_seq 更高。
- kernel 时间并未相应下降, 甚至上升(聚合/访存/缓存失配成为瓶颈)。

因此: 在没有更强的端到端 cost 拟合或真实 gather 成本模型之前, 直接删 overhead 风险很高。

---

### 4. 可行的“移除机制”定义: 把 overhead 并入拆分后 CTA 成本

为了达到“代码更简单, 但效果不退化”的目的, 推荐把 overhead 从“gain 中扣除”改为“拆分后 CTA 的等效成本上浮”。

#### 4.1 定义等效成本

对候选拆分动作 (b, k), 令 added=k-1。

把拆分后最重的一段 KV 长度(实现里用对齐策略估计)记为 kv_worst(b,k)。

定义拆分后的候选瓶颈成本为:

  cost_split_eff(b,k) = cost(q(b), kv_worst(b,k); HR) + added * overhead_per_added_cta(b)

直觉: overhead 的单位被放进同一“成本”空间, 与 makespan 模型一致。

#### 4.2 统一的决策准则

令 max_other(b) 表示“除 b 外其他 CTA 的最大成本”(若 b 是唯一最大, 用 second_max, 否则仍用 max)。

定义:

  T_before = waves(N_before) * max_cost
  T_after  = waves(N_after)  * max( max_other(b), cost_split_eff(b,k) )

其中 N_after = N_before + added。

贪心收益为:

  gain(b,k) = T_before - T_after

这样就不需要单独写 `makespan_benefit - overhead` 两块, 逻辑更直观。

这属于“移除显式固定开销计算机制”: overhead 不再作为独立条目出现, 但仍被保留在等效成本中。

---

### 5. 统一全局贪心算法(单一路径)

本节给出一个“单一 greedy 引擎”的形式, 通过参数支持两种原语义。

#### 5.1 参数与模式

- N0 = base_cta = 初始 CTA 数
- L = cta_limit
- 模式参数:
  - budgeted = (N0 < L)
  - N_max = (budgeted ? L : INT_MAX)

说明:
- budgeted 模式下禁止超过 N_max=L。
- 非 budgeted 模式下可继续拆分, 但仍受 `max_actions / max_split_n / min_piece / split_per_seq<=32` 限制。

#### 5.2 “不计多一轮 makespan”的 waves 定义

按你的需求, 建议定义一个 budget-aware waves:

  waves_budget(N) = ceil( max(N, L) / denom )   if budgeted
                   ceil( N / denom )            otherwise

这样在预算区间内 waves 固定为 ceil(L/denom), 不会因为 N 从 N0 增加到 L 而上升。

注意: 若 denom >= L, 上式等价于 waves=1, 因此不会改变现有决策。

#### 5.3 统一 greedy(伪代码)

Input: boxes B, HR, denom, L, budgeted, N_max

1) 预计算 min_kv_tokens_per_piece(与当前一致)
2) N <- |B|
3) repeat up to max_actions:
4)   recompute max_cost, second_max_cost, max_count
5)   best_action <- none
6)   for each box b in B:
7)     determine feasible k candidates (k in {2,3,4,6,8,12,16} intersect constraints)
8)     for k in candidates:
9)       if budgeted and N + (k-1) > N_max: continue
10)      compute kv_worst(b,k) with current alignment policy
11)      compute cost_split_eff(b,k)
12)      T_before <- waves*(current) * max_cost
13)      T_after  <- waves*(after)   * max(max_other(b), cost_split_eff(b,k))
14)      gain <- T_before - T_after
15)      pick argmax gain
16)   if best_gain <= 0: break
17)   apply split action to B, update split_per_seq, N
18) return B

该算法同时覆盖:
- 原 @cap 分支(非 budgeted)
- 原非 @cap 分支(budgeted, N_max=L)

关键差异是: 原非 @cap 的 “global L_kv 预算拆分 + spare refine” 会变成 “纯 greedy”。

为了保收益, 推荐分两阶段迁移(见第 6 节路线)。

---

### 6. 工程技术路线(推荐分阶段, 可回滚)

#### Phase A: 先合并代码结构, 不改变策略语义(低风险)

目标: 把两分支抽象成同一个 greedy 引擎骨架, 但先保持非 @cap 的现有策略, 只做代码精简。

- A1. 抽出统一的工具函数:
  - `RecomputeCostStats(B)` -> (sum, max, second, count)
  - `MaxSplitLimitBySeq(b)`
  - `CanSplitK(b,k,min_piece)`
  - `EstimateKvWorstAfterSplit(b,k,min_piece)`
  - `ApplySplit(b,k)`
- A2. 统一成本模型接口:
  - 提供 `Cost(q,kv)`
  - 提供 `OverheadPerAddedCta(q)`
  - 提供 `SplitEffectiveCost(q,kv_worst,added)`
- A3. 把 @cap 分支改写为使用统一的 `gain = T_before - T_after` 形式(只改写表达式, 不改变系数/约束/候选集合)。
- A4. 非 @cap 分支保持 “global_L_kv + refine” 不动, 但复用 `ApplySplit` 与 piece 约束检查, 去掉重复逻辑。

验收标准:
- schedule_test 的输出与当前版本等价或差异可解释。
- RL smoke case 运行通过。

#### Phase B: 在 budgeted 模式引入 greedy(中风险, 建议加开关)

目标: 真正合并成单一路径, 非 @cap 从 “配额分配” 迁移到 greedy。

- B1. 新增 env 开关(默认关闭): `PAT_SOTA_UNIFIED_GREEDY=1`
- B2. 打开时, 非 @cap 走统一 greedy 且 `budgeted=true, N_max=L`。
- B3. 实现 `waves_budget(N)` 以满足“不计多一轮 makespan”规则。
- B4. 先限制动作集合为 k=2(只二分)以贴近现有 refine 行为, 再逐步放开到 {2,3,4,6,8,12,16}。

验收标准:
- 在 RL case 集合上, 平均不退化; 若有退化, 通过调整 overhead 或候选集回滚。

#### Phase C: 移除旧分支(高收益, 但最后做)

目标: 删除 `GlobalKVBudgetSplit + RefineWithSpareCTA` 的专用实现, 只保留 unified greedy。

前置条件:
- 已证明 unified greedy 在代表性 workload 上稳定不差。

---

### 7. 验证与回归检查清单

1) 正确性:
- `benchmark/schedule_test.py` 在多个 seed/配置下通过。
- `benchmark/benchmark_kernel.py` 至少对 1-3 个 RL case + 典型 head 配置跑通。

2) 行为边界:
- `split_per_seq` 不超过 32。
- `min_blocks_per_piece=4` 与 `min_kv_tokens_per_piece` 约束仍生效。
- budgeted 模式严格不超过 `cta_limit`。

3) 性能指标:
- 记录: 输出 CTA 数、max_split_per_seq、最大 box kv、估计 max_cost。
- kernel 时间: baseline vs sota vs unified-greedy。

4) 可回滚:
- Phase B 引入开关, 默认保持现状。

---

### 8. 如果你坚持“完全不计 overhead”, 最小化伤害的约束建议(不推荐)

若必须把 overhead 从模型里彻底删掉, 为降低过拆分风险, 至少需要:

- 限制候选只允许 k=2(二分), 禁止 3/4/6/8/12/16。
- 非 budgeted 模式强制 `N_after` 不允许增加 waves: 仅当 `ceil((N+added)/denom) == ceil(N/denom)` 才允许拆分。
- 引入更强的停止条件, 例如 `split_cta_cost` 必须降到 `<= avg_cost` 才允许。

这会显著改变当前策略, 且可能丢失一部分收益, 仅作为实验选项。
