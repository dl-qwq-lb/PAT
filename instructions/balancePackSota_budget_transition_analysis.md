# balancePackSota 预算阶段分配不均分析

## 结论

当前两个典型用例的分配不均，不是单一候选排序错误，而是预算阶段的约束语义过强：

1. `1,10_4096,416` 主要卡在 `L_KV` 预分割阈值被当成硬约束，导致已经有剩余 CTA 空位时，某些 chunk 仍然不能继续拆分。
2. `1,16_16384,128` 主要卡在 waves 感知的增益判别过于保守，导致在波次边界附近，贪心不愿意继续把 CTA 用满。

这两类问题都适合在当前全局贪心框架下做最小侵入修正，而不需要退回到削峰式策略。

## 现象对应的证据

- `1,10_4096,416` 的 SOTA 输出仍停留在明显少于理论可用 CTA 的状态，且 bucket 内 chunk 长度差异很大，说明预算阶段没有继续把空 CTA 吃满。参考 [benchmark/kernel_perf.json](benchmark/kernel_perf.json#L938) 和 [benchmark/kernel_perf.json](benchmark/kernel_perf.json#L1240)。
- `1,16_16384,128` 的 SOTA 也没有把可用 CTA 填到上限，`max_split_per_seq` 明显低于 baseline，说明 budget 阶段在 waves 跳变附近提前保守停止。参考 [benchmark/kernel_perf.json](benchmark/kernel_perf.json#L1374)。

## 原因分析

### 1. `1,10_4096,416` 为什么在 `L_KV` 达到后仍不继续分割

当前实现中，budget 阶段会先计算一个全局目标：

- `global_L_kv = ceil(total_kv_all / cta_limit)`
- 然后做 block 对齐，得到 `global_L_kv_aligned`
- 再把它作为 budget 阶段 piece 的硬下限

相关位置见 [csrc/prefix_tree.h](csrc/prefix_tree.h#L791) 到 [csrc/prefix_tree.h](csrc/prefix_tree.h#L797)。

问题在于，这个下限是“全局平均值”，不是“当前剩余槽位下的局部目标”，更不是软约束。于是像 `416 -> 2 * 208` 这种拆分，即使能够显著消耗空 CTA，也会因为 `208 < global_L_kv_aligned` 直接被拒绝。

也就是说，当前代码把 `L_KV` 从“预分割参考”变成了“硬 veto”，这会直接导致 budget 阶段在还有空 CTA 时停住。

### 2. `1,16_16384,128` 为什么在 waves 跳变附近过于保守

当前增益判断仍然是典型的波次感知比较：

- 先估计 `waves_before`
- 再估计 `waves_after`
- 用 `est_time_before - est_time_after` 作为 gain

同时 budget 阶段还会对 `new_max_cost` 做额外保护，避免分裂后带来更大的 max cost。

相关位置见 [csrc/prefix_tree.h](csrc/prefix_tree.h#L803) 到 [csrc/prefix_tree.h](csrc/prefix_tree.h#L914)。

这套逻辑的问题是：一旦候选分裂在波次边界附近看起来“不够赚”，它就会被当成不可接受。对于 `1,16_16384,128` 这类长序列场景，这种保守性会让算法在 CTA 还没用满时就停止继续拆分。

更具体地说，当前算法把“不要增加波次”近似成“尽量不要让候选带来任何额外风险”，这会把本来应该用于补齐 CTA 的动作挡掉。

## 最小侵入、最自然的修正方式

### 方案 A：把 `L_KV` 从硬门槛改成软参考

这是最自然的替代方式，也是对现有框架最小侵入的方式。

建议做法：

1. 保留全局 `L_KV`，但只把它当作预算阶段的软目标，而不是硬约束。
2. budget 阶段的硬约束只保留 block 下限，例如每个 piece 至少若干 block。
3. 当候选 piece 小于 `L_KV` 时，不直接拒绝，而是给 gain 增加一个轻量惩罚，让贪心自己权衡是否值得继续拆。

这样做的好处是：

- `1,10_4096,416` 这类 case 可以继续拆出更多 CTA，不会被全局平均值卡死。
- 仍然保留 `L_KV` 的引导作用，不会退化成纯盲拆。

当前代码已经按这个方向调整：

- `L_KV` 只作为 `budget_kv_target`
- budget 阶段 piece 的硬下限改成 block floor
- 软惩罚通过 `soft_kv_penalty` 进入 gain

见 [csrc/prefix_tree.h](csrc/prefix_tree.h#L796) 到 [csrc/prefix_tree.h](csrc/prefix_tree.h#L914)。

### 方案 B：budget 阶段的 `k` 上限必须受剩余空闲 CTA 约束

这个约束是必要的，因为它能减少无效候选搜索，并让贪心优先尝试“刚好补位”的动作。

建议做法：

- `max_k = min(max_k, 1 + remaining_slots)`
- 其中 `remaining_slots = cta_limit - total_cta`

这样可以避免候选动作一次性把剩余 CTA 估过头，也避免过多无效尝试。

当前实现已包含这条裁剪：见 [csrc/prefix_tree.h](csrc/prefix_tree.h#L885) 到 [csrc/prefix_tree.h](csrc/prefix_tree.h#L886)。

### 方案 C：budget 填满后不要退出循环，而是切换到非 budget 再继续全局贪心

这是解决 `1,16_16384,128` 的关键。

建议做法：

1. budget_phase 下优先把 CTA 补满。
2. 一旦 `total_cta >= cta_limit`，不要 `break`。
3. 把阶段切换成 non-budget，继续沿用同一套全局贪心。
4. non-budget 阶段再恢复更严格的削峰判据。

这样做的好处是：

- 不会把“填满 CTA”与“后续均衡优化”割裂成两套互相独立的方法。
- 保持单一贪心主循环，状态连续。
- 对长序列 case 更自然。

当前实现已经采用这种阶段切换方式：见 [csrc/prefix_tree.h](csrc/prefix_tree.h#L975)。

## 为什么不建议把 budget 结果再丢进另一套方法重跑

这个思路能做，但不是最自然的过渡。

问题在于：

1. 两套方法的目标函数不同，一个是“补 CTA”，一个是“削峰”。
2. 第二次重跑会覆盖第一次结果，可能把已经填好的 CTA 又改回去。
3. 状态会被拆成两段，后续调参和分析都更难解释。

相比之下，单循环里做阶段切换，保留的是同一个全局贪心框架，更稳也更容易维护。

## 当前代码已做的对应修改

1. 抽出统一 CTA 上限 helper：`_bp_cta_limit_by_kvhead`。见 [csrc/prefix_tree.h](csrc/prefix_tree.h#L639)。
2. 引入 `L_KV` 的软参考 `budget_kv_target`，不再作为 budget 阶段硬门槛。见 [csrc/prefix_tree.h](csrc/prefix_tree.h#L796) 到 [csrc/prefix_tree.h](csrc/prefix_tree.h#L797)。
3. budget 阶段的 `k` 上限受剩余 CTA 槽位裁剪。见 [csrc/prefix_tree.h](csrc/prefix_tree.h#L885) 到 [csrc/prefix_tree.h](csrc/prefix_tree.h#L886)。
4. budget 填满后切换到 non-budget，不提前退出。见 [csrc/prefix_tree.h](csrc/prefix_tree.h#L975)。

## 结论

- `1,10_4096,416` 的不均衡，核心是 `L_KV` 被硬化后，剩余 CTA 无法继续被拆分吃满。
- `1,16_16384,128` 的不均衡，核心是 waves 临界点的收益判别过保守，导致 CTA 没有继续被利用。
- 最小侵入、最自然的修正方式，就是保留全局贪心框架，把 `L_KV` 改成软参考，并在 budget 阶段用剩余 CTA 槽位驱动候选裁剪，填满后切换到 non-budget 继续贪心。
