# balancePackSota 设计（基于 PAT 约束的新版方案）

本文件在重新阅读 FlashInfer 论文（3.3 节）并结合 `kernel_perf.json` vs `kernel_perf_baseline.json` 的实验结果后，**完全重写** 了对 `balancePackSota` 的设计说明：

- 聚焦 PAT 自身的前缀打包与 kernel 约束，而不是简单“照搬 FlashInfer”。
- 明确 correctness 不变量与可/不可做的拆分类型。
- 解释当前场景下 CTA 数量与调度能力的边界。
- 给出一版更合理、可实现的 `balancePackSota` 设计草案，作为后续 C++ 实现的规范。

---

## 1. 正确性重新评议：从 FlashInfer 回到 PAT

FlashInfer 的 Algorithm 1 核心思想是：

- 把注意力工作拆成细粒度的 **(query tile × KV chunk)** 任务；
- 用简单线性 cost 模型 `cost = α · l_q + β · l_kv` 估算每个任务代价；
- 通过 LPT（最长任务优先）+ 最小堆贪心，将任务均匀分配给固定数量的 CTA；
- 同一 CTA 可以**顺序执行多个任务**，并在 contraction 阶段聚合部分结果。

在 PAT 中，现有 kernel 和数据结构有几个关键差异：

1. **不支持 split-Q**  
     - 每个 `PackedBox` 对应一个 CTA，一次性处理一组 query：`q_table`。  
     - kernel 假设：一个 CTA 内所有 query 共享同一段 KV 覆盖（同一个 `block_table_ptr` 子数组），并在 CTA 内完成所有 softmax/归约。  
     - 没有“将同一条 query 切成多个 query tile 分别给不同 CTA 再合并”的机制，因此 **不能像 FlashInfer 那样对 query 维度做拆分**。

2. **只能做 split-KV，且粒度受结构限制**  
     - 我们可以把一个 `PackedBox` 的 KV 区间（由 `block_table_ptr` 指定）切成若干连续的子区间，每个子区间对应一个新的 `PackedBox`；  
     - 这些子盒的 `q_table` 必须与原盒完全一致（不能改变 query 集合，只能改变它看到的 KV 段）；  
     - 每个子盒的 `block_table_ptr` 必须是原 `block_table_ptr` 的**连续切片**，不能把两个不相邻的 KV 段“拼接”进同一个 `PackedBox`——否则会破坏 kernel 中对 block_table 的线性访问假设。

3. **一个 CTA 只能处理一个 `PackedBox`，不能在单次 kernel 启动里“串行执行多任务”**  
     - 目前 PAT 的 kernel 不是 FlashInfer 那种 persistent kernel + work queue 结构，而是一次性 launch：`gridDim.x = num_boxes`；  
     - 每个 CTA 只拿到自己的 `PackedBox` 元信息，在 kernel 内只做这一份工作；
     - 想让一个 CTA 顺序处理多个 KV chunk，需要**重写 kernel**（引入内部任务队列和循环），这超出 `balancePackSota` 能控制的范围。

4. **CTA 数量有上限但不是当前瓶颈**  
     - 硬件层面有 `gridDim.x` 的上限（远大于当前使用规模），更关键的是 **并发 resident CTA 数 + 元数据内存**；  
     - 实践中，我们可以让 `num_boxes`（也就是 CTA 数）略大于 `SM × 每 SM 最大 CTA 数`，多余 CTA 会排队，并不会导致 correctness 问题，只是并行度饱和后收益趋于递减；  
     - 因此，在 PAT 当前规模和 GPU 条件下，可以把 **“合理增大 CTA 数以平衡负载”视为允许的调度空间**，但必须通过提前拆分出足够多的 `PackedBox` 来实现。

**结论（correctness 层面）**：

- 正确的 SOTA 调度策略必须满足：
    1. **query 维度绝不能拆分**：每个 baseline 盒子的 `q_table` 在 SOTA 下保持完全一致；  
    2. **只在 KV 维度拆分，且每个子盒的 KV 区间是原区间的连续子区间**；  
    3. **对每条序列，所有参与它的 CTA 要保持严格有序且可数的 `CTA_rank` / `split_per_seq`**：
         - `CTA_rank` 单调递增，
         - `split_per_seq[seq_id]` 等于该序列被拆成的 CTA 数；
    4. **所有 KV token 覆盖完整且无重叠**：对每条序列，所有子盒 KV 区间的并集等于 baseline 盒子的 KV 区间，且相互不重叠。

- “保护长公共前缀不拆分”**不是 correctness 必需条件**，只是一个性能启发式：
    - 早期我们为了尽量保留前缀复用，曾经对 `kv_in_CTA` 很长且 `|q_table|` 较大的盒子设置“强保护不拆分”；  
    - 从 correctness 角度，只要满足上面的 4 条，即便拆分长前缀也不会出错；  
    - 从实验结果看，这种“绝对保护”在部分长前缀 tree 上会导致 CTA 负载严重不均，反而拖慢整体。

---

## 2. 现有 balancePackSota 实验评估小结

基于 `kernel_perf.json`（视为 SOTA 调度结果）与 `kernel_perf_baseline.json`（baseline 调度）对齐分析的结论：

- 在 67 组可对齐配置上：
    - SOTA 更快（`pat_sota / pat_base < 0.98`）：8 组；
    - SOTA 更慢（`> 1.02`）：20 组；
    - 其余约 39 组在 ±2% 内基本打平；
    - 全局平均比值 `pat_sota / pat_base ≈ 1.022`：**SOTA 平均略慢约 2%**。

- 按 tree 统计：
    - 一些中长 KV、非极端长前缀的 tree 上 SOTA 有几乎 1% 左右的小幅加速；
    - 但在长前缀和复杂结构的 tree 上，例如：
        - `1,10_4096,416`：SOTA 平均慢 ~5%；
        - `256_4096`：SOTA 慢 ~1.5%；
        - `4,16,256,512_512,32,128,32`：SOTA 慢 ~15%。

**说明**：

- correctness 方面：
    - 新版 light SOTA 相比旧激进 SOTA 已经解决 correctness FAILED 问题，PAT 在 `kernel_perf_baseline.json` 中均为 PASSED；
    - 说明我们当前对 KV 拆分和 `CTA_rank` 维护在语义上是正确的。

- 性能方面：
    - 由于不能像 FlashInfer 那样让一个 CTA 依次处理多个不相邻的 KV chunk，现有 SOTA 只能通过“离线拆分更多盒子 + 静态分配 CTA”来逼近负载均衡；
    - 在 baseline 本身已经比较均衡的 tree 上，额外拆分只增加元数据和调度开销，收益有限甚至略有回退；
    - 在长前缀 tree 上，“长前缀强保护 + 简单 split-KV” 的组合没有找到最优点：前缀保护导致少数超大盒子无法缓解，而拆分策略又不足以把尾部不共享 KV 切得更合理。

**结论（设计层面）**：

- 现有 light 版 `balancePackSota` 在 correctness 上是可信的，但在性能上 **没有显著优于 baseline 的证据，甚至平均略差**。  
- 后续设计应以“在不退化整体性能、不破坏 correctness 的前提下，局部提升不均衡 tree 的表现”为目标，而不是追求过于复杂的全局调度。

---

## 3. 新版 balancePackSota 设计目标与约束

### 3.1 设计目标

1. **严格遵守 correctness 不变量**（见第 1 节）。
2. 在大部分配置上 **不显著弱于 baseline**，尤其是复杂或长前缀 tree 不出现明显退化；
3. 在以下场景争取可见收益：
     - KV 明显长、但前缀共享并不极端的 tree；
     - baseline 中存在极大盒子拖慢整体的情况（明显的 tail-CTAs）。
4. 尽量 **不修改 kernel 接口与数据结构**：
     - 不改 `PackedBox` / `KernelInfo` 结构；
     - 不引入 persistent kernel 或 GPU 侧任务队列；
     - 全部逻辑在 C++ host 侧的 `balancePackSota` 内完成。

### 3.2 约束重申（回应几个具体问题）

1. **“长公共前缀保护策略是否有问题？”**  
     - 从 correctness 看：保护/不保护都不会影响数学正确性，只要只拆 KV 且保持 query 数组与 baseline 一致；  
     - 从性能看：
         - 过于绝对的“长前缀一律不拆”会在 `1,10_4096,416` 等 tree 上导致严重 tail，SOTA 慢于 baseline；
         - 更合理的策略是：
             - **不把“保护长前缀”当作硬规则，而是一个调节拆分力度的软约束**；
             - 比如对多 query + 长 KV 盒子，只允许有限次数、较粗粒度的 KV 拆分，而不是完全禁止拆分，也不是像旧 SOTA 那样激进均匀切碎。

2. **“正确的分割策略是否必须只拆 KV，且保持 query 数组与 baseline 一致？”**  
     - 是的，这在当前 PAT 内核下应被视作 **硬性 correctness 约束**：
         - 不允许改变 `q_table` 的成员集合；
         - 不允许让同一条 query 的不同时段落在不同 CTA，然后再在 kernel 外合并；  
         - 唯一允许的自由度是在 KV 维度上，把原本统一的 KV 覆盖区间切成若干连续子区间，分别交给不同 CTA 处理；
         - 这是新版设计必须坚持的一条底线。

3. **“CTA 数量是否有上限？能否贪心均衡不同 KV 任务，让一个 CTA 先后执行不同任务？”**  
     - CTA 数量：
         - 硬件有 `gridDim.x` 的形式上限，但远大于我们需要；
         - 实际更关心的是：当 CTA 数远大于 `SM × 每 SM 最大 CTA` 时，再增加 CTA 只会增加排队，不带来明显收益，但 correctness 不受影响；
         - 因此，在 SOTA 设计中可以把“目标 CTA 数”当作一个可调参数，用来控制拆分粒度，而不用过于担心理论上限。
     - 单 CTA 多任务：
         - 现有 PAT kernel 是“一次性 launch，每个 CTA 绑定一个 `PackedBox`”，执行完就结束，没有内建任务循环；
         - 要让一个 CTA 先后执行多个不同 KV 任务，需要改成 persistent kernel + GPU 端任务队列，与当前实现差别很大；
         - 在不改 kernel 的前提下，**我们无法像 FlashInfer 那样让 CTA 自己从队列中拉取多个任务执行**，只能通过在 host 侧预先拆出足够多的 `PackedBox`，静态映射到 CTA。

---

## 4. 新版 balancePackSota：算法框架

在上述约束下，新版 `balancePackSota` 采用两级思路：

1. 以 baseline 盒子（`_tree_heuristics` + 原始 `balancePack` 之后的结果）为“参考正确状态”；
2. 在 **仅拆 KV、不动 query** 的前提下，进一步对部分“重盒子”做更合理的 KV 细分，以增加 CTA 数量、缓解 tail；
3. 使用简单的 cost 模型和统计量来决定“拆不拆、拆几份”，不做完整的全局任务队列调度。

### 4.1 输入/输出

- 输入：baseline 得到的一批 `PackedBox`：`boxes_baseline`；
- 输出：SOTA 版 `PackedBox`：`boxes_sota`，满足：
    - 对于每个 baseline 盒子 B：
        - `⋃ KV_range(子盒) = KV_range(B)` 且互不重叠；
        - 每个子盒的 `q_table` 与 B 完全一致；
    - `boxes_sota` 经 `KernelInfo` 打平后可以直接传入现有 kernel，无需修改 kernel 代码。

### 4.2 关键参数与统计量

1. **盒子成本**  
     对每个 baseline 盒子 B 定义：

     $$
     	ext{cost}(B) = |q\_table(B)| + \frac{\text{kv\_in\_CTA}(B)}{H\_{ratio}}
     $$

     - `|q_table|` 近似反映该盒子中 query 的数量（注意 decode 场景下一般是 batch 大小）；
     - `kv_in_CTA / Hratio` 近似反映每个 query 需要遍历的 KV 量；
     - 这与 FlashInfer cost 模型精神一致，只是去掉了显式 `α, β`，用简单归一化代替。

2. **tree 级统计**  
     - 总成本：`C_total = sum_B cost(B)`；
     - 总“KV 工作量”：`KV_total = sum_B kv_in_CTA(B) * |q_table(B)| / Hratio`；
     - kv_len 分布的中位数、p90 等，用于识别“异常重”的盒子。

3. **目标 CTA 数**  
     - baseline 已经有一套 `threshold_cnt` / `target_CTA` 逻辑，我们可以简单复用（例如 `target_CTA = max(baseline_cta, min_CTA)`）；
     - SOTA 只在 `target_CTA` 附近微调，而不是随意放大。

### 4.3 拆分规则（仅 KV，移除“绝对长前缀保护”）

对每个 baseline 盒子 B：

1. 计算其“期望 split 数”：

     - 设 `L_target = KV_total / target_CTA / avg_q`, 其中 `avg_q` 为平均 `|q_table|`；  
     - 或更简单地：`L_target = p50_kv` 或 `p75_kv`，即 kv_len 分布的一个分位数；
     - 则：`split_k_raw = ceil(kv_in_CTA(B) / L_target)`。

2. 应用平滑与限制：

     - 若 `split_k_raw <= 1`，则不拆；
     - 若 `split_k_raw > K_max`，仅取 `K_max`（例如 4）；
     - 为了兼顾“长前缀 + 多 query”的场景，可以引入一个软约束：
         - 若 `|q_table(B)| >= Q_large` 且 `kv_in_CTA(B)` 特别大，则强制 `split_k = min(split_k_raw, K_long_prefix)`（例如 2），**不再采用“完全不拆”** 的硬限制。

3. 根据 `split_k` 做 KV 均匀切分：

     - 将 `block_table_ptr` 按 block 数均匀划分为 `split_k` 段，每段对应一段连续 KV 范围；
     - 为每段新建一个子盒 `B_i`：
         - `q_table(B_i) = q_table(B)`；
         - `block_table_ptr(B_i) = slice(block_table_ptr(B), i)`；
         - `kv_in_CTA(B_i)` 精确为该 slice 覆盖的 token 数；
         - `num_seqs_per_CTA` 沿用原值；
         - `CTA_rank` / `split_per_seq` 按照“同一条序列的 CTA 依次递增”的规则更新（可直接仿照现有实现）。

4. 生成所有子盒集合 `boxes_splitted` 作为 SOTA 盒集合。

> 与之前版本相比，这里最大的变化是：
> - 不再对“长前缀 + 多 query”强制禁止拆分，而是只限制最大拆分次数；
> - 所有盒子都在同一套规则下由 `kv_in_CTA` + 分布统计共同决定是否拆分，避免个别 tree 出现极端 tail。

### 4.4 简单的贪心均衡（可选）

在仅靠 above 规则拆分后，如果 `boxes_splitted` 的 kv_len 分布仍然极不均匀，可以再做一层轻量的“盒子排序 + 贪心挑 CTA”：

1. 将 `boxes_splitted` 按 `kv_in_CTA` 降序排序；
2. 维护一个长度为 `target_CTA` 的数组 `load[cta]`，初始为 0；
3. 依次取最长盒子，丢给当前 `load` 最小的 CTA：`cta = argmin(load)`，并累加 `load[cta] += kv_in_CTA(B)`；
4. 最终得到“每个 CTA 负责哪些盒子”的分组信息；
5. **不改变盒子本身**，也不合并不同 KV 段到一个盒子，只在需要统计/调试时使用该分组（例如观察是否还有严重 tail）。

在当前 PAT 实现中，这一步更多是分析/调参工具，不必强行编码进 `KernelInfo`，因为 kernel 本身并不会感知 CTA 对应的“负载桶”。

---

## 5. 长前缀 KV 进一步细分的可行性讨论

结合上面的正确性约束和实验结果，可以明确几点：

1. **从 correctness 角度，细分长前缀 KV 是完全可行的**：
     - 只要不改变 `q_table`，不跨序列混合 KV，并且保持 `CTA_rank` / `split_per_seq` 一致，拆分多少段都不会改变数学结果；
     - 旧 SOTA 出错是因为实现上的边界 bug（KV 覆盖/ rank 处理错），而非“长前缀不能拆”的理论限制。

2. **从性能角度，理想的做法是“前缀感知的 tail-only 拆分”**：
     - 真正高价值的是被大量 query 共享的前缀段，它们确实不宜被随意切碎分摊到太多 CTA 上；
     - 但很多长前缀 tree 的 KV 区间末尾存在较长的、只被少数 leaf 使用的尾巴（tail），这部分如果完全不拆，就会制造 CTA tail；
     - 如果我们在 `PackedBox` 中能拿到“前缀长度 vs tail 长度”的更细粒度信息，就可以：
         - 固定前缀段由少数 CTA 整块处理；
         - 仅对 tail 部分做更细力度的 KV split，从而兼顾前缀复用和尾部并行度。

3. **在当前数据结构下，难以做到真正 prefix-aware，只能用近似规则**：
     - 目前 `PackedBox` 只暴露 `kv_in_CTA`、`num_seqs_per_CTA`、`q_table` 等汇总信息，没有直接告诉我们“多少 KV 是高共享的前缀，多少是低共享的 tail”；
     - 若要实现 prefix-aware tail splitting，需要：
         - 在构建 `PackedBox` 的阶段（`_tree_heuristics`）就记录更多结构信息（例如每条序列的前缀长度）；
         - 或者修改 `PackedBox` / `KernelInfo` 结构，这会放大改动范围。

4. **折中方案**：
     - 在不改数据结构的前提下，采用“**软保护 + 有上限拆分**”的策略（见 4.3），尽量减少极端 tail 情况：
         - 让“长前缀 + 多 query”盒子仍然可以拆 2 段左右，缓解最严重的尾巴；
         - 同时通过全局 `L_target` 控制整体拆分粒度，避免旧 SOTA 那种过度碎片化；
     - 后续如果实验表明仍无法拉平长前缀 tree 的尾部，可以再考虑在 `_tree_heuristics` 中增加 prefix/tail 标记，设计真正的 tail-only splitting。

---

## 6. 小结

1. 回到 FlashInfer 论文重新审视后，可以确认：
     - PAT 现有 kernel 无法直接实现 FlashInfer 的 split-Q 和“单 CTA 多任务队列”，这也是 `balancePackSota` 设计空间的硬约束；
     - 正确的调度策略必须 **只拆 KV、不拆 query**，并保持 `CTA_rank` / `split_per_seq` 与 baseline 一致，这也是 correctness 的根本。

2. 现有 light SOTA 实验显示：
     - correctness 是 OK 的；
     - 性能整体略逊于 baseline，特别是在长前缀与结构复杂的 tree 上，这与“绝对长前缀保护 + 粗糙拆分”有关。

3. 新版设计主张：
     - 取消“长前缀绝对保护”，改为统一基于 `kv_in_CTA` 分布和简单 cost 模型决定拆分次数，并对多 query + 长 KV 盒子施加**拆分次数上限**而非禁拆；
     - 只在 KV 维度做等分拆分，保持 query 数组与 baseline 一致，并严格维护 `CTA_rank` / `split_per_seq`；
     - 通过 tree 级统计与适度阈值控制，力求在“不比 baseline 差太多”的基础上，为部分不均衡 tree 提供可见收益。

4. 后续工作建议：
     - 按本设计在 C++ 中整理/精简 `balancePackSota` 实现，使之与文档行为一致；
     - 对长前缀 tree（如 `1,10_4096,416`, `256_4096` 等）做针对性 micro-benchmark，验证“软保护 + 有上限拆分”是否能消除当前 ~5% 左右的退化；
     - 根据评估结果再决定是否需要在 `_tree_heuristics` 阶段引入 prefix/tail 标记，实现真正的 prefix-aware tail-only splitting。

---

## 7. 基于 KV 成本的贪心二分拆分策略评估

这一节专门讨论一种更“激进但仍受控”的 SOTA 策略，并分析其可行性与相对 `balancePack` 的潜在改进空间：

- 思路：
     - 完全放弃对“长前缀 vs 尾巴”的显式区分；
     - 把每个 `PackedBox` 看成一个任务，按 **KV 成本**（或综合 cost）排序；
     - 始终对当前 **代价最大的盒子做二分拆分**：
          - 把它的 KV 区间按 block 均匀一分为二，生成两个子盒；
     - 重复此过程，直到：
          - 盒子总数达到了预设的 CTA 上限（或目标 CTA 数）；或
          - 所有盒子的成本都低于某个阈值。
     - 整个过程中只做“再次二分”，不会像旧 SOTA 那样把 KV 彻底打碎成很多小块。

### 7.1 算法框架（概念性描述）

1. **初始化**：
      - 从 baseline 得到初始盒集合 `boxes`；
      - 为每个盒子定义成本：

           $$
           	ext{cost}(B) = |q\_table(B)| + \frac{\text{kv\_in\_CTA}(B)}{H\_{ratio}}
           $$

           或者在 decode 场景下近似为 `cost(B) ≈ kv_in_CTA(B)`（因为 `|q_table|` 变化不大）。

2. **优先队列**：
      - 构造最大堆 `PQ`，按 `cost(B)` 从大到小存放所有盒子；
      - 记录当前盒子数 `num_boxes = |boxes|`。

3. **迭代二分**：

      - 设 `CTA_target` 为允许的最大 CTA 数（可基于 baseline_cta 放宽一定倍数，如 `CTA_target = max(baseline_cta, min_CTA)`）；
      - 循环条件：
           - `num_boxes < CTA_target`；
           - 且 `PQ.top().cost > cost_threshold`（可选，例如大于全局中位数或 p75）。

      在每步迭代中：

      1. 从 `PQ` 取出成本最大的盒子 `B_max`；
      2. 若 `B_max` 的 block 数为 1，则无法再二分，跳过（或直接终止）；
      3. 将 `B_max` 的 `block_table_ptr` 按 block 个数均分为两段（或尽量接近均分）：
               - 左段 `[0, mid)`，右段 `[mid, end)`；
      4. 生成两个子盒 `B_left`, `B_right`：
               - `q_table` 原样拷贝；
               - `block_table_ptr` 指向对应连续 KV 段；
               - 精确更新 `kv_in_CTA`；
               - `num_seqs_per_CTA` 沿用；
               - 按照“同一序列 CTA_rank 递增”的规则，给两个子盒分配新的 `CTA_rank`，更新 `split_per_seq`；
      5. 将 `B_left`, `B_right` 的成本重新计算并入堆；
      6. `num_boxes` 加 1（原 B_max 变成两个盒子，总数 +1）。

4. **结束条件与输出**：
      - 当 `num_boxes` 达到 `CTA_target` 或堆顶成本不再“大到值得拆分”时停止；
      - 堆中所有盒子即为 SOTA 版本 `boxes_sota`。

> 关键点：
> - 永远只处理当前“最重”的盒子，确保每一步拆分都尽可能减小 overall tail；
> - 每次只二分一次，不继续把子盒递归拆到极小，避免 KV 被打碎成大量 tiny 盒子；
> - 完全不显式区分“前缀段”和“尾巴”，只看成本本身。

### 7.2 正确性与约束符合性

这个贪心二分策略在 correctness 上是 **完全可行的**，理由如下：

1. **只拆 KV，不动 query**：
      - 每次拆分只改 `block_table_ptr`，`q_table` 完整复制；
      - 保证每个子盒处理的 query 集合和 baseline 保持一致。

2. **连续 KV 区间**：
      - 按 block 索引做 `[0, mid)` / `[mid, end)` 等连续切片；
      - 对每条序列而言，KV token 的覆盖仍是原区间的划分，而非交错拼接。

3. **CTA_rank / split_per_seq 可按现有逻辑维护**：
      - 每次拆出新的子盒，就为其中涉及的序列追加一个新的 rank，保证单调递增；
      - `split_per_seq[seq_id]` 始终等于该序列被拆成的 CTA 数；
      - 这套逻辑已经在现有 light SOTA 中跑通，只是现有实现的拆分触发规则不同。

4. **CTA 数量受控**：
      - 通过 `CTA_target` 控制最大拆分次数，每次拆分 `num_boxes += 1`；
      - 不会无限膨胀，也不会超出我们能接受的 `gridDim.x` 范围。

因此，从“是否会破坏数学正确性”的角度，这种“按 KV 成本贪心二分直到 CTA 上限”的策略是安全的。

### 7.3 与当前“软保护 + 有上限拆分”方案的关系

两者的差别主要在于 **拆分触发策略和优先级**：

- 当前文档第 4 节的方案：
     - 先基于 `kv_in_CTA` 分布估算一个 `L_target`；
     - 对每个盒子独立算 `split_k_raw = ceil(kv_in_CTA / L_target)`，再裁剪到 `[1, K_max]`；
     - 所有盒子一起跑完这个规则，得到 `boxes_splitted`；
     - 本质是“**按各自 KV 长度一次性决定拆几份**”。

- 本节贪心二分方案：
     - 不提前给每个盒子定 `split_k`，只要还没到 CTA 上限，就一直找当前最大盒子拆一刀；
     - 本质是“**逐步平衡：每次把当前最重的盒子切成两个，直到够多/够均匀**”。

优劣对比（理论上）：

- 贪心二分的优点：
     - 更直接针对 tail 问题：每一步都在缩小最长盒子的 cost，有点类似多机调度里的 LPT + longest-job-splitting；
     - 不需要精确挑 `L_target`，即使 KV 分布很偏，也能自适应多拆几刀；
     - 每个盒子最多被拆成 `O(log(kv_len))` 个子盒（如果设置“只拆一次”则更少），不会出现被均匀切成很多 tiny 盒的极端情况。

- 现方案的优点：
     - 规则更简单：看 `kv_in_CTA` 和全局统计值即可决定拆分次数，易于推理和调参；
     - 若 `L_target` 选得合理，对大部分盒子都能给出较合适的 split_k，展开一次就结束，避免迭代堆操作。

从“接口和约束”角度看，两者都是合法的 SOTA 策略，只是设计风格不同。

### 7.4 是否要显式区分“长共享前缀”和“尾巴”？

你提出的设想是：

> “可不可以对长共享序列和尾巴进行平等拆分，只看谁的 kv 最长或者成本最大，是整个 batch 时间的决定因素？”

在当前数据结构下：

- **只看盒子的 `kv_in_CTA` / cost，而不尝试显式识别 prefix vs tail，是一种合理且实现简单的选择**：
     - 我们无法从 `PackedBox` 里直接知道哪一部分 KV 是“高度共享的前缀”，哪一部分是“尾巴”；
     - 在这种情况下，过强的“前缀保护”往往是拍脑袋的启发式，可能导致长前缀 tree 的 tail 无法被缓解；
     - `kernel_perf` 分析已经表明，以“绝对保护长前缀”为核心的策略在某些 tree 上会让 SOTA 慢于 baseline。

- 贪心二分方案等于是说：
     - 不再试图在没有足够信息的前提下硬区分 prefix/tail；
     - 只看 cost，谁拖慢整体就优先被切一刀；
     - 这和“所有盒子在调度视角上一律平等，只按负载决定命运”的思想是一致的。

因此，在现有信息不够 prefix-aware 的前提下，**用 cost 驱动的均衡通常比拍脑袋的 prefix/tail 启发式更稳健**。

### 7.5 避免“tail-only 决策”的动机与影响

你还担心的一点是：

> “不要将 kv 彻底打碎，只是对 kv 进行一次又一次的二分，避免 tail_only 决策？”

可以这样理解：

- 之前讨论的 prefix-aware tail-only 拆分，本质上是“只拆最后那段看起来像 tail 的部分”；
- 若 tail 区域过长，光拆尾巴而不动中间部分，未必能把负载完全拉平；
- 而且在缺乏明确 prefix/tail 边界信息的情况下，“哪里是 tail”本身就很难界定。

贪心二分本身就是一种 **“不特判 tail，只要重就劈一半”的策略**：

- 它不会只盯着最后一段 KV，看哪里长就从哪里切；
- 若某个盒子中前缀和尾巴都很长，二分会在中间某处把整个区间切为两半，两边都包含一些 prefix 和一些 tail；
- 多步迭代后，整个 KV 区间会被均匀切成 2、4、8… 段，而不是“一个大前缀 + 若干小尾巴”。

这样做的好处是：

- 更接近“每一段 KV 负载相似”的理想状态，而不是只照顾某一头（tail）；
- 不依赖我们是否准确识别 tail，少了一层不确定性；
- 通过把拆分次数限制为 1〜2 次/盒，可以避免旧 SOTA 那种大量 tiny chunk 导致的元数据/调度开销放大。

### 7.6 相对 balancePack 是否有“较为明显的改进”？

这一点需要诚实区分：

- **从理论负载均衡角度**：
     - 纯粹看 1D 负载（每盒 cost），“总是对当前最长任务做二分，直到任务数足够多”是一个常见且有效的启发式；
     - 相比 baseline `balancePack` 那种固定阈值或简单按 kv_len 拆一刀的逻辑，
          - 贪心二分更关注“全局最重的盒子”，而不是每个盒子独立决定拆多少；
          - 对负载特别偏的 tree（有极少数超大盒子）理论上更有机会把 tail 拉平。

- **从 PAT 当前 kernel 约束和实验现实出发**：
     - 我们无法让一个 CTA 串行执行多个 KV 任务，所以所有拆分都必须体现在“多出的盒子 = 多出的 CTA”上；
     - 当 `CTA_target` 已经足够大、GPU 并发也饱和时，再额外多几个盒子带来的实际加速会逐渐变小；
     - 现有的 `kernel_perf` 表明，即使用了比 baseline 更聪明的启发式（light SOTA），整体平均也只是略微退步/打平，说明 **空间确实有限**。

基于这些考虑，可以给出一个相对保守但现实的预期：

1. **可行性**：
      - 按 KV cost 做贪心二分、直到 CTA 上限，是完全可实现且 correctness 无虞的方案；
      - 不需要修改 kernel，只需在 `balancePackSota` 内新增一个简单的最大堆和若干拆分逻辑。

2. **相对 balancePack 的改进空间**：
      - 在存在**极少数超大盒子**的 tree 上（例如长前缀 + 很长上下文），这个方案比“按固定阈值拆一次就算了”的 baseline 更有机会把 tail 压下去；
      - 若我们把 `CTA_target` 设定为“略高于 baseline_CTA 的一个安全倍数”（比如 ×1.5），理论上可以在少量 critical tree 上获得几〜十几个百分点的 tail 改善；
      - 但考虑到 kernel 并发与其他 pipeline 开销，**是否能在整体平均 latency 上看到“明显优于 balancePack”的提升，还需要实测**，不能光靠分析保证。

3. **与当前 light SOTA 相比**：
      - 当前 light SOTA 是“预估 L_target + 每盒单独算 split_k”，在 KV 分布极端不均匀时，可能对某个超级大盒子拆得还不够；
      - 贪心二分则保证“只要某个盒子仍然显著偏重且 CTA 数尚未达上限，就继续对它动刀”，理论上对这类极端 case 更友好；
      - 但对那些本来就比较均匀的 tree，两者都只能打平，贪心二分不会带来神迹般的提升。

综上，这种“基于 KV 成本的贪心二分拆分”策略：

- **在设计上是可行的，且比目前的 rule-based 拆分更系统**；
- **有望在一部分负载极不均的 tree 上比 balancePack / 现有 light SOTA 表现更好**，尤其是少数超大盒子的长前缀场景；
- 但考虑到 kernel 模式和 GPU 并发已经较充分，**不宜期待它在整个测试矩阵上带来“显著、普遍”的大幅改进**，更现实的目标是：
     - 保持大部分配置与 baseline 打平；
     - 修复/改善当前 SOTA 在个别 tree 上的退化（比如 `1,10_4096,416`, `4,16,256,512_512,32,128,32`）；
     - 把 tail 明显的 case 拉回甚至略优于 balancePack。

建议的下一步是：

- 在 C++ 中实现一个简化版的“贪心二分 SOTA”开关（例如 `use_greedy_split`），只在少量典型长前缀 tree 上试运行；
- 与 baseline 和当前 light SOTA 做三者对比，验证上述理论判断是否成立，再决定是否推广为默认 `balancePackSota` 策略。
