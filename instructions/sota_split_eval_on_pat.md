# PAT 中 SOTA split-KV 异常现象根因分析（聚焦 tree_1,10_4096,416）

## 1. 你观察到的现象

在 tree_1,10_4096,416 及其若干 head 配置下，SOTA 后的输出表现为：

1. 共享前缀看起来“消失”了。
2. query 的拆分统计看起来没有体现“共享前缀贡献”。
3. 与 baseline 相比，KernelInfo 的分组形态明显不同，部分 comparison_pass 为 false。

这些现象是真实的，但其根因不是“只处理了部分 KernelInfo 字段”，而是 SOTA 对 boxes 的重组方式改变了调度语义。

## 2. 核心原因（最关键结论）

核心原因是：SOTA 在 balancePackSota 里做的是 split-KV 子段化，而不是前缀结构保持式分裂。

在当前实现中：

1. 每个原始 PackedBox 会按 blocks_per_chunk 被切成多个连续子段。
2. 每个子段都会复制同一份 q_table，但 block_table 只保留该子段。
3. 除第一个子段外，后续子段天然不再包含完整共享前缀片段。

所以在打印的 block_tables 视角下，会看到“共享前缀消失”。

这不是 prefix tree 本身丢失，而是调度阶段将一个“完整前缀块列表”拆成了多个“局部 KV 子窗口”，导致可视化上不再呈现完整公共前缀。

## 3. 为什么 tree_1,10_4096,416 特别容易触发该问题

该树结构具有“长公共前缀 + 相对短尾部”的特征：

1. 原始 box 的 KV 范围很长，且 batch 中 query 对同一前缀高度共享。
2. SOTA 依据 L_kv 切块时，长 box 很容易被切成多个 chunk。
3. chunk 从中后段开始时，不再携带完整前缀块序列，表现为共享前缀在输出中大面积缺失。

这类数据形态正好放大了 split-KV 的副作用。

## 4. 是否因为 SOTA 只处理了部分 KernelInfo

不是。

当前 pack_schedule_sota 最终写入 KernelInfo 的字段仍是全量主字段：

1. q_tables
2. block_tables
3. num_seqs_per_CTAs
4. CTA_ranks
5. kv_in_CTAs
6. num_split_per_seq

问题不在“字段没写”，而在“字段所承载的语义已变”。

更准确地说：
- baseline 的 block_tables 更像“完整共享前缀上下文”
- sota 的 block_tables 更像“被切分后的 KV 子段任务”

因此你看到的是“语义失配”，不是“字段缺失”。

## 5. SOTA 相对 baseline 对 boxes 做了什么

### baseline 的典型行为

1. 仅在阈值触发时拆分较长 box。
2. 拆分规模相对保守。
3. 多数场景仍保持较完整的共享前缀块结构。

### sota 的当前行为

1. 通过全局工作量估计 L_kv，默认更积极地切分长 box。
2. 每个切分子任务都复制 q_table，但 block_table 仅保留局部子段。
3. 再做 LPT + 最小堆分配，打散原 box 的局部顺序和聚合形态。
4. split_per_seq 会因为子任务增多而显著增长。

这几步叠加后，KernelInfo 中“原有完整共享前缀形态”会明显减弱。

## 6. 这会不会导致内核计算错误

结论：不必然错误，但风险增大，必须做输出正确性验证。

### 为什么不必然错误

1. kernel 输入结构仍满足接口：q_tables、block_tables、CTA_ranks、num_split_per_seq 都有值。
2. 每个 CTA 仍对应一份共享 block_table（只是变成子段）。
3. 若 reduction 语义与 rank 排序一致，理论上可得到正确聚合结果。

### 为什么风险增大

1. q_table 被复制到多个子段后，依赖 CTA_rank 和 num_split_per_seq 做跨片段聚合，调度顺序更脆弱。
2. 若某些 query 的 split rank 语义与预期不一致，可能造成聚合差异。
3. comparison_pass=false 已表明“调度结构差异”显著，需进一步验证数值输出是否等价。

因此不能仅凭 KernelInfo 结构完整就认定“计算正确”。

## 7. 回答“是否打破 PAT 原逻辑”

### 不打破的层面

1. prefix tree 构建流程未改。
2. pack_schedule 输出通路未改。
3. kernel 调用接口未改。

### 被改变的层面

1. 一个原始共享前缀 box 被分解成多个 KV 子窗口 box。
2. query split 行为与 baseline 不再一致。
3. 共享前缀可视表现被子段化掩盖。

所以它是“接口兼容但策略重写”，不是“框架断裂”。

## 8. 你这个现象的真正一句话解释

tree_1,10_4096,416 下共享前缀看似消失，根本原因是 SOTA 对 block_table 做了子段切分并复制 query 到多个子段任务，导致 KernelInfo 展示的是局部 KV 窗口而非完整前缀视图；这属于调度语义变化，不是 KernelInfo 字段缺失。

## 9. 建议的修正方向（紧贴当前问题）

1. 前缀保护切分策略：前缀公共块区间禁止切分，只切后缀可变区。
2. 条件切分：仅当 box 超过双阈值（KV 长度阈值 + 负载离散阈值）才 split。
3. 限制最大 split 次数：避免 query 被过度复制。
4. 组内切分优先：优先在同前缀组内平衡，减少跨组打散。
5. 增加一致性审计：在日志中输出每个 qid 的 split rank 序列与覆盖区间，定位异常聚合链。

## 10. 下一步验证清单

1. 数值正确性: 对同一输入比较 baseline/sota 的 attention 输出误差。
2. 结构正确性: 验证每个 qid 的 split 片段按 rank 可完整覆盖原 KV 区间且无重叠缺口。
3. 性能正确性: 只有在数值等价前提下再比较端到端吞吐与延迟。

如果第 1 项未通过，当前 SOTA 只能视为“调度实验版本”，不应进入默认路径。
