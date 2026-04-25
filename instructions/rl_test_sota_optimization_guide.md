# RL_test（Prefix Attn）SOTA 性能分析与改进方向说明书

本文基于一次完整跑通后的三类产物：

- GPU kernel 性能：`benchmark/kernel_perf.json`
- CPU 调度性能：`benchmark/schedule_perf.json`
- 调度详细 kernel_info：`benchmark/schedule.log`

目标：回答三个问题

1) 当前 synthetic trees 与 RL_test 的结果呈现什么特点？
2) 现状是否“令人满意”（在目标明确的前提下）？
3) 结合 SOTA 调度代码（`balancePackSota`）分析：RL_test 还有多大提升空间，以及可执行的改进路线。

---

## 1. 结论先行（TL;DR）

- **RL_test 的 GPU 侧增益稳定但偏小**：`pat_sota` 相对 `pat_baseline` 全部为正收益，范围约 **+0.8% ~ +6.2%**，中位数约 **+4.8%**。
- **RL_test 的 CPU 调度侧“多数更慢”但绝对值很小**：`pack_schedule_sota` 在 17 个 RL 记录里多数比 baseline 慢（11/17），但两者都在 **~0.2ms** 量级，通常不会成为端到端瓶颈。
- **SOTA 在 RL_test 的主要作用是“削峰填谷”（降低 tail CTA）**：从 `schedule.log` 可见，`max_blocks_in_CTA` 与 `kv_in_CTAs` 的最大值能下降 **15%~21%**。
- **为何削峰 15~21% 只换来 1~6% GPU 加速？**
  - RL_test 下 baseline 已有较多 CTA（base_cta ≫ `cta_limit`），整体并行度高，尾部 CTA 对 makespan 的支配程度下降。
  - SOTA 通过 split 会引入 **Q duplication** 与 **gather kernel** 等系统性开销，抵消部分收益。

---

## 2. 现有测试结果的特点

### 2.1 synthetic trees（非 RL）

从 `kernel_perf.json` 汇总：

- 样本量：24 条
- `pat_sota` 相对 `pat_baseline`：
  - 19 胜 / 5 负
  - 平均加速比（baseline/sota）≈ 1.09
  - 最好样本可达 **1.67×**（例如 `1,16_16384,32`）
  - 也存在明显回退（例如 `1,16_16384,128` 的部分 head config）

**解释**：synthetic trees 覆盖更多“极端不均衡”的树形结构，SOTA 的“拆超长 prefix box”更容易带来可观收益；同时也更容易出现“拆分过度 + gather 开销更大”导致的回退。

### 2.2 RL_test（真实 trace）

从 `kernel_perf.json` 汇总（仅 RL 条目）：

- 样本量：16 条（4 个 RL case × 4 组 head config）
- `pat_sota` 相对 `pat_baseline`：**16/16 全胜**
- 加速比分布（baseline/sota）：
  - min ≈ 1.008
  - p50 ≈ 1.048
  - mean ≈ 1.041
  - max ≈ 1.062

按 head config 分组（baseline/sota 的均值）：

- (32, 32)（HRatio=1）：≈ 1.015（约 +1.5%）
- (16, 8)（HRatio=2）：≈ 1.039（约 +3.9%）
- (32, 8)（HRatio=4）：≈ 1.055（约 +5.5%）
- (64, 8)（HRatio=8）：≈ 1.053（约 +5.3%）

**解释**：RL_test 下 SOTA 的收益与 HRatio 有明显正相关（HRatio 越大越容易出现不均衡/拖尾 CTA）。HRatio=1 时，瓶颈更不“尖”，SOTA 只能做小幅削峰，因此收益偏小。

---

## 3. 从 schedule.log 看 SOTA 在 RL_test 做了什么

以 `tree_rl_case_0000_pid_1955100`（HRatio=1, kvHead=32）为例，`schedule.log` 中 baseline vs SOTA 的关键差异：

- baseline：`max_blocks_in_CTA = 235`，`kv_in_CTAs` 最大值为 7520
- SOTA：`max_blocks_in_CTA = 199`，`kv_in_CTAs` 最大值降到 6368

进一步对全部 RL case 汇总（每个 case 4 个 head config 的平均值）：

- `tree_rl_case_0000_pid_1955100`：`kv_max` 7520 → 6368（-15.3%）
- `tree_rl_case_0001/2/3_pid_1954796`：`kv_max` 5696 → 4512（-20.8%）

**结论**：SOTA 确实在 RL_test 把最大的 KV chunk 拆小，并显著降低了“最长 CTA”工作量。

---

## 4. 代码层面的解释：为什么 RL_test 收益不大但很稳定

SOTA 调度入口：

- `PrefixTree::pack_schedule_sota(...)` 只是 `pack_schedule(..., use_sota=true)`
- 关键逻辑在 `PrefixTree::balancePackSota(...)`（文件：`csrc/prefix_tree.h`）

### 4.1 触发分支：base_cta >= cta_limit（@cap）

RL_test 的 base_cta 通常已经较大（例如 bucket0 里 CTA 数就有 16+），并且 kvHead=32 时 `cta_limit=6`，因此会进入 `balancePackSota` 的 **@cap 分支**：

- 不再像 baseline 的 `balancePack()` 那样“CTA 数够多就直接返回”
- 改为：
  - 评估不均衡度（max/avg）
  - 对明显拖尾的 box，选择 split_k（最多 8 份）把 block_table 均分

### 4.2 为什么 CPU schedule 时间可能更慢？

@cap 分支会对多个候选 box、多个 k 做收益评估，并可能产生更多 CTA（更多 PackedBox），因此：

- `time_sota` 可能略高于 baseline
- 但绝对值仍很小（~0.2ms），一般不会成为瓶颈

### 4.3 为什么 GPU 侧只得到 1~6%？

- RL_test 本身 base_cta 高，并行度强；把最大 CTA 减少 15~21% 不一定能线性映射到 makespan。
- split 会导致同一批 q_table 重复出现在多个 CTA（schedule.log 中可见），这意味着：
  - attention kernel 有额外的 split accumulation（写入 `oaccum/lseaccum`）
  - 需要 `gather_kernel` 合并（文件：`csrc/pat_fwd_kernel.h`）

也就是说：SOTA 的收益主要来自“减少最长 CTA 的 KV 扫描”，而成本来自“多 CTA + gather”。两者叠加后的净收益在 RL_test 中就表现为稳定但不大的 +1~6%。

---

## 5. RL_test 还有多大性能提升空间？（可量化的判断方式 + 预估）

### 5.1 判断方式（推荐先做的量化）

建议把 RL_test 的潜在空间拆成两类：

1) **调度更激进能带来多少上限？**
   - 观察 `schedule.log`：baseline 的 `kv_max` 与 SOTA 的 `kv_max` 是否仍显著高于 `avg`。
   - 若 SOTA 后 `kv_max/avg` 仍很高，说明“还有明显尾部”，可能继续 split 或改 packing。

2) **split 系统开销占比有多大？**
   - 如果更激进 split 后 GPU 反而不涨或下降，往往是 gather/duplication 成本主导。
   - 需要对 `gather_kernel` 做 profile（Nsight Systems/Compute）确认占比。

### 5.2 对当前 4 个 RL case 的“谨慎预估”

基于现象：

- SOTA 已经把 `kv_max` 再降 15~21%
- GPU 侧收益只有 1~6%

推断：

- **对这批 RL case，继续靠“更激进 split”获得大幅提升的空间有限**（大概率是 **额外 +0~5%** 量级，取决于 gather 开销占比）。
- **更大的空间可能来自“降低 split 的系统开销”或“换一种更精准的成本模型/packing 策略”**，从而允许更聪明的 split（少做无收益 split，或用更低成本的合并方式）。

注意：这是对“当前 RL_testcase.json 中 4 个 case”的判断；如果换成更长前缀/更尖锐的 kv 分布，潜在空间会更大。

---

## 6. 详细改进路线（按投入产出排序）

### A. 立刻可做（Quick Wins，1~2 天）

1) **把 @cap 分支的关键决策 trace 打出来（只在 env 开关下）**
   - 现有：`PAT_DEBUG_IMBALANCE=1` 只打印少量摘要
   - 建议补充：对每个被 split 的 box 打印 `old_cost / best_k / net_gain / kv_before/after`
   - 目的：快速回答“为什么只 split 到 2/4，而不是 8？”、“为什么某些 case 不 split？”

2) **把 max_split_n（当前 8）做成 env 可调**
   - 对 RL_test 这种 base_cta 高的情况，允许更大 k（比如 16）有时能继续压低 kv tail
   - 但要同时观察 gather 开销是否导致回退

3) **把 min_blocks_per_piece（当前 4）做成 env 可调**
   - 某些 case 可能因为 piece 太短被禁止 split

### B. 中期（Algorithm 改造，3~7 天）

4) **把“max/avg + max_cost”改成轻量的 wave makespan 估计**
   - 当前模型主要围绕 `max_cost`，在 base_cta 很大时可能高估/低估 split 收益
   - 建议用简单的 LPT/greedy bin packing 模拟 `N_SM` 个槽位的 makespan

5) **cost model 系数校准（至少做一次粗拟合）**
   - 目前 `cost_a/cost_b/cost_c` 是固定经验值
   - 建议用少量 microcase（q, kv, k）跑真实 kernel 时间，拟合到更贴近 RL 分布

6) **split 按 tile 边界/kv token 边界，而不是按 block 均分**
   - 当前 `split_box_even_blocks` 只按 blocks 均分，可能造成 tile padding 额外浪费
   - 如果能按 kernel 的实际 tile N（或至少按 128/64 对齐）切，会更稳定

### C. 长期（Kernel 侧，1~3 周）

7) **优化或融合 gather_kernel**（文件：`csrc/pat_fwd_kernel.h`）
   - gather 是 split 的系统性成本来源
   - 方向：减少 gmem 往返、减少同步、或把部分合并逻辑融合进主 kernel

8) **减少 Q duplication 的实际代价**
   - 如果 split 的 k 增大，Q 相关的重复工作会增多
   - 方向：让 Q 侧重用（例如更好的缓存/更小的重复读写）或让 split 只发生在最尾部超长 box

---

## 7. 验收标准（建议明确“满意”的定义）

建议在 RL_test 上把目标拆成两个可验收指标：

- **吞吐/延迟指标**：在 4 个 head config 上，`pat_sota` 平均加速 ≥ X%（例如 5%）且无回退
- **稳定性指标**：SOTA 允许 schedule 输出与 baseline 不同，但应满足：
  - kernel 输出数值正确（当前 RL perf 未跑 correctness；建议补一个轻量 correctness 采样）
  - `num_split_per_seq` 不超过合理上限（避免 gather 成本爆炸）

---

## 8. 附录：相关代码位置

- SOTA 调度入口：`csrc/prefix_tree.h` 的 `pack_schedule_sota()` / `balancePackSota()`
- baseline 调度：`csrc/prefix_tree.h` 的 `balancePack()`
- split 合并（gather）：`csrc/pat_fwd_kernel.h` 的 `gather_kernel()`

---

## 9. SOTA 超参数（含调参方法）

这部分专门解释：`balancePackSota@cap`（base_cta >= cta_limit）分支里影响 split 激进度的“超参数”有哪些，以及一套可复现的调参流程。

> 背景：@cap 分支会对候选 split 做收益评估：
>
> - 估计收益：`makespan_benefit ~= waves_before * max_cost - waves_after * new_max_cost`
> - 估计开销：`net_overhead ~= (k-1) * (q_dup + gather + launch)`
> - 选择策略：Stage 1 全局贪心（每轮挑一个 net_gain 最大的 split 动作并立即应用），直到无正收益或达到限制。

### 9.1 参数说明（已固化为编译期常量）

以下参数不再通过环境变量读取；它们已在 `csrc/prefix_tree.h` 中固化：

- `PAT_SPLIT_QDUP_COEFF`：固定为 `0.25*a`
- `PAT_GATHER_KERNEL_QH_COEFF`：固定为 `0.10*a`
- `PAT_GATHER_KERNEL_BASE_COST`：已移除（固定为 0，不再计入开销）
- `PAT_SPLIT_CTA_LAUNCH_COST`：已移除（固定为 0，不再计入开销）

如果需要调参：直接修改 `csrc/prefix_tree.h` 中 `_bp_cost_model` 的常量并重新编译扩展（例如 `python setup.py build_ext --inplace`），再跑 `benchmark/run_kernel_bench.sh`。

### 9.2 代码常量（需要改代码）

这类“超参数”写死在 `balancePackSota@cap` 中：

- `max_split_n=16`：单个 box 最多拆成多少份。
- `min_blocks_per_piece=4`：每段最少 blocks，避免碎片化。
- `min_kv_tokens_per_piece=max(min_blocks_per_piece*block_size, batch_min_kv)`：每段最短 KV，进一步抑制极短 piece。
- `cur_max_split < 32`：每条 seq 的 split 上限。
- 候选 `k` 集合：当前是 `{2,3,4,6,8,12,16}`，减少搜索开销。
- `max_actions=64`：全局贪心最多应用多少次 split 动作（避免最坏情况过慢）。

如果你确认某类 workload 需要更细粒度的拆分（例如更长 prefix、更尖的 KV tail），通常优先考虑：

- 把候选 `k` 增加一些（如加入 `5,7,10,14`），或直接改成全范围 `k=2..max_k`。
- 在确认 gather 占比可接受的前提下，提高 `max_split_n`。

### 9.3 推荐调参流程（可执行）

目标：在固定 RL_testcase.json 的前提下，找到“平均加速最大且无回退”的一组系数。

1) 先固定测试集与脚本

- 使用现有脚本跑 RL：`benchmark/run_kernel_bench.sh`
- 确认输出里包含 `pat_baseline` / `pat_sota`（GPU kernel latency），并写入 `benchmark/kernel_perf.json`。

2) 单变量 sweep（需要改代码 + 重编译）

- 优先只动 `PAT_GATHER_KERNEL_QH_COEFF`（它最直接影响 HRatio 大的 case 是否敢拆）。
- 每次修改 `csrc/prefix_tree.h` 后执行一次 `python setup.py build_ext --inplace`，然后重复跑 `benchmark/run_kernel_bench.sh`。

3) 观察两个信号，决定下一步

- 信号 A（是否拆得过多/过少）：打开 `PAT_DEBUG_IMBALANCE=1` 跑一两个 case，看 `greedy_actions`/`final_cta`。
- 信号 B（split 开销是否主导）：用 Nsight Systems/Compute 看 `gather_kernel` 是否占用显著。

经验规则：

- 如果 `greedy_actions` 很大且 GPU 不涨：优先调大 `PAT_SPLIT_QDUP_COEFF`。
- 如果 HRatio 大（4/8）时收益本来应该更好但没起来：尝试调小 `PAT_GATHER_KERNEL_QH_COEFF`。

4) 再做一个小网格（2×2 或 3×3）

建议把变量控制在两个维度，避免组合爆炸：

- 维度 1：`PAT_GATHER_KERNEL_QH_COEFF`（控制对大 QH 的保守程度）
- 维度 2：`PAT_SPLIT_QDUP_COEFF`（控制总体拆分的保守程度）

选 2~3 个离散值，形成 4~9 组组合即可。

### 9.4 如何判断“调参方向对了”

建议把验收拆成三条曲线：

- RL_test 总体均值/中位数加速（baseline/sota）
- 分 config 的加速（尤其看 `kvHead=32, HRatio=1` 是否仍稳定为正）
- `greedy_actions` 分布（是否出现异常极大值，避免潜在 gather 爆炸）

当你发现“加速提升来自某一小撮 case，但另一些 case 回退”，通常意味着：

- cost model 对那一类 case 的 `makespan_benefit` 高估，或
- gather/duplication 开销在那一类 case 上被低估。

此时优先通过调大 `PAT_GATHER_KERNEL_*` 或 `PAT_SPLIT_QDUP_COEFF` 做保守化；再考虑做更精细的 cost model 拟合。

