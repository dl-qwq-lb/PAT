## PAT balancePackSota 动态负载均衡调度（论文式伪代码与成本模型）

本文以论文式写法说明 PAT 中 `balancePackSota` 的动态负载均衡调度：给出符号定义、明确的成本函数与拖尾（tail）时间代理目标，并以伪代码（Algorithm blocks）描述两条分支的确定性 KV 拆分（split）策略。

与 FlashInfer 的“拆分 + 负载均衡”思想相似，但 PAT 的关键差异是：不做跨 CTA 的重排/重分配，而是在既有 CTA 列表上进行增量式拆分，以降低最大 CTA 成本 $C_{\max}$ 和/或提高可并发度，从而降低 $\widehat{T}$。

---

### 1. 问题定义与符号

#### 1.1 输入与输出

- 输入：
  - `boxes`：初始 CTA 工作单元列表（PackedBox）。
  - `kvHead`：KV head 数，用来确定本 batch 的 CTA 上限阈值 `cta_limit`。
  - `HRatio`：常用定义为 $HRatio = n^{(q)}_{heads} / n^{(kv)}_{heads}$；实现中使用 $HR=\max(1, HRatio)$。
- 输出：
  - `boxes'`：对部分 box 进行 KV 维度拆分（split）后的 PackedBox 列表（CTA 数可能增加）。

#### 1.2 PackedBox（调度视角）

对任意 PackedBox $b$（对应一个 CTA），抽象出三个调度相关量：

- $q(b)$：该 CTA 覆盖的 query 数（等价于 query tile / q-table 的长度）。
- $kv(b)$：该 CTA 需要处理的 KV token 数（下文默认使用 $\max(0, kv(b))$）。
- $blocks(b)$：该 CTA 的 KV block 数（若存在 block table；实现通常以 block 粒度拆分）。

此外存在按序列维护的拆分计数：

- `split_per_seq[qid]`：序列 `qid` 已被拆分的次数。
- 约束：对任意 `qid`，`split_per_seq[qid] \le 32`。

---

### 2. 成本模型与 makespan（拖尾）代理

本节给出与实现一致的、可快速计算的代理模型，用于比较“是否值得继续拆分”。

#### 2.1 单 CTA 成本函数（显式算式）

令 $HR=\max(1, HRatio)$。定义单个 CTA 的代理成本函数：

$$
\mathrm{cost}(q, kv; HR) = a\cdot q\cdot HR + b\cdot kv + c\cdot q\cdot kv\cdot HR
$$

其中系数为常量：$a=0.1$，$b=0.1$，$c=2.0$。

于是 box $b$ 的成本为 $C(b)=\mathrm{cost}(q(b), \max(0,kv(b)); HR)$。

直觉：$q\cdot kv\cdot HR$ 对应注意力计算的乘性主项；$q\cdot HR$ 与 $kv$ 的线性项用于吸收额外开销或非理想缩放。

#### 2.2 waves 形式的拖尾时间代理

设当前 CTA 集合为 $\mathcal{B}$，$N=|\mathcal{B}|$。令 `denom` 为每个 wave 最多并发 CTA 数：实现取 `max(1, (cta_cap>0 ? cta_cap : cta_limit))`。

$$
\mathrm{waves}(N) = \left\lceil \frac{N}{\mathrm{denom}} \right\rceil
\qquad
C_{\max}(\mathcal{B}) = \max_{b\in\mathcal{B}} C(b)
$$

拖尾时间的代理为：

$$
\widehat{T}(\mathcal{B}) = \mathrm{waves}(|\mathcal{B}|)\cdot C_{\max}(\mathcal{B})
$$

该代理不模拟细粒度调度队列，但能捕捉两类关键因素：并行度不足导致的 `waves` 增加，以及瓶颈 CTA 导致的 $C_{\max}$ 拖尾。

---

### 3. 拆分的固定开销（显式建模）

将 box $b$ 沿 KV 维度拆成 $k$ 份，会新增 $added=k-1$ 个 CTA。为抑制过度拆分，算法对每个候选动作扣除固定开销。

令 $qh = q(b)\cdot HR$，每新增一个 CTA 的开销为：

$$
\mathrm{overhead}_1(b) = \gamma_{qdup}\cdot qh + \gamma_{gqh}\cdot qh
$$

则拆成 $k$ 份的总开销为：

$$
\mathrm{Overhead}(b,k) = (k-1)\cdot \mathrm{overhead}_1(b)
$$

默认系数（已固化为编译期常量）：

- $\gamma_{qdup}$：`PAT_SPLIT_QDUP_COEFF`，固定为 $0.25a = 0.025$
- $\gamma_{gqh}$：`PAT_GATHER_KERNEL_QH_COEFF`，固定为 $0.10a = 0.01$
- $\gamma_{g0}$：已移除（固定为 $0$，不再计入开销）
- $\gamma_{launch}$：已移除（固定为 $0$，不再计入开销）

---

### 4. `cta_limit` 的分段规则（与 baseline 一致）

`cta_limit = ThresholdByKvHead(kvHead)`：

| kvHead | cta_limit |
|---:|---:|
| $\le 4$ | 54 |
| $\le 8$ | 27 |
| $\le 16$ | 13 |
| $\le 32$ | 6 |
| else | 4 |

---

### 5. 顶层策略：两条分支

令 `base_cta = |boxes|`。

- 当 `base_cta >= cta_limit`（@cap）：并行度基本够用，重点在削峰（降低 $C_{\max}$）与减少拖尾。
- 当 `base_cta < cta_limit`：并行度不足，先尽量补足 CTA（全局 KV 预算拆分），再用 spare CTA 做细化。

---

## Algorithm 1 balancePackSota（顶层入口）

1: **Input:** `boxes`, `kvHead`, `HRatio`
2: $B \leftarrow boxes$ (move)
3: **if** $B$ is empty: **return** $[\,]$
4: `base_cta ← |B|`
5: `cta_limit ← ThresholdByKvHead(kvHead)`
6: `cta_cap ← InferMaxCtaFromSM()`
7: $HR \leftarrow \max(1, HRatio)$
8: `cur_max_split ← max(split_per_seq)`
9: **if** `base_cta >= cta_limit`:
10: **return** `CapGreedySplit(B, cta_limit, cta_cap, HR, cur_max_split)` (Algorithm 2)
11: **else**:
12: $B_1 \leftarrow$ `GlobalKVBudgetSplit(B, base_cta, cta_limit, cur_max_split)` (Algorithm 3)
13: **return** `RefineWithSpareCTA(B1, cta_limit, cta_cap, HR, cur_max_split)` (Algorithm 4)

---

## Algorithm 2 CapGreedySplit（@cap：全局贪心拆分，压缩版）

设 $\mathrm{denom}=\max\{1,(cta\_cap>0?cta\_cap:cta\_limit)\}$，$m=\max\{4\cdot block\_size,\min(kv>0)\}$。

1. $N\leftarrow|B|$，重复至多 $A$ 次（或 `cur_max_split=32`）。
2. 计算 $C_{\max},C_{2nd},cnt$ 与 $w_-=\lceil N/\mathrm{denom}\rceil$。
3. 对每个可行候选 $(b,k)$（$k\in\{2,3,4,6,8,12,16\}$）定义
4. $$kv_{worst}=\max\big(\mathrm{AlignUp}(\lceil kv/k\rceil,m),\;\max\{0,kv-(k-1)\cdot\mathrm{AlignUp}(\lceil kv/k\rceil,m)\}\big).$$
5. $$C'_{\max}=\max\big(C_{other}(b),\;\mathrm{cost}(q,kv_{worst};HR)\big).$$
6. $$\Delta T=w_-\,C_{\max}-\left\lceil \frac{N+k-1}{\mathrm{denom}}\right\rceil\,C'_{\max}.$$
7. $$\mathrm{Overhead}(b,k)=(k-1)\cdot(\gamma_{qdup}\,qHR+\gamma_{g0}+\gamma_{gqh}\,qHR+\gamma_{launch}).$$
8. $$\mathrm{gain}(b,k)=\Delta T-\mathrm{Overhead}(b,k).$$
9. $(b^*,k^*)=\arg\max_{b,k}\;\mathrm{gain}(b,k)$（tie-break：更小索引、更小 $k$）。
10. 若 $\mathrm{gain}(b^*,k^*)\le 0$ 则停止；否则拆分 $b^*$ 为 $k^*$ 份并令 $N\leftarrow N+(k^*-1)$。

---
## Algorithm 3 GlobalKVBudgetSplit（非 @cap：全局 KV 预算补并行，压缩版）

1. 设 $E=cta\_limit-|B|$；若 $E\le 0$ 或 $\sum\max(0,kv)=0$ 则返回 $B$。
2. $$L_{kv}=\left\lceil \frac{\sum_{b\in B}\max(0,kv(b))}{cta\_limit}\right\rceil,\qquad L_{kv}^{eff}=\max\big(L_{kv},\min\{kv(b)>0\}\big).$$
3. $$d_i=\left\lceil kv(b_i)/L_{kv}^{eff}\right\rceil,\quad d_i\leftarrow\min\{d_i,8,blocks(b_i)\},\quad need_i=\max(0,d_i-1).$$
4. $$x_i=E\,need_i/\sum need,\qquad alloc_i\leftarrow\min\big(need_i,\lfloor x_i\rfloor\big).$$
5. 余量按 $frac_i=x_i-\lfloor x_i\rfloor$ 降序补齐（tie-break：更小索引）。
6. 设 $k_i=1+alloc_i$；若不满足可行性（每片至少 4 blocks 且 KV 不小于 $m$）则回退 $k_i$。
7. 依原顺序执行 $b_i\to\mathrm{SplitEvenBlocks}(b_i,k_i)$，生成 $B'$。
8. 若 $|B'|$ 达到 `cta_limit` 或 `cur_max_split=32` 则停止并返回 $B'$。

---
## Algorithm 4 RefineWithSpareCTA（spare CTA 二分细化，压缩版）

设 $R=cta\_limit-|B|$，$\mathrm{denom}=\max\{1,(cta\_cap>0?cta\_cap:cta\_limit)\}$；若 $R\le 0$ 或 `cur_max_split=32` 则返回。

1. while $R>0$：计算 $C_{\max},C_{2nd},cnt$，并令 $w_-=\lceil |B|/\mathrm{denom}\rceil$，$w_+=\lceil(|B|+1)/\mathrm{denom}\rceil$。
2. 对每个可二分 $b$：令 $kv' = \max(4\cdot block\_size,\lceil kv(b)/2\rceil)$。
3. $$\Delta T(b)=w_-\,C_{\max}-w_+\,\max\big(C_{other}(b),\mathrm{cost}(q(b),kv';HR)\big).$$
4. $$\mathrm{overhead}_1(b)=\gamma_{qdup}\,q(b)HR+\gamma_{g0}+\gamma_{gqh}\,q(b)HR+\gamma_{launch}.$$ 
5. $$\mathrm{gain}(b)=\Delta T(b)-\mathrm{overhead}_1(b).$$
6. 若 $w_+=w_-$ 且 $b$ 为唯一瓶颈：$\mathrm{gain}(b)\leftarrow\max\{\mathrm{gain}(b),\varepsilon+(C(b)-\mathrm{cost}(q(b),kv';HR))-\mathrm{overhead}_1(b)\}.$$
7. 取 $b^*=\arg\max_b\;\mathrm{gain}(b)$；若 $\mathrm{gain}(b^*)\le 0$ 则停止。
8. 否则二分拆分 $b^*$，并令 $R\leftarrow R-1$。
---
