## FlashInfer 动态负载均衡调度技术文档

### 一、论文原文（节选）

以下内容摘自 FlashInfer 论文第 3.3 节 “Dynamism-Aware Runtime” 及第 3.3.1 节 “Load-balanced Scheduling”。

#### 3.3 Dynamism-Aware Runtime

In this section we introduce the runtime design of FlashInfer, including the dynamic scheduling framework, and the composable formats for memory efficient attention.

##### 3.3.1 Load-balanced Scheduling

In FlashInfer, the load-balanced scheduling algorithm aims to minimize SM idle time by distributing the workload evenly across all SMs. It takes the sequence length information of the query/output and key/value dimensions as input, and produces both the mapping between the workload and Cooperative Thread Arrays (CTAs) and the index mapping for partial and final outputs. The algorithm is presented in Algorithm 1 (the head dimension is omitted for simplicity). Our approach is inspired by Stream-K (Osama et al., 2023); however, because LLM serving requires deterministic outputs, we did not incorporate atomic aggregation in Stream-K implementation to avoid non-deterministic behavior. The scheduling algorithm generates deterministic aggregation order when provided with identical sequence length information.

---

**Algorithm 1 FlashInfer’s balanced scheduling algorithm**

1: **Input:** \(\{l_{qo}(i), l_{kv}(i)\}_i\), query tile size \(T_q\).  
2: Define the cost of a tile \(l_{qo}, l_{kv}\) as \((\alpha, \beta)\) are hyperparameters:  

   \[\text{cost}(l_{qo}, l_{kv}) = \alpha l_{qo} + \beta l_{kv}\]

3: Compute the maximum KV chunk size \(L_{kv}\) by  

   \[L_{kv} = \frac{\sum_i \left[ \frac{l_{qo}(i)}{T_q} \right] \cdot l_{kv}(i)}{\#\text{CTA}}\]

4: Split each query tile’s KV into chunks, with maximum size \(L_{kv}\), we assign each chunk a work index \(w\), and the length of the chunk is \(l_{kv}(w)\).  
5: Let \(W = \{(w, l_{kv}(w))\}\) and sort the entries in descending order of length.  
6: \(Q \leftarrow \text{PriorityQueue}(\{(c, 0)\})\) where \(c\) is the CTA index.  
7: **while** \(W \neq \emptyset\) **do**  
8:    \(c, \text{current\_cost} \leftarrow Q.\text{pop\_min}()\)  
9:    \(w, l_{kv}(w) \leftarrow W.\text{pop}()\)  
10:    \(\text{new\_cost} \leftarrow \text{current\_cost} + \text{cost}(T_q, l_{kv}(w))\)  
11:    Assign chunk \(w\) to CTA \(c\)  
12:    \(Q.\text{push}((c, \text{new\_cost}))\)  
13: **end while**

---

The workflow of FlashInfer’s runtime scheduler is shown in Figure 6. FlashInfer’s load-balanced runtime scheduler, sequence length information (both on query/output and key/value dimension) are provided to the scheduler to compute the plan information: (1) Work queue of each CTA (2) Index mapping between partial and final outputs. These plan information are cached at GPU-side and used as inputs for persistent attention/contraction kernels.

FlashInfer guarantees both attention and contraction stage are compatible with CUDAGraphs (Gray, 2019; Nguyen et al., 2021). Both attention and contraction stage use persistent kernel and the grid size is fixed once compiled, which means the kernel is launched with the same grid size for each generation step. We set the fixed offset for each section of the workspace buffer to store partial outputs and plan information to make sure the pointers passed to the kernel are the same for each generation step, meeting the requirement of CUDAGraphs (see Appendix D.1 for details). We merge the two stages into one persistent kernel, eliminating intra-kernel overhead.

---

### 二、技术阐述

#### 2.1 调度目标与背景

在 LLM 推理服务中，一个批次（batch）内同时处理多个请求，每个请求的查询长度 \(l_{qo}\)（query 中的 token 数）和 KV 缓存长度 \(l_{kv}\)（历史上下文 token 数）可能差异很大。传统的按请求顺序执行注意力计算的方式，会导致部分 GPU 流处理器（SM）因处理短序列而空闲，而另一些 SM 因处理长序列而过载，造成硬件资源浪费和延迟增加。

FlashInfer 的负载均衡调度器旨在解决这一问题，通过将计算任务切分为细粒度的“块”（chunks），并采用贪心算法将这些块均匀分配给所有 CTA（Cooperative Thread Array，即 CUDA 线程块），从而使各 SM 的工作量接近，减少空闲时间，提高整体吞吐。

#### 2.2 成本模型（第 2 行）

调度器使用一个线性成本模型来估算每个“查询块 × KV 块”的计算成本：

\[
\text{cost}(l_{qo}, l_{kv}) = \alpha \cdot l_{qo} + \beta \cdot l_{kv}
\]

其中：
- \(l_{qo}\)：查询块中包含的 query token 数（在算法中固定为 \(T_q\)）。
- \(l_{kv}\)：该查询块需要处理的 KV 片段长度（token 数）。
- \(\alpha, \beta\)：超参数，通过微基准测试拟合，用于反映查询长度和 KV 长度对执行时间的不同影响（例如，处理 query 可能涉及更多计算，而加载 KV 受内存带宽影响）。

该模型虽然简单，但能有效指导调度决策，且计算开销极低。

#### 2.3 确定 KV 块大小上限 \(L_{kv}\)（第 3 行）

公式：
\[
L_{kv} = \frac{\sum_i \left\lceil \frac{l_{qo}(i)}{T_q} \right\rceil \cdot l_{kv}(i)}{\#\text{CTA}}
\]

**解释**：
- 分子：对每个请求，计算其查询块个数 \(\lceil l_{qo}(i)/T_q \rceil\)，乘以该请求的 KV 长度 \(l_{kv}(i)\)，再对所有请求求和。这代表了批次中所有查询块的总工作量（以 KV 长度为单位）。
- 分母：CTA 总数，即本次 kernel 启动时使用的线程块数量（通常等于或略多于 SM 数量）。
- 结果：每个 CTA 平均需要处理的 KV 长度（token 数）。

将 \(L_{kv}\) 作为 KV 分块的上限，可以使得拆分后产生的 KV 块数量大致等于 CTA 总数，为后续负载均衡奠定基础。

#### 2.4 KV 分块与任务生成（第 4-5 行）

- 对于每个请求的每个查询块（大小为 \(T_q\)），将其 KV 缓存（长度 \(l_{kv}(i)\)）按最大长度 \(L_{kv}\) 切分成若干个连续的 chunk。每个 chunk 分配一个全局工作索引 \(w\)，并记录其实际 KV 长度 \(l_{kv}(w)\)。
- 将所有 chunk 按 \(l_{kv}(w)\) 降序排序（第 5 行）。优先分配长 chunk 可以避免最后剩余小 chunk 导致的负载不均。

**注意**：此处查询块是固定大小 \(T_q\) 的，而 KV 块是变长的（不超过 \(L_{kv}\)）。这种设计使得调度器可以独立处理查询维度和 KV 维度的并行性。

#### 2.5 贪心分配算法（第 6-13 行）

- 初始化一个优先队列 \(Q\)，包含所有 CTA，每个 CTA 的初始累计成本为 0（第 6 行）。
- 循环直到所有 chunk 分配完毕：
  1. 从优先队列中取出累计成本最小的 CTA（第 8 行）。
  2. 从排序后的 chunk 列表中取出当前最长的 chunk（第 9 行）。
  3. 计算该 chunk 的成本 \(\text{cost}(T_q, l_{kv}(w))\)，并更新该 CTA 的累计成本（第 10 行）。
  4. 将该 chunk 分配给该 CTA（第 11 行）。
  5. 将 CTA 及其新成本重新插入优先队列（第 12 行）。

该算法本质上是一种 **LPT（Longest Processing Time first）贪心调度**，常用于多机负载均衡问题。在 GPU 场景下，它能有效平衡各 CTA 的工作量，且由于成本模型线性、分配过程确定，保证了输出结果的可复现性。

#### 2.6 与 CUDA Graph 的协同

CUDA Graph 是一种将一系列 GPU 操作记录为单个图并整体重放的机制，可大幅减少 kernel 启动开销。但其要求每次重放时 kernel 的 grid 大小、参数指针等必须完全相同。

FlashInfer 通过以下设计保证兼容性：
- **固定 CTA 数量**：调度算法中的 \(\#\text{CTA}\) 在编译时确定，并在整个推理过程中保持不变。
- **固定内存偏移**：用于存储 partial outputs 和 plan information 的 workspace buffer 具有固定的内存偏移量，每次执行时指针地址不变。
- **持久内核**：将注意力计算和结果合并两个阶段融合为一个 persistent kernel，启动后持续运行直到处理完所有工作，消除了阶段间的 kernel 启动开销。

在运行时，调度器在 CPU 端执行 `plan` 函数（包含算法 1），生成计划信息并写入 GPU 的固定区域。实际 kernel 执行时使用这些计划信息。由于计划信息存储在固定位置，且 kernel 启动配置不变，整个 attention 计算步骤可以被 CUDA Graph 捕获，从而实现低延迟的推理。

#### 2.7 总结与意义

FlashInfer 的动态负载均衡调度通过以下设计要点，实现了高效的注意力计算：
1. **细粒度任务拆分**：将查询和 KV 分别切块，实现查询维度和 KV 维度的双重并行。
2. **低成本工作量估计**：使用线性模型，快速指导调度。
3. **贪心分配**：基于 LPT 算法，使 CTA 负载均衡。
4. **确定性执行**：避免原子操作，确保可复现性。
5. **CUDA Graph 友好**：通过固定配置和持久内核，降低 CPU 开销。

这一调度策略是 FlashInfer 能够在多种 LLM 推理场景（如长上下文、并行生成、混合请求）中取得显著性能提升的关键原因之一。