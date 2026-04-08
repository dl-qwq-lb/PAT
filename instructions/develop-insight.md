balancePackSota vs 现有 balancePack 对比分析
一、现有 balancePack 的工作方式
balancePack 是一个无参数纯启发式，分两阶段： prefix_tree.h:481-482

阶段 1（离群值拆分）：对 kv >= 128 的 CTA 计算平均 block 数，超过 5 × avg 的 CTA 按 avg_blocks 大小均分： prefix_tree.h:496-506

阶段 2（SM 填充）：若 CTA 总数不足 threshold_cnt，反复二分最长 CTA 直到达标或最长 CTA < 8 blocks： prefix_tree.h:564-570

二、balancePackSota 的核心改进
维度	balancePack	balancePackSota
决策依据	纯启发式（5x 阈值 + 最长优先）	7 项成本模型（Q dup, tile pad, warmup, wave, gather, page, benefit）
拆分粒度	阶段 1 按 avg 均分；阶段 2 固定二分	Branch A 搜索 k∈[2, max_k]；Branch B 按 L_kv 比例分配 + 贪心二分
CTA 预算	无显式预算（阶段 1 无限制）	Branch A 无硬限制但有成本约束；Branch B 受 cta_limit 约束
Wave 感知	无	有（delta_waves × new_bottleneck）
Gather 感知	无	有（首次触发 + 增量成本）
SM 数量感知	无（用 threshold_cnt 近似）	有（cudaDeviceGetAttribute 查询实际 SM 数）
理论上 balancePackSota 更优，因为它用成本模型替代了硬编码阈值，能在以下场景做出更好的决策：

异构 CTA 负载：当 CTA 间 KV 长度差异大时，balancePack 的 5x 阈值可能过于保守或激进；balancePackSota 的 1.2x avg 触发 + 成本模型能更精确地识别瓶颈
Wave 溢出敏感场景：当 CTA 数接近 SM 数的整数倍时，多加一个 CTA 可能导致额外 wave；balancePackSota 显式建模了这一点
Gather 开销敏感场景：当 max_split_per_seq 从 1 变为 2 时触发 gather kernel 的阶跃成本；balancePack 完全忽略
三、balancePackSota 的具体改进空间
问题 1：original_cost 缺少计算项
constexpr double cost_a = 0.1;  
constexpr double cost_b = 0.1;  
auto original_cost = [&](int q, int H, int KV) -> double {  
    return cost_a * (double)q * (double)H + cost_b * (double)KV;  
};
这个成本函数只有 Q 项和 KV 项，缺少 q × H × KV 的计算项。虽然 decoding 场景通常 memory-bound，但 PAT 的前缀 CTA 可以有 q=8, H=8，此时 AI = 2 × 8 × 8 = 128，接近 A100 的 ridge point（156）。缺少计算项会导致对前缀 CTA 的成本低估。

修正：

auto original_cost = [&](int q, int H, int KV) -> double {  
    const double mem = cost_b * (double)KV;  
    const double compute = cost_c * (double)q * (double)H * (double)KV;  
    return cost_a * (double)q * (double)H + std::max(mem, compute);  
};  
// cost_c ≈ cost_b / R, R = ridge point ≈ 156 (A100) or 165 (4090)
问题 2：cost_a 和 cost_b 都是 0.1，语义不清
cost_a = 0.1 来自 PAT 的 _Cred 系数，表示 reduction 相对于主计算的比例。但 cost_b = 0.1 没有物理依据——KV 读取是主成本，应该是基准值（1.0）。当前设置下两者等权，虽然因为 KV >> q*H 所以 KV 项仍然主导，但在 C_tile_pad = cost_b × BN_new × tile_delta 中，cost_b = 0.1 会严重低估 tile padding 的实际开销。 prefix_tree.py:30-31

修正：

constexpr double cost_a = 0.1;   // reduction / Q-side (相对于 KV 读取)  
constexpr double cost_b = 1.0;   // KV 读取 (基准值)  
constexpr double cost_c = 1.0 / 156.0;  // 计算项 (roofline)
问题 3：Branch A 的 base_cta 在迭代中不更新
Branch A 对所有 CTA 独立评估 split_plan，但 wave 模型中的 base_cta 始终是初始值。当多个 CTA 同时被拆分时，实际的 total_after 应该是 base_cta + Σ(k_i - 1)，而非 base_cta + (k_i - 1)。

// 当前：每个 CTA 独立评估  
const int total_after = (base_cta + added_cta) * kvHead;  // 只考虑当前 CTA 的拆分  
  
// 应改为：累积所有已决定拆分的 CTA  
int accumulated_added = 0;  
for (size_t i = 0; i < crop.size(); ++i) {  
    // ... evaluate with base_cta + accumulated_added ...  
    if (best_k > 1) {  
        split_plan[i] = best_k;  
        accumulated_added += best_k - 1;  
    }  
}
问题 4：Branch B 的 L_kv 分配不使用成本模型
Branch B 中 base_cta < cta_limit 时，先用 global_L_kv = ceil(total_kv / cta_limit) 做均匀分配，再用贪心二分补充。L_kv 分配阶段完全不考虑成本模型，可能导致：

短 KV 的 CTA 被不必要地拆分（因为 kv > L_kv）
长 KV 但 Q 很多的 CTA 拆分不够（因为 L_kv 只看 KV 长度）
修正：用成本模型替代 L_kv 分配：

// 替代 L_kv 方案：用 priority queue 贪心拆分最重 CTA  
// 每次拆分后重新评估，直到用完预算或无有利拆分
问题 5：缺少 occupancy 建模
不同 tile 配置的 shared memory 用量差异很大，直接影响每个 SM 能并发执行的 CTA 数： kernel_traits.h:71-73

SmemSize = kBlockM × kHeadDim × sizeof(Element) + kBlockN × kHeadDim × 2 × sizeof(Element)  
  
Tile (64,128,4): SmemSize = 64×128×2 + 128×128×2×2 = 16KB + 64KB = 80KB → 1-2 CTAs/SM  
Tile (16,64,1):  SmemSize = 16×128×2 + 64×128×2×2  = 4KB + 32KB  = 36KB → 2-4 CTAs/SM  
拆分后子 CTA 的 KV 变短，可能匹配更小的 tile（如从 (64,128) 降到 (64,64)），shared memory 减半，occupancy 翻倍。这个收益在当前成本模型中完全缺失。

问题 6：infer_BN 与实际 getMNW 不一致
balancePackSota 中的 infer_BN 是手写的近似逻辑，而实际的 tile 分配在 pack_schedule 中通过 getMNW(m_val, kv_in_CTA) 完成： prefix_tree.h:229-235

两者的逻辑可能不一致，导致成本模型中的 BN 与实际 kernel 使用的 kBlockN 不同。应直接复用 getMNW。

问题 7：调度开销
Branch A 的复杂度为 O(n × max_k)（n 为 CTA 数），Branch B 的贪心二分为 O(spare × n)。对于大 batch（n > 1000），这可能成为 CPU 侧瓶颈。PAT 的调度在每次 decode step 都执行，需要控制在微秒级。

四、更优的架构设计
推荐采用全局优先队列 + 增量成本评估的架构,核心思想：

std::vector<PackedBox> balancePackOptimal(std::vector<PackedBox> boxes,  
                                          int kvHead, int HRatio) {  
    // ---- 全局状态 ----  
    int total_ctas = boxes.size();  
    int global_max_split = *std::max_element(split_per_seq.begin(), split_per_seq.end());  
  
    // ---- 成本函数 (roofline) ----  
    auto cta_cost = [&](const PackedBox& b) -> double {  
        int q = b.q_table.size();  
        int H = std::max(1, HRatio);  
        int kv = b.kv_in_CTA;  
        double mem = (double)kv;                              // KV 读取  
        double compute = (double)q * H * kv / RIDGE_POINT;   // GEMM  
        return 0.1 * q * H + std::max(mem, compute);         // reduction + max(mem, compute)  
    };  
  
    // ---- 优先队列 (max-heap by cost) ----  
    auto cmp = [&](int a, int b) { return cta_cost(boxes[a]) < cta_cost(boxes[b]); };  
    std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);  
    for (int i = 0; i < boxes.size(); ++i) pq.push(i);  
  
    double global_max_cost = cta_cost(boxes[pq.top()]);  
  
    // ---- 目标成本 (wave-aware) ----  
    int N_SM = get_sm_count();  
    int total_grid = total_ctas * kvHead;  
    int waves = (total_grid + N_SM - 1) / N_SM;  
    double target_cost = global_max_cost * 0.8;  // 目标：降低瓶颈 20%  
  
    // ---- 贪心拆分 ----  
    while (!pq.empty() && global_max_split < 32) {  
        int idx = pq.top();  
        double top_cost = cta_cost(boxes[idx]);  
  
        // 终止条件：最重 CTA 已足够轻  
        if (top_cost <= target_cost) break;  
  
        pq.pop();  
  
        // 评估最优 k  
        int best_k = 1;  
        double best_net = 0.0;  
        for (int k = 2; k <= max_k; ++k) {  
            double net = evaluate_split_overhead(  
                boxes[idx], k, total_ctas, global_max_cost,  
                second_max_cost, global_max_split, N_SM, kvHead);  
            if (net < best_net) {  
                best_net = net;  
                best_k = k;  
            }  
        }  
  
        if (best_k <= 1 || best_net >= 0.0) continue;  // 无有利拆分  
  
        // 执行拆分  
        auto pieces = split_box(boxes[idx], best_k);  
        boxes[idx] = std::move(pieces[0]);  
        for (int i = 1; i < pieces.size(); ++i) {  
            boxes.push_back(std::move(pieces[i]));  
        }  
        total_ctas += best_k - 1;  
  
        // 更新全局状态  
        update_global_max_cost();  
        for (auto& new_piece : new_pieces) pq.push(new_piece_idx);  
    }  
  
    return boxes;  
}
这种架构的优势：

特性	balancePack	balancePackSota	优先队列方案
决策顺序	固定（阶段 1 → 阶段 2）	分支（A 或 B）	自适应（始终拆最重）
全局状态一致性	N/A	部分（base_cta 不更新）	完全一致（每次拆分后更新）
终止条件	硬编码阈值	成本模型 + 硬编码	纯成本驱动
复杂度	O(n)	O(n × max_k)	O(n log n)（heap）
可扩展性	差	中	好（加新成本项只需改 evaluate）
关键改进点：

不区分 Branch A/B：统一用优先队列处理，无论 base_cta 是否超过 cta_limit
增量更新：每次拆分后立即更新 total_ctas、global_max_cost、global_max_split，避免 stale state
自然终止：当 best_net >= 0（无有利拆分）或 top_cost <= target_cost 时自动停止
Occupancy 感知：在 evaluate_split_overhead 中加入 tile size 变化导致的 occupancy 变化
五、总结
balancePackSota 相比 balancePack 在理论上更优，因为它引入了成本模型、wave 感知和 gather 感知。但它存在以下主要问题需要修正：

cost_b 应为 1.0（而非 0.1），否则 tile padding 和 KV 读取成本被低估
缺少计算项 cost_c × q × H × KV，对前缀 CTA 成本低估
Branch A 的 wave 模型不累积多个 CTA 的拆分效果
Branch B 的 L_kv 分配绕过了成本模型
infer_BN 应复用 getMNW 以保证一致性
缺少 occupancy 建模（tile size 变化影响 SM 并发度）
最有利于平衡分割的架构是全局优先队列 + 增量成本评估：始终拆分当前最重的 CTA，每次拆分后更新全局状态，用统一的成本模型（roofline + wave + gather + occupancy）评估收益，直到无有利拆分为止。这种架构避免了 Branch A/B 的分支逻辑，保证全局状态一致性，且复杂度为 O(n log n)，适合大 batch 场景。