#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <queue>
#include <cstdlib>

// 前缀树类定义

struct KernelInfo {
    std::vector<torch::Tensor> q_tables;
    std::vector<torch::Tensor> block_tables;
    std::vector<torch::Tensor> num_seqs_per_CTAs;
    std::vector<torch::Tensor> CTA_ranks;
    std::vector<torch::Tensor> kv_in_CTAs;

    std::vector<std::vector<int>> storage_q;
    std::vector<std::vector<int>> storage_block;
    std::vector<std::vector<int>> storage_num_seqs;
    std::vector<std::vector<int>> storage_ranks;
    std::vector<std::vector<int>> storage_kv;

    std::vector<std::vector<int>> MNWs;         // List[List[int]]

    torch::Tensor num_split_per_seq;
    int max_split_per_seq = 0;
    int max_seqs_in_CTA = 0;
    int max_blocks_in_CTA = 0;

    void to_gpu(torch::Device device) {
        size_t n = q_tables.size();
        for (size_t i = 0; i < n; ++i) {
            q_tables[i] = q_tables[i].to(device, /*non_blocking=*/true);
            block_tables[i] = block_tables[i].to(device, true);
            num_seqs_per_CTAs[i] = num_seqs_per_CTAs[i].to(device, true);
            CTA_ranks[i] = CTA_ranks[i].to(device, true);
            kv_in_CTAs[i] = kv_in_CTAs[i].to(device, true);
        }

        num_split_per_seq = num_split_per_seq.to(device, true);
    }
};

struct Node {
    int parent = -1;
    int s_value = 0;
    int length = 0;

    const int* block_ptr = nullptr;

    std::vector<int> seq_indices;
    std::unordered_map<int, int> children;

    Node(int p, int s, int l, const std::vector<int>& seq, const int* bp)
        : parent(p), s_value(s), length(l), seq_indices(seq), block_ptr(bp) {}
};

struct PackedBox {
    std::vector<int> q_table;
    std::shared_ptr<std::vector<int>> block_table_ptr;
    int num_seqs_per_CTA = 0;
    int kv_in_CTA = 0;
    int CTA_rank = 0;
    int split_depth = 0;  // 记录该盒子由原始盒拆分的深度，用于限制最大拆分次数
};

class PrefixTree {
public:
    int block_size;
    std::vector<Node> nodes;
    int root;
    int num_nodes = 0;
    std::vector<int> split_per_seq;
    KernelInfo _internal_info;

    PrefixTree(int bs) : block_size(bs) {
        add_node(-1, 0, 0, {}, nullptr);
        root = 0;
    }

    int add_node(int parent, int s_value, int length,
                 const std::vector<int>& seq_indices,
                 const int* block_ptr) {
        nodes.emplace_back(parent, s_value, length, seq_indices, block_ptr);
        return num_nodes++;
    }

    void build_radix_tree(const int* seq_lens_ptr,
                          const int* flat_table_ptr,
                          const int num_seqs,
                          int max_blocks) {
        nodes.reserve(num_seqs * 2);

        for (int i = 0; i < num_seqs; ++i) {
            // Row i start = Base + i * Stride
            const int* row_ptr = flat_table_ptr + (i * max_blocks);

            int seq_len = seq_lens_ptr[i];
            int current_block_count = 0;
            current_block_count = (seq_len + block_size - 1) / block_size;

            insert(i, seq_len, row_ptr, current_block_count);
        }
        split_per_seq = std::vector<int>(num_seqs, 0);
    }

    void insert(int sId, int seq_len, const int* input_blocks_ptr, int input_block_count) {
        int node_idx = root;
        int res_block = input_block_count;
        int current_offset = 0;

        while (res_block > 0) {
            int first_block = input_blocks_ptr[current_offset];

            auto it = nodes[node_idx].children.find(first_block);

            if (it != nodes[node_idx].children.end()) {
                int child_id = it->second;

                int child_num_blocks = (nodes[child_id].length + block_size - 1) / block_size;
                const int* child_ptr = nodes[child_id].block_ptr;


                int limit = std::min(child_num_blocks, res_block);
                int common_len = 0;

                for (int i = 0; i < limit; ++i) {
                    if (input_blocks_ptr[current_offset + i] == child_ptr[i]) {
                        common_len++;
                    } else {
                        break;
                    }
                }

                if (common_len == child_num_blocks) {
                    nodes[child_id].s_value += 1;
                    nodes[child_id].seq_indices.push_back(sId);

                    node_idx = child_id;
                    res_block -= common_len;
                    seq_len -= common_len * block_size;
                    current_offset += common_len;
                } else {
                    std::vector<int> mid_seq = nodes[child_id].seq_indices;
                    const int* mid_block_ptr = nodes[child_id].block_ptr;

                    int split_block_id = child_ptr[common_len];
                    int original_head_block = child_ptr[0];

                    int mid = add_node(node_idx,
                                       nodes[child_id].s_value + 1,
                                       common_len * block_size,
                                       mid_seq,
                                       mid_block_ptr);

                    nodes[mid].children[split_block_id] = child_id;
                    nodes[node_idx].children[original_head_block] = mid;

                    nodes[child_id].parent = mid;
                    nodes[child_id].block_ptr += common_len;
                    nodes[child_id].length -= common_len * block_size;

                    if (common_len == res_block) {
                        nodes[mid].seq_indices.push_back(sId);
                        break;
                    }

                    const int* new_leaf_ptr = input_blocks_ptr + current_offset + common_len;
                    // int new_len = std::min(res_block * block_size, seq_len) - common_len * block_size;
                    int new_len = seq_len - common_len * block_size;

                    int new_node_id = add_node(mid, 1, new_len, {sId}, new_leaf_ptr);

                    nodes[mid].seq_indices.push_back(sId);
                    nodes[mid].children[new_leaf_ptr[0]] = new_node_id;
                    break;
                }
            } else {
                const int* new_leaf_ptr = input_blocks_ptr + current_offset;
                int new_node = add_node(node_idx, 1, seq_len, {sId}, new_leaf_ptr);
                nodes[node_idx].children[first_block] = new_node;
                break;
            }
        }
    }

    void pack_schedule(std::optional<std::vector<std::vector<int>>> MNWs,
                       int HRatio = 1,
                       int kvHead = 8,
                       bool use_sota = false,
                       int max_cta = -1) {

        std::vector<std::vector<int>> buckets;
        if (!MNWs.has_value() || MNWs->empty()) {
            buckets = {
                {64, 128, 4}, {64, 64, 4}, {64, 32, 4},
                {32, 128, 2}, {32, 64, 2}, {32, 32, 2}, {32, 16, 2},
                {16, 128, 1}, {16, 64, 1}, {16, 32, 1}, {16, 16, 1}
            };
        } else {
            buckets = *MNWs;
        }


        int max_bucket_m = 0;
        for (const auto& b : buckets) {
            if (b[0] > max_bucket_m) max_bucket_m = b[0];
        }
        int mm = max_bucket_m / HRatio;
        if (mm < 1) mm = 1;

        std::vector<PackedBox> packed_boxes;

        const auto& root_children = nodes[root].children;
        for (const auto& kv : root_children) {
            int child_node_id = kv.second;
            auto new_boxes = _tree_heuristics(child_node_id, mm, {});
            packed_boxes.insert(packed_boxes.end(),
                                std::make_move_iterator(new_boxes.begin()),
                                std::make_move_iterator(new_boxes.end()));
        }

        // Balance Pack
        if (use_sota) {
            packed_boxes = balancePackSota(packed_boxes, kvHead, HRatio, max_cta);
        } else {
            packed_boxes = balancePack(packed_boxes, kvHead, HRatio);
        }

        std::map<std::vector<int>, std::vector<PackedBox>> grouped;
        for (const auto& b : buckets) {
            grouped[b] = {};
        }

        if (buckets.size() == 1) {
            grouped[buckets[0]] = std::move(packed_boxes);
        } else {
            for (auto& box : packed_boxes) {
                int m_val = (int)box.q_table.size() * HRatio;

                std::vector<int> target_mnw = getMNW(m_val, box.kv_in_CTA);

                if (grouped.find(target_mnw) != grouped.end()) {
                    grouped[target_mnw].push_back(std::move(box));
                } else {
                    grouped[buckets[0]].push_back(std::move(box));
                }
            }
        }

        auto cpu_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

        for (const auto& mnw : buckets) {
            auto it = grouped.find(mnw);
            if (it == grouped.end() || it->second.empty()) continue;

            const auto& group_boxes = it->second;
            int num_CTAs = (int)group_boxes.size();

            int cur_max_seqs = 0;
            int cur_max_blocks = 0;
            for (const auto& box : group_boxes) {
                if ((int)box.q_table.size() > cur_max_seqs) cur_max_seqs = box.q_table.size();
                if ((int)box.block_table_ptr->size() > cur_max_blocks) cur_max_blocks = box.block_table_ptr->size();
            }

            if (cur_max_seqs > _internal_info.max_seqs_in_CTA) _internal_info.max_seqs_in_CTA = cur_max_seqs;
            if (cur_max_blocks > _internal_info.max_blocks_in_CTA) _internal_info.max_blocks_in_CTA = cur_max_blocks;

            std::vector<int> flat_q(num_CTAs * cur_max_seqs, 0);
            std::vector<int> flat_block(num_CTAs * cur_max_blocks, 0);
            std::vector<int> flat_num_seqs; flat_num_seqs.reserve(num_CTAs);
            std::vector<int> flat_rank;     flat_rank.reserve(num_CTAs);
            std::vector<int> flat_kv;       flat_kv.reserve(num_CTAs);

            for (int i = 0; i < num_CTAs; ++i) {
                const auto& box = group_boxes[i];
                std::copy(box.q_table.begin(), box.q_table.end(), flat_q.begin() + i * cur_max_seqs);
                std::copy(box.block_table_ptr->begin(), box.block_table_ptr->end(), flat_block.begin() + i * cur_max_blocks);

                flat_num_seqs.push_back(box.num_seqs_per_CTA);
                flat_rank.push_back(box.CTA_rank);
                flat_kv.push_back(box.kv_in_CTA);
            }

            _internal_info.MNWs.push_back(mnw);

            _internal_info.storage_q.push_back(std::move(flat_q));
            _internal_info.q_tables.push_back(
                torch::from_blob(_internal_info.storage_q.back().data(),
                                 {num_CTAs, cur_max_seqs},
                                 cpu_opts)
            );

            _internal_info.storage_block.push_back(std::move(flat_block));
            _internal_info.block_tables.push_back(
                torch::from_blob(_internal_info.storage_block.back().data(),
                                 {num_CTAs, cur_max_blocks},
                                 cpu_opts)
            );

            _internal_info.storage_num_seqs.push_back(std::move(flat_num_seqs));
            _internal_info.num_seqs_per_CTAs.push_back(
                torch::from_blob(_internal_info.storage_num_seqs.back().data(),
                                 {num_CTAs},
                                  cpu_opts)
            );

            _internal_info.storage_ranks.push_back(std::move(flat_rank));
            _internal_info.CTA_ranks.push_back(
                torch::from_blob(_internal_info.storage_ranks.back().data(),
                                 {num_CTAs},
                                 cpu_opts)
            );

            _internal_info.storage_kv.push_back(std::move(flat_kv));
            _internal_info.kv_in_CTAs.push_back(
                torch::from_blob(_internal_info.storage_kv.back().data(),
                                 {num_CTAs},
                                 cpu_opts)
            );
        }

        // Shared Info
        _internal_info.num_split_per_seq = torch::from_blob(split_per_seq.data(),
                                                           {(long)split_per_seq.size()},
                                                           cpu_opts);

        _internal_info.max_split_per_seq = 0;
        for(int v : split_per_seq) {
            if(v > _internal_info.max_split_per_seq) _internal_info.max_split_per_seq = v;
        }
    }

    void pack_schedule_sota(std::optional<std::vector<std::vector<int>>> MNWs,
                             int HRatio = 1,
                             int kvHead = 8,
                             int max_cta = -1) {
        pack_schedule(MNWs, HRatio, kvHead, true, max_cta);
    }

private:
    inline int ceil_div(int a, int b) {
        return (a + b - 1) / b;
    }

    std::vector<int> getMNW(int m_val, int kv_in_CTA) {
        int m = 16, w = 1;
        if (m_val > 32) { m = 64; w = 4; }
        else if (m_val > 16) { m = 32; w = 2; }

        int n = 128;
        if (kv_in_CTA < 32) n = 16;
        else if (kv_in_CTA < 64) n = 32;
        else if (kv_in_CTA < 128) n = 64;
        else n = 128;
        if (m == 64) {
            n = std::max(32, n);
        }

        return {m, n, w};
    }

    std::vector<PackedBox> _tree_heuristics(int node_id, int mm,
                                            const std::vector<int>& inherited_block_id) {
        std::vector<PackedBox> res;
        Node& node = nodes[node_id];

        std::vector<int> current_blocks = inherited_block_id;
        int num_blocks_node = (node.length + block_size - 1) / block_size;
        if (node.block_ptr) {
            current_blocks.insert(current_blocks.end(),
                                  node.block_ptr,
                                  node.block_ptr + num_blocks_node);
        }
        auto shared_blocks = std::make_shared<std::vector<int>>(std::move(current_blocks));
        int current_kv_len = block_size * (int)inherited_block_id.size() + node.length;

        if (node.children.empty()) {
            int S = node.s_value;

            for (int s = 0; s < S; s += mm) {
                PackedBox box;
                int end = std::min(s + mm, S);
                box.q_table.assign(node.seq_indices.begin() + s, node.seq_indices.begin() + end);
                box.block_table_ptr = shared_blocks;
                box.num_seqs_per_CTA = box.q_table.size();
                box.kv_in_CTA = current_kv_len;

                if (!box.q_table.empty()) {
                    box.CTA_rank = split_per_seq[box.q_table[0]];
                    for (int qid : box.q_table) {
                        split_per_seq[qid]++;
                    }
                }
                res.push_back(std::move(box));
            }
        } else {
            int S = node.s_value;
            int current_kv_len = block_size * (int)inherited_block_id.size() + node.length;

            std::vector<int> merged_blocks = inherited_block_id;
            int num_blocks = (node.length + block_size - 1) / block_size;
            if (node.block_ptr) {
                merged_blocks.insert(merged_blocks.end(),
                                     node.block_ptr,
                                     node.block_ptr + num_blocks);
            }

            std::vector<int> ops(node.children.size(), 0);
            std::vector<int> split_child_ids;
            split_child_ids.reserve(node.children.size());
            std::vector<int> merge_child_ids;
            merge_child_ids.reserve(node.children.size());

            int idx = 0;
            for (auto& kv : node.children) {
                int child_id = kv.second;
                Node& child = nodes[child_id];

                int child_s = child.s_value;

                if (S == child_s ||
                    (ceil_div(S, mm) - ceil_div(S - child_s, mm) - ceil_div(child_s, mm)) * current_kv_len + 4 * child_s >= 0)
                {
                    ops[idx] = 1;
                    S -= child_s;
                    merge_child_ids.push_back(child_id);
                } else {
                    split_child_ids.push_back(child_id);
                }
                idx++;
            }

            if (S != 0) {
                size_t batch_size = split_per_seq.size();

                std::vector<uint8_t> is_merged(batch_size, 0);
                int c_idx = 0;
                for (auto& kv : node.children) {
                    if (ops[c_idx] == 1) {
                        int child_id = kv.second;
                        const auto& child_seqs = nodes[child_id].seq_indices;
                        for (int sId : child_seqs) {
                            is_merged[sId] = 1;
                        }
                    }
                    c_idx++;
                }

                std::vector<int> remaining_seqs;
                remaining_seqs.reserve(node.s_value);

                for (int sId : node.seq_indices) {
                    if (is_merged[sId] == 0) {
                        remaining_seqs.push_back(sId);
                    }
                }

                size_t total_rem = remaining_seqs.size();

                for (size_t s = 0; s < total_rem; s += mm) {
                    PackedBox box;
                    size_t end = std::min(s + (size_t)mm, total_rem);
                    box.q_table.assign(remaining_seqs.begin() + s, remaining_seqs.begin() + end);
                    box.block_table_ptr = shared_blocks;
                    box.num_seqs_per_CTA = (int)box.q_table.size();
                    box.kv_in_CTA = current_kv_len;

                    if (!box.q_table.empty()) {
                        box.CTA_rank = split_per_seq[box.q_table[0]];
                        for (int qid : box.q_table) {
                            split_per_seq[qid]++;
                        }
                    }

                    res.push_back(std::move(box));
                }
            }

            for (int child_id : split_child_ids) {
                auto child_boxes = _tree_heuristics(child_id, mm, {});
                res.insert(res.end(),
                           std::make_move_iterator(child_boxes.begin()),
                           std::make_move_iterator(child_boxes.end()));
            }

            for (int child_id : merge_child_ids) {
                auto child_boxes = _tree_heuristics(child_id, mm, merged_blocks);
                res.insert(res.end(),
                           std::make_move_iterator(child_boxes.begin()),
                           std::make_move_iterator(child_boxes.end()));
            }
        }
        return res;
    }

    std::vector<PackedBox> balancePack(std::vector<PackedBox>& boxes, int kvHead, int HRatio = 1) {
        if (boxes.empty()) return {};

        std::vector<PackedBox> cropPack;
        cropPack.reserve(boxes.size() * 2);

        long total_blocks = 0;
        int total_packs = 0;
        for (const auto& box : boxes) {
            if (box.kv_in_CTA >= 128) {
                total_blocks += box.block_table_ptr->size();
                total_packs++;
            }
        }

        if (total_packs != 0) {
            double avg_blocks = (double)total_blocks / total_packs;
            double threshold = avg_blocks * 5.0;

            for (auto& pack : boxes) {
                size_t current_size = pack.block_table_ptr->size();

                if (current_size > threshold) {
                    int num_splits = ceil_div((int)current_size, (int)avg_blocks);
                    int split_size = ceil_div((int)current_size, num_splits);

                    for (int i = 0; i < num_splits; ++i) {
                        int start = i * split_size;
                        int end = std::min((i + 1) * split_size, (int)current_size);
                        if (start >= end) break;

                        PackedBox new_pack;
                        new_pack.q_table = pack.q_table;
                        new_pack.num_seqs_per_CTA = pack.num_seqs_per_CTA;

                        auto new_vec = std::make_shared<std::vector<int>>();
                        new_vec->reserve(end - start);
                        new_vec->assign(pack.block_table_ptr->begin() + start,
                                        pack.block_table_ptr->begin() + end);
                        new_pack.block_table_ptr = new_vec;

                        int offset_tokens = i * split_size * block_size;
                        int chunk_cap_tokens = split_size * block_size;
                        int remaining_tokens = pack.kv_in_CTA - offset_tokens;

                        new_pack.kv_in_CTA = std::min(remaining_tokens, chunk_cap_tokens);

                        if (i == 0) {
                            new_pack.CTA_rank = pack.CTA_rank;
                        } else {
                            if (!new_pack.q_table.empty()) {
                                new_pack.CTA_rank = split_per_seq[new_pack.q_table[0]];
                                for (int qid : new_pack.q_table) {
                                    split_per_seq[qid]++;
                                }
                            }
                        }

                        cropPack.push_back(std::move(new_pack));
                    }
                } else {
                    cropPack.push_back(std::move(pack));
                }
            }
        } else {
            cropPack = std::move(boxes);
        }

        int threshold_cnt;
        if (kvHead <= 4) threshold_cnt = 54;
        else if (kvHead <= 8) threshold_cnt = 27;
        else if (kvHead <= 16) threshold_cnt = 13;
        else if (kvHead <= 32) threshold_cnt = 6;
        else threshold_cnt = 4;

        if (cropPack.size() >= threshold_cnt) return cropPack;

        auto get_max_split = [&]() {
            int m = 0;
            for(int v : split_per_seq) if(v > m) m = v;
            return m;
        };

        while (cropPack.size() < threshold_cnt && get_max_split() < 32) {
            int longest_idx = -1;
            size_t max_len = 0;
            for (size_t i = 0; i < cropPack.size(); ++i) {
                if (cropPack[i].block_table_ptr->size() > max_len) {
                    max_len = cropPack[i].block_table_ptr->size();
                    longest_idx = (int)i;
                }
            }

            if (longest_idx == -1) break;

            PackedBox pack = std::move(cropPack[longest_idx]);
            if (longest_idx != cropPack.size() - 1) {
                cropPack[longest_idx] = std::move(cropPack.back());
            }
            cropPack.pop_back();

            int total_blks = (int)pack.block_table_ptr->size();

            if (total_blks < 8) {
                cropPack.push_back(std::move(pack));
                break;
            }

            int half = ceil_div(total_blks, 2);

            PackedBox p1;
            p1.q_table = pack.q_table;
            p1.num_seqs_per_CTA = pack.num_seqs_per_CTA;
            p1.CTA_rank = pack.CTA_rank;
            p1.kv_in_CTA = std::min(pack.kv_in_CTA, half * block_size);

            auto vec1 = std::make_shared<std::vector<int>>();
            vec1->assign(pack.block_table_ptr->begin(), pack.block_table_ptr->begin() + half);
            p1.block_table_ptr = vec1;

            PackedBox p2;
            p2.q_table = pack.q_table;
            p2.num_seqs_per_CTA = pack.num_seqs_per_CTA;

            int p2_reduce = half * block_size;
            int p2_limit = (total_blks - half) * block_size;
            p2.kv_in_CTA = std::min(pack.kv_in_CTA - p2_reduce, p2_limit);

            auto vec2 = std::make_shared<std::vector<int>>();
            vec2->assign(pack.block_table_ptr->begin() + half, pack.block_table_ptr->end());
            p2.block_table_ptr = vec2;
            p2.CTA_rank = split_per_seq[p2.q_table[0]];

            for(int qid : p1.q_table) split_per_seq[qid]++;

            cropPack.push_back(std::move(p1));
            cropPack.push_back(std::move(p2));
        }

        return cropPack;
    }

    std::vector<PackedBox> balancePackSota(const std::vector<PackedBox>& boxes,
                                           int kvHead,
                                           int HRatio = 1,
                                           int max_cta = -1) {
        // 设计目标：
        //  1）严格保留 correctness，只拆 KV，不拆 Q，保持 block_table 连续性和 CTA_rank / split_per_seq 语义；
        //  2）仅做一阶段的 MNW 结构性预拆分，去掉削峰阶段；
        //  3）CTA 额度自适应：随 base_cta 增大而相对收紧，随 KV 不均衡度增大而适当放宽。

        if (boxes.empty()) return {};

        // 1. 初始 crop：_tree_heuristics 的输出已经正确设置了 split_per_seq / CTA_rank
        std::vector<PackedBox> crop = boxes;
        if (crop.empty()) return crop;

        const int base_cta = (int)crop.size();

        // 与 baseline 相同的最小 CTA 数逻辑，用于确定合理的基础增量上限
        int threshold_cnt;
        if      (kvHead <= 4)  threshold_cnt = 54;
        else if (kvHead <= 8)  threshold_cnt = 27;
        else if (kvHead <= 16) threshold_cnt = 13;
        else if (kvHead <= 32) threshold_cnt = 6;
        else                   threshold_cnt = 4;

        // 根据初始 crop 的 KV 分布计算不均衡度 imbalance_ratio = max_kv / avg_kv
        double total_kv = 0.0;
        double max_kv   = 0.0;
        for (const auto& b : crop) {
            double kv = (double)std::max(1, b.kv_in_CTA);
            total_kv += kv;
            if (kv > max_kv) max_kv = kv;
        }
        double avg_kv = base_cta > 0 ? (total_kv / (double)base_cta) : 1.0;
        if (avg_kv <= 0.0) avg_kv = 1.0;
        double imbalance_ratio = max_kv / std::max(1.0, avg_kv);

        // base_factor：随 base_cta 增大而逐渐减小的基础放大因子（整体偏保守）
        int safe_base_cta = std::max(1, base_cta);
        double base_factor = 1.0 + 6.0 / (double)safe_base_cta;

        // imbalance_factor：随 KV 不均衡度增大而略微增加的系数（防止过度放大）
        double imbalance_factor = 1.0 + imbalance_ratio / 15;

        double factor = base_factor * imbalance_factor;
        if (factor < 1.0) factor = 1.0;
        if (factor > 2.0) factor = 2.0;

        // 结合 base_cta / threshold_cnt 和 imbalance，得到本树的 CTA 上限
        double target_mul = (double)base_cta * factor;
        // 对于 threshold_cnt，加权也适当收紧，避免小树被一次性放大太多 CTA
        double target_add = (double)base_cta + (double)threshold_cnt * (imbalance_factor * 0.7);
        double target = std::min(target_mul, target_add);
        if (target < (double)base_cta) target = (double)base_cta;
        int CTA_target = (int)(target + 0.999999);  // 近似向上取整


        if (std::getenv("PAT_DEBUG_IMBALANCE") != nullptr) {
            std::cout << "[balancePackSota] kvHead=" << kvHead
                << ", HRatio=" << HRatio
                << ", base_cta=" << base_cta
                // << ", c_max=" << global_max_cost
                // << ", c_avg=" << global_avg_cost
                << ", ratio=" << imbalance_ratio
                << ", CTA_target=" << CTA_target
                << std::endl;
        }

        // 进一步受外部传入的 CTA 上限约束（例如 SM 数量 × 系数）。
        if (max_cta > 0) {
            CTA_target = std::min(CTA_target, max_cta);
        }

        // 复用 baseline 的 token 计数与 CTA_rank / split_per_seq 更新规则，仅做 KV 拆分
        auto split_box_even_blocks = [&](PackedBox& src, int split_k, std::vector<PackedBox>& out) {
            if (!src.block_table_ptr) {
                out.push_back(std::move(src));
                return;
            }

            int total_blocks = (int)src.block_table_ptr->size();
            if (split_k <= 1 || total_blocks <= 1) {
                out.push_back(std::move(src));
                return;
            }

            int base_sz = total_blocks / split_k;
            int rem  = total_blocks % split_k;
            int start = 0;

            for (int i = 0; i < split_k; ++i) {
                int blk_cnt = base_sz + (i < rem ? 1 : 0);
                int end = start + blk_cnt;
                if (blk_cnt <= 0 || start >= total_blocks) break;
                if (end > total_blocks) end = total_blocks;

                PackedBox sub;
                sub.q_table          = src.q_table;
                sub.num_seqs_per_CTA = src.num_seqs_per_CTA;
                sub.split_depth      = src.split_depth + 1;

                auto vec = std::make_shared<std::vector<int>>();
                vec->assign(src.block_table_ptr->begin() + start,
                            src.block_table_ptr->begin() + end);
                sub.block_table_ptr = vec;

                int offset_tokens    = start * block_size;
                int chunk_cap_tokens = (end - start) * block_size;
                int remain_tokens    = src.kv_in_CTA - offset_tokens;
                sub.kv_in_CTA        = std::max(0, std::min(remain_tokens, chunk_cap_tokens));

                if (i == 0) {
                    sub.CTA_rank = src.CTA_rank;
                } else if (!sub.q_table.empty()) {
                    sub.CTA_rank = split_per_seq[sub.q_table[0]];
                    for (int qid : sub.q_table) {
                        split_per_seq[qid]++;
                    }
                }

                out.push_back(std::move(sub));
                start = end;
                if (start >= total_blocks) break;
            }
        };

        const int max_split_per_seq = 32;

        // 第一阶段：按 MNW 理想 N 对每个盒子做一次“结构性预拆分”，使 kv_in_CTA 更接近理想 N 或其整数倍。
        auto mnw_pre_split = [&](std::vector<PackedBox>& cur_boxes) {
            std::vector<PackedBox> new_boxes;
            new_boxes.reserve(cur_boxes.size() * 2);

            int cur_cta = (int)cur_boxes.size();
            const int max_k_per_box = 8;  // 单盒最大拆成 8 份，防止过细

            for (auto& b : cur_boxes) {
                if (!b.block_table_ptr || b.block_table_ptr->size() < 2) {
                    new_boxes.push_back(std::move(b));
                    continue;
                }

                int qn = std::max(1, (int)b.q_table.size());
                int kv = std::max(1, b.kv_in_CTA);

                int eff_heads = std::max(1, HRatio);
                int m_val = qn * eff_heads;
                std::vector<int> target_mnw = getMNW(m_val, kv);
                if ((int)target_mnw.size() < 2) {
                    new_boxes.push_back(std::move(b));
                    continue;
                }
                int ideal_n = target_mnw[1];
                if (ideal_n <= 0) {
                    new_boxes.push_back(std::move(b));
                    continue;
                }

                // 已经在理想 N 附近，或本身比一个 tile 还小，则不在预拆阶段动它
                if (kv <= ideal_n * 13 / 10) {  // 约 1.3 倍以内认为足够接近
                    new_boxes.push_back(std::move(b));
                    continue;
                }

                // 目标拆分份数：kv / ideal_n，向最近的整数取整，限制在 [2, max_k_per_box]
                int split_k = kv / ideal_n;
                int rem_n  = kv % ideal_n;
                if (split_k <= 1) {
                    new_boxes.push_back(std::move(b));
                    continue;
                }
                // 简单双向舍入：如果余数超过半个 ideal_n，就向上加 1
                if (rem_n * 2 > ideal_n) split_k += 1;
                if (split_k < 2) split_k = 2;
                if (split_k > max_k_per_box) split_k = max_k_per_box;

                // CTA 预算约束：不超过 CTA_target
                if (CTA_target > 0) {
                    int max_extra = CTA_target - cur_cta;
                    if (max_extra <= 0) {
                        new_boxes.push_back(std::move(b));
                        continue;
                    }
                    if (split_k - 1 > max_extra) {
                        split_k = max_extra + 1;
                    }
                }

                // per-seq 拆分深度约束：确保不会超过 max_split_per_seq
                int cur_max_seq_split = 0;
                for (int qid : b.q_table) {
                    if (qid >= 0 && qid < (int)split_per_seq.size()) {
                        if (split_per_seq[qid] > cur_max_seq_split) {
                            cur_max_seq_split = split_per_seq[qid];
                        }
                    }
                }
                if (cur_max_seq_split >= max_split_per_seq) {
                    new_boxes.push_back(std::move(b));
                    continue;
                }
                int allowed_extra = max_split_per_seq - cur_max_seq_split;
                if (split_k - 1 > allowed_extra) {
                    split_k = allowed_extra + 1;
                }

                if (split_k <= 1) {
                    new_boxes.push_back(std::move(b));
                    continue;
                }

                split_box_even_blocks(b, split_k, new_boxes);
                cur_cta += (split_k - 1);
            }

            cur_boxes = std::move(new_boxes);
        };

        // 执行 MNW 预拆分，使各盒子的 kv_in_CTA 更贴近其 MNW 理想配置
        mnw_pre_split(crop);

        // 一阶段拆分完成后直接返回，不再进行削峰二分
        return crop;
    }
};