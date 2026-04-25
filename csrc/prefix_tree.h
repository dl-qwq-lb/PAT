#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <queue>
#include <chrono>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <climits>

#include <cuda_runtime_api.h>

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
                       bool use_sota = false) {

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
            packed_boxes = balancePackSota(std::move(packed_boxes), kvHead, HRatio);
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
                             int kvHead = 8) {
        pack_schedule(MNWs, HRatio, kvHead, true);
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

        const int threshold_cnt = _bp_cta_limit_by_kvhead(kvHead);

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

private:
    static int _bp_cta_limit_by_kvhead(int kvHead) {
        if (kvHead <= 4) return 54;
        if (kvHead <= 8) return 27;
        if (kvHead <= 16) return 13;
        if (kvHead <= 32) return 6;
        return 4;
    }

    static int _bp_floor_pow2(int x) {
        if (x <= 0) return 0;
        unsigned int v = (unsigned int)x;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return (int)(v - (v >> 1));
    }

    static int _bp_align_down_multiple(int x, int m) {
        if (m <= 0) return x;
        return (x / m) * m;
    }

    static int _bp_align_up_multiple(int x, int m) {
        if (m <= 0) return x;
        return ((x + m - 1) / m) * m;
    }

    int _bp_aligned_piece_tokens(int ideal_len, int min_len) const {
        int t = std::max(min_len, _bp_floor_pow2(ideal_len));
        t = _bp_align_down_multiple(t, block_size);
        if (t < block_size) t = block_size;
        return t;
    }

    static int _bp_infer_max_cta_from_sm_cached() {
        static int cached_device = -1;
        static int cached_max_cta = -1;

        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return -1;
        if (cached_max_cta > 0 && cached_device == dev) return cached_max_cta;

        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return -1;
        cached_device = dev;
        cached_max_cta = std::max(1, prop.multiProcessorCount);
        return cached_max_cta;
    }

    struct _bp_cost_model {
        int HR = 1;
        static constexpr double a = 0.1;
        static constexpr double b = 0.1;
        static constexpr double c = 2.0;

        double operator()(int q_tokens, int kv_tokens) const {
            return a * (double)q_tokens * (double)HR + b * (double)kv_tokens + c * (double)q_tokens * (double)kv_tokens * (double)HR;
        }
    };

    void _bp_split_box_even_blocks(PackedBox src,
                                   int split_k,
                                   std::vector<PackedBox>& out,
                                   int min_kv_tokens_per_piece) {
        if (!src.block_table_ptr) {
            out.push_back(std::move(src));
            return;
        }

        const int total_blocks = (int)src.block_table_ptr->size();
        if (split_k <= 1 || total_blocks <= 1) {
            out.push_back(std::move(src));
            return;
        }

        const int total_kv_tokens = std::max(0, src.kv_in_CTA);
        int start = 0;

        for (int i = 0; i < split_k; ++i) {
            const int pieces_left = split_k - i;
            const int remaining_blocks = total_blocks - start;
            if (remaining_blocks <= 0) break;

            int end = total_blocks;
            if (pieces_left > 1) {
                const int remaining_kv_tokens = std::max(0, total_kv_tokens - start * block_size);
                const int ideal_len = ceil_div(std::max(1, remaining_kv_tokens), pieces_left);
                const int aligned_len = _bp_aligned_piece_tokens(ideal_len, min_kv_tokens_per_piece);

                int desired_blocks = std::max(1, aligned_len / block_size);
                const int max_blocks_this = std::max(1, remaining_blocks - (pieces_left - 1));
                if (desired_blocks > max_blocks_this) desired_blocks = max_blocks_this;

                end = start + desired_blocks;
                if (end <= start) end = start + 1;
                if (end > total_blocks) end = total_blocks;
            }

            if (end - start <= 0 || start >= total_blocks) break;

            PackedBox sub;
            sub.q_table = src.q_table;
            sub.num_seqs_per_CTA = src.num_seqs_per_CTA;

            auto vec = std::make_shared<std::vector<int>>();
            vec->assign(src.block_table_ptr->begin() + start, src.block_table_ptr->begin() + end);
            sub.block_table_ptr = vec;

            const int offset_tokens = start * block_size;
            const int chunk_cap_tokens = (end - start) * block_size;
            const int remain_tokens = src.kv_in_CTA - offset_tokens;
            sub.kv_in_CTA = std::max(0, std::min(remain_tokens, chunk_cap_tokens));

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
    }

public:
    std::vector<PackedBox> balancePackSota(std::vector<PackedBox> boxes, int kvHead, int HRatio = 1) {
        // NOTE: 使用 move 语义避免深拷贝。
        std::vector<PackedBox> crop = std::move(boxes);
        if (crop.empty()) return {};
        const int base_cta = (int)crop.size();
        const int cta_limit = _bp_cta_limit_by_kvhead(kvHead);
        const int cta_cap = _bp_infer_max_cta_from_sm_cached();
        const _bp_cost_model cost_fn{.HR = std::max(1, HRatio)};
        bool budget_phase = (base_cta < cta_limit);
        const int max_split_n = 16;
        const int min_blocks_per_piece = 4;

        int batch_min_kv = INT_MAX;
        long long total_kv_all = 0;
        for (const auto& b : crop) {
            const int kv = std::max(0, b.kv_in_CTA);
            total_kv_all += (long long)kv;
            if (kv > 0 && kv < batch_min_kv) batch_min_kv = kv;
        }
        if (batch_min_kv == INT_MAX) batch_min_kv = 1;

        const int global_L_kv = (cta_limit > 0)
            ? ceil_div((int)std::max(1LL, total_kv_all), cta_limit)
            : std::max(1, batch_min_kv);
        const int global_L_kv_aligned = _bp_align_up_multiple(global_L_kv, block_size);

        bool budget_strict_lkv = budget_phase;
        const int min_kv_tokens_budget_strict = std::max(min_blocks_per_piece * block_size, global_L_kv_aligned);
        const int min_kv_tokens_budget_relaxed = min_blocks_per_piece * block_size;
        const int min_kv_tokens_nonbudget = std::max(min_blocks_per_piece * block_size, batch_min_kv);

        const int HR = std::max(1, HRatio);
        const int denom = std::max(1, (cta_cap > 0 ? cta_cap : cta_limit));

        auto waves_est = [&](int n_cta) -> int {
            if (budget_phase) n_cta = std::max(n_cta, cta_limit);
            return (n_cta + denom - 1) / denom;
        };

        auto recompute_cost_stats = [&](const std::vector<PackedBox>& v,
                                        double& sum_cost_out,
                                        double& max_cost_out,
                                        double& second_max_cost_out,
                                        int& max_cost_count_out) {
            sum_cost_out = 0.0;
            max_cost_out = 0.0;
            second_max_cost_out = 0.0;
            max_cost_count_out = 0;
            for (const auto& b : v) {
                const int q = std::max(0, (int)b.q_table.size());
                const int kv = std::max(0, b.kv_in_CTA);
                const double cst = cost_fn(q, kv);
                sum_cost_out += cst;
                if (cst > max_cost_out) {
                    second_max_cost_out = max_cost_out;
                    max_cost_out = cst;
                    max_cost_count_out = 1;
                } else if (cst == max_cost_out) {
                    max_cost_count_out += 1;
                } else if (cst > second_max_cost_out) {
                    second_max_cost_out = cst;
                }
            }
        };

        const int cand_k[] = {2, 3, 4, 6, 8, 12, 16};
        int total_cta = (int)crop.size();
        int applied_actions = 0;
        const int max_actions = 64;
        bool plateau_unlock_used = false;
        bool plateau_round_active = false;
        int plateau_target_kv = -1;

        while (applied_actions < max_actions) {
            double sum_cost_it = 0.0;
            double max_cost_it = 0.0;
            double second_max_cost_it = 0.0;
            int max_cost_count_it = 0;
            recompute_cost_stats(crop, sum_cost_it, max_cost_it, second_max_cost_it, max_cost_count_it);
            const double avg_cost_it = (crop.empty() ? 0.0 : (sum_cost_it / (double)crop.size()));
            if (!(avg_cost_it > 0.0) || !(max_cost_it > 0.0)) break;

            const int waves_before = waves_est(total_cta);

            int best_i = -1;
            int best_k = 1;
            int best_added_cta = 0;
            double best_gain = budget_phase ? -1e100 : 0.0;
            int plateau_i = -1;
            int plateau_kv = -1;
            const bool budget_relaxed = (budget_phase && !budget_strict_lkv);

            const int min_kv_tokens_per_piece = budget_phase
                ? (budget_strict_lkv ? min_kv_tokens_budget_strict : min_kv_tokens_budget_relaxed)
                : min_kv_tokens_nonbudget;

            for (size_t i = 0; i < crop.size(); ++i) {
                const auto& b = crop[i];
                const int q = std::max(0, (int)b.q_table.size());
                const int kv = std::max(0, b.kv_in_CTA);
                const double old_cost = cost_fn(q, kv);
                if (!budget_phase && old_cost <= avg_cost_it) continue;

                const bool is_bottleneck_cta = (old_cost == max_cost_it);
                const double max_cost_others = (is_bottleneck_cta && max_cost_count_it == 1) ? second_max_cost_it : max_cost_it;

                int max_by_seq = 1;
                if (!b.q_table.empty()) {
                    int add_lim = INT_MAX;
                    for (int qid : b.q_table) {
                        if (qid < 0 || qid >= (int)split_per_seq.size()) continue;
                        add_lim = std::min(add_lim, 32 - split_per_seq[qid]);
                    }
                    if (add_lim == INT_MAX) add_lim = 0;
                    if (add_lim < 0) add_lim = 0;
                    max_by_seq = 1 + add_lim;
                }

                int blocks = 0;
                if (b.block_table_ptr) blocks = std::max(0, (int)b.block_table_ptr->size());
                const int max_by_blocks = (blocks > 0 ? std::max(1, blocks) : 1);

                int max_k = std::min({max_split_n, max_by_blocks, max_by_seq});
                if (budget_phase) {
                    const int remaining_slots = std::max(0, cta_limit - total_cta);
                    max_k = std::min(max_k, 1 + remaining_slots);
                }
                if (max_k <= 1) continue;

                auto can_split_k = [&](int k) -> bool {
                    if (k <= 1) return true;
                    if (blocks > 0 && (blocks + k - 1) / k < min_blocks_per_piece) return false;
                    if ((kv + k - 1) / k < min_kv_tokens_per_piece) return false;
                    return true;
                };

                auto try_k = [&](int k) {
                    if (k > max_k) return;
                    if (budget_relaxed && k != 2) return;
                    if (!can_split_k(k)) return;

                    const int added_cta = k - 1;
                    if (budget_phase && (total_cta + added_cta > cta_limit)) return;
                    const int waves_after = waves_est(total_cta + added_cta);

                    const int ideal_len = ceil_div(std::max(0, kv), k);
                    const int kv_piece_aligned = _bp_aligned_piece_tokens(ideal_len, min_kv_tokens_per_piece);
                    const int kv_last = std::max(0, kv - (k - 1) * kv_piece_aligned);
                    const int kv_piece_max = std::max(kv_piece_aligned, kv_last);

                    const double dup_q_cost = _bp_cost_model::a * (double)q * (double)HR;
                    const double split_cta_cost_eff = cost_fn(q, kv_piece_max) + (double)added_cta * dup_q_cost;
                    double new_max_cost = std::max(max_cost_others, split_cta_cost_eff);
                    if (budget_phase) {
                        new_max_cost = std::min(new_max_cost, max_cost_it);
                    }

                    const double est_time_before = (double)waves_before * max_cost_it;
                    const double est_time_after  = (double)waves_after  * new_max_cost;
                    const double gain = est_time_before - est_time_after;
                    if (!budget_phase) {
                        if (gain > best_gain) {
                            best_gain = gain;
                            best_i = (int)i;
                            best_k = k;
                            best_added_cta = added_cta;
                        }
                        const bool plateau_enter_candidate = (!plateau_round_active && !plateau_unlock_used &&
                                                              max_cost_count_it > 1 && k == 2 &&
                                                              is_bottleneck_cta && waves_after == waves_before &&
                                                              std::abs(gain) <= 1e-9);
                        const bool plateau_round_candidate = (plateau_round_active && k == 2 && is_bottleneck_cta &&
                                                              waves_after == waves_before && kv == plateau_target_kv);
                        const bool plateau_candidate = plateau_enter_candidate || plateau_round_candidate;
                        if (plateau_candidate && kv > plateau_kv) {
                            plateau_kv = kv;
                            plateau_i = (int)i;
                        }
                    } else if (budget_relaxed) {
                        // 放开 L_kv 后沿用原候选框架，但只做全局最长 chunk 的贪心二分。
                        const double score = (double)kv;
                        const bool better_score = (score > best_gain);
                        const bool tie_more_fill = (score == best_gain && added_cta > best_added_cta);
                        if (better_score || tie_more_fill) {
                            best_gain = score;
                            best_i = (int)i;
                            best_k = k;
                            best_added_cta = added_cta;
                        }
                    } else {
                        const bool better_gain = (gain > best_gain);
                        const bool tie_more_fill = (gain == best_gain && added_cta > best_added_cta);
                        if (better_gain || tie_more_fill) {
                            best_gain = gain;
                            best_i = (int)i;
                            best_k = k;
                            best_added_cta = added_cta;
                        }
                    }
                };

                for (int k : cand_k) {
                    try_k(k);
                }

                if (budget_phase) {
                    const int remaining_slots = std::max(0, cta_limit - total_cta);
                    if (remaining_slots > 0) {
                        const int k_fill = std::min(max_k, 1 + remaining_slots);
                        try_k(k_fill);
                    }
                }
            }

            if (!budget_phase) {
                if (plateau_round_active) {
                    if (plateau_i >= 0) {
                        best_i = plateau_i;
                        best_k = 2;
                        best_added_cta = 1;
                    } else {
                        break;
                    }
                } else if (best_i < 0 || best_k <= 1 || best_gain <= 0.0) {
                    if (plateau_i >= 0) {
                        best_i = plateau_i;
                        best_k = 2;
                        best_added_cta = 1;
                        plateau_unlock_used = true;
                        plateau_round_active = true;
                        plateau_target_kv = plateau_kv;
                    } else {
                        break;
                    }
                }
            } else {
                if (best_i < 0 || best_k <= 1) {
                    if (budget_strict_lkv && total_cta < cta_limit) {
                        budget_strict_lkv = false;
                        continue;
                    }
                    break;
                }
            }

            PackedBox to_split = std::move(crop[best_i]);
            std::vector<PackedBox> pieces;
            pieces.reserve(best_k);
            _bp_split_box_even_blocks(std::move(to_split), best_k, pieces, min_kv_tokens_per_piece);
            if (pieces.size() < 2) {
                crop[best_i] = std::move(pieces.empty() ? to_split : pieces[0]);
                break;
            }

            crop[best_i] = std::move(pieces[0]);
            for (size_t j = 1; j < pieces.size(); ++j) {
                crop.push_back(std::move(pieces[j]));
            }
            total_cta += (int)pieces.size() - 1;
            applied_actions += 1;

            if (budget_phase && total_cta >= cta_limit) {
                budget_phase = false;
                budget_strict_lkv = false;
            }
        }
        return crop;
    }
};