#include <torch/extension.h>    // PyTorch API + pybind11
#include "api.h"
#include "prefix_tree.h"
#include <pybind11/stl.h>

// a wrapper of pat::pat_mha_fwd_kvcache
// pybind11将声明的python函数与参数控制流转到c++，是一份桥接定义文件
void prefix_attn_with_kvcache(
    const at::Tensor &q,
    const at::Tensor &k_cache_paged,
    const at::Tensor &v_cache_paged,
    const at::Tensor &num_split_per_seq,
    const std::vector<at::Tensor> &query_tables,
    const std::vector<at::Tensor> &block_tables,
    const std::vector<at::Tensor> &num_seqs_per_CTAs,
    const std::vector<at::Tensor> &CTA_ranks,
    const std::vector<at::Tensor> &kv_in_CTAs,
    const std::vector<std::vector<int>> &MNW,
    const int32_t max_split_per_seq,
    const int32_t max_seqs_in_CTA,
    const int32_t max_blocks_in_CTA,
    float softmax_scale,
    at::Tensor &out,
    std::optional<at::Tensor> timing_ = std::nullopt
) {
    pat::pat_mha_fwd_kvcache(
        q,
        k_cache_paged,
        v_cache_paged,
        num_split_per_seq,
        query_tables,
        block_tables,
        num_seqs_per_CTAs,
        CTA_ranks,
        kv_in_CTAs,
        MNW,
        max_split_per_seq,
        max_seqs_in_CTA,
        max_blocks_in_CTA,
        softmax_scale,
        out,
        timing_
    );
}

void prefix_attn_with_tree_wrapper(
    const at::Tensor &q,
    const at::Tensor &k_cache_paged,
    const at::Tensor &v_cache_paged,
    PrefixTree& tree,
    float softmax_scale,
    at::Tensor &out
) {
    const KernelInfo& info = tree._internal_info;
    pat::pat_mha_fwd_kvcache(
        q,
        k_cache_paged,
        v_cache_paged,
        info.num_split_per_seq,
        info.q_tables,
        info.block_tables,
        info.num_seqs_per_CTAs,
        info.CTA_ranks,
        info.kv_in_CTAs,
        info.MNWs,
        info.max_split_per_seq,
        info.max_seqs_in_CTA,
        info.max_blocks_in_CTA,
        softmax_scale,
        out,
        std::nullopt
    );
}

PYBIND11_MODULE(_prefix_attn, m) {
    m.doc() = "prefix_attn_with_kvcache PyBind11 binding";
    m.def(
        "prefix_attn_with_kvcache",
        &prefix_attn_with_kvcache,
        py::arg("q"),
        py::arg("k_cache_paged"),
        py::arg("v_cache_paged"),
        py::arg("num_split_per_seq"),
        py::arg("query_tables"),
        py::arg("block_tables"),
        py::arg("num_seqs_per_CTAs"),
        py::arg("CTA_ranks"),
        py::arg("kv_in_CTAs"),
        py::arg("MNW"),
        py::arg("max_split_per_seq"),
        py::arg("max_seqs_in_CTA"),
        py::arg("max_blocks_in_CTA"),
        py::arg("softmax_scale"),
        py::arg("out"),
        py::arg("timing_") = std::nullopt
    );

    py::class_<KernelInfo>(m, "KernelInfo")
        .def("to_gpu", &KernelInfo::to_gpu, py::arg("device"))
        // balance pack debug
        .def_readonly("q_tables",          &KernelInfo::q_tables)  
        .def_readonly("block_tables",      &KernelInfo::block_tables)  
        .def_readonly("num_seqs_per_CTAs", &KernelInfo::num_seqs_per_CTAs)  
        .def_readonly("CTA_ranks",         &KernelInfo::CTA_ranks)  
        .def_readonly("kv_in_CTAs",        &KernelInfo::kv_in_CTAs)  
        .def_readonly("MNWs",              &KernelInfo::MNWs)  
        .def_readonly("num_split_per_seq", &KernelInfo::num_split_per_seq)  
        .def_readonly("max_split_per_seq", &KernelInfo::max_split_per_seq)  
        .def_readonly("max_seqs_in_CTA",   &KernelInfo::max_seqs_in_CTA)  
        .def_readonly("max_blocks_in_CTA", &KernelInfo::max_blocks_in_CTA);

    py::class_<PackedBox>(m, "PackedBox")
        .def_readonly("q_table", &PackedBox::q_table)
        .def_property_readonly("block_table", [](const PackedBox& p){ return *p.block_table_ptr; })
        .def_readonly("num_seqs_per_CTA", &PackedBox::num_seqs_per_CTA)
        .def_readonly("kv_in_CTA", &PackedBox::kv_in_CTA)
        .def_readonly("CTA_rank", &PackedBox::CTA_rank);
    // debug code end here

    py::class_<PrefixTree>(m, "PrefixTreeCPP")
        .def(py::init<int>(), py::arg("block_size"))

        .def("build_radix_tree", [](PrefixTree& tree,
                                    py::object seq_lens_obj,
                                    torch::Tensor block_table)
        {
            int num_seqs = block_table.size(0);
            int max_blocks = block_table.size(1);
            const int* table_ptr = block_table.data_ptr<int>();

            std::vector<int> temp_lens_vec;
            const int* lens_ptr = nullptr;

            if (py::isinstance<torch::Tensor>(seq_lens_obj)) {
                auto t = seq_lens_obj.cast<torch::Tensor>();
                lens_ptr = t.data_ptr<int>();
            } else if (py::isinstance<py::list>(seq_lens_obj)) {
                temp_lens_vec = seq_lens_obj.cast<std::vector<int>>();
                if (temp_lens_vec.size() != num_seqs) throw std::runtime_error("Seq lens list mismatch");
                lens_ptr = temp_lens_vec.data();
            } else {
                throw std::runtime_error("seq_lens must be a List[int] or a CPU Int32 Tensor");
            }
            py::gil_scoped_release release;  // release GIL during building
            tree.build_radix_tree(lens_ptr, table_ptr, num_seqs, max_blocks);
        }, py::arg("seq_lens"), py::arg("block_table"))

        .def("pack_schedule", [](PrefixTree& tree,
                             py::object MNWs_obj,  // optional
                             int HRatio,
                             int kvHead,
                             bool use_sota)
        {
            std::vector<std::vector<int>> MNWs_vec;
            if (!MNWs_obj.is_none()) {
                MNWs_vec = MNWs_obj.cast<std::vector<std::vector<int>>>();
            }
            py::gil_scoped_release release;  // release GIL during packing
            tree.pack_schedule(MNWs_vec, HRatio, kvHead, use_sota);
        }, py::arg("MNWs") = py::none(), 
           py::arg("HRatio") = 1,
           py::arg("kvHead") = 8,
           py::arg("use_sota") = false
        )

        .def("pack_schedule_sota", [](PrefixTree& tree,
                             py::object MNWs_obj,
                             int HRatio,
                             int kvHead)
        {
            std::vector<std::vector<int>> MNWs_vec;
            if (!MNWs_obj.is_none()) {
                MNWs_vec = MNWs_obj.cast<std::vector<std::vector<int>>>();
            }
            py::gil_scoped_release release;
            tree.pack_schedule_sota(MNWs_vec, HRatio, kvHead);
        }, py::arg("MNWs") = py::none(),
           py::arg("HRatio") = 1,
           py::arg("kvHead") = 8
        )

        .def_readonly("kernel_info", &PrefixTree::_internal_info, py::return_value_policy::reference_internal);

    m.def(
        "prefix_attn_with_kvcache",
        &prefix_attn_with_tree_wrapper,
        py::arg("q"),
        py::arg("k_cache_paged"),
        py::arg("v_cache_paged"),
        py::arg("tree"),
        py::arg("softmax_scale"),
        py::arg("out")
    );
}
