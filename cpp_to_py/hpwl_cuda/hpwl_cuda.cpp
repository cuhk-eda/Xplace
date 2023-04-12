#include <torch/extension.h>

torch::Tensor hpwl_cuda(torch::Tensor pos, torch::Tensor hyperedge_list, torch::Tensor hyperedge_list_end);
torch::Tensor node_pos_to_pin_pos_cuda(torch::Tensor node_pos,
                                       torch::Tensor pin_id2node_id,
                                       torch::Tensor pin_rel_cpos);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor hpwl(torch::Tensor pos, torch::Tensor hyperedge_list, torch::Tensor hyperedge_list_end) {
    CHECK_INPUT(pos);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);

    return hpwl_cuda(pos, hyperedge_list, hyperedge_list_end);
}

torch::Tensor node_pos_to_pin_pos(torch::Tensor node_pos, torch::Tensor pin_id2node_id, torch::Tensor pin_rel_cpos) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(pin_rel_cpos);

    return node_pos_to_pin_pos_cuda(node_pos, pin_id2node_id, pin_rel_cpos);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hpwl", &hpwl, "get HPWL wirelength from hyperedge");
    m.def("node_pos_to_pin_pos", &node_pos_to_pin_pos, "get pin pos from node pos");
}