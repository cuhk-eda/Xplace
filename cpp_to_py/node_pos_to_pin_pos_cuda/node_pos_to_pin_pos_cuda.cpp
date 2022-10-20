#include <torch/extension.h>

torch::Tensor node_pos_to_pin_pos_cuda_forward(torch::Tensor node_pos,
                                               torch::Tensor pin_id2node_id,
                                               torch::Tensor pin_rel_cpos);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor node_pos_to_pin_pos_forward(torch::Tensor node_pos,
                                          torch::Tensor pin_id2node_id,
                                          torch::Tensor pin_rel_cpos) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(pin_rel_cpos);

    return node_pos_to_pin_pos_cuda_forward(node_pos, pin_id2node_id, pin_rel_cpos);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &node_pos_to_pin_pos_forward, "get pin pos from node pos");
}