#include <torch/extension.h>

std::vector<torch::Tensor> wa_wirelength_timing_weight_cuda(torch::Tensor node_pos,
                                                            torch::Tensor timing_pin_grad,
                                                            torch::Tensor pin_id2node_id,
                                                            torch::Tensor pin_rel_cpos,
                                                            torch::Tensor node2pin_list,
                                                            torch::Tensor node2pin_list_end,
                                                            torch::Tensor hyperedge_list,
                                                            torch::Tensor hyperedge_list_end,
                                                            torch::Tensor net_mask,
                                                            torch::Tensor net_weight,
                                                            torch::Tensor hpwl_scale,
                                                            float gamma,
                                                            bool deterministic);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> wa_wirelength_timing_weight(torch::Tensor node_pos,
                                                       torch::Tensor timing_pin_grad,
                                                       torch::Tensor pin_id2node_id,
                                                       torch::Tensor pin_rel_cpos,
                                                       torch::Tensor node2pin_list,
                                                       torch::Tensor node2pin_list_end,
                                                       torch::Tensor hyperedge_list,
                                                       torch::Tensor hyperedge_list_end,
                                                       torch::Tensor net_mask,
                                                       torch::Tensor net_weight,
                                                       torch::Tensor hpwl_scale,
                                                       float gamma,
                                                       bool deterministic) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(timing_pin_grad);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(pin_rel_cpos);
    CHECK_INPUT(node2pin_list);
    CHECK_INPUT(node2pin_list_end);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);
    CHECK_INPUT(net_mask);
    CHECK_INPUT(net_weight);
    CHECK_INPUT(hpwl_scale);

    return wa_wirelength_timing_weight_cuda(
        node_pos, timing_pin_grad, pin_id2node_id, pin_rel_cpos, node2pin_list, node2pin_list_end, hyperedge_list, hyperedge_list_end, net_mask, net_weight, hpwl_scale, gamma, deterministic);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("merged_wl_loss_grad_timing", &wa_wirelength_timing_weight, "calculate timing-driven WA wirelength pin grad"); }