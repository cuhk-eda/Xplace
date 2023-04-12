#include <torch/extension.h>

torch::Tensor masked_scale_hpwl_sum_cuda(torch::Tensor node_pos,
                                         torch::Tensor pin_id2node_id,
                                         torch::Tensor pin_rel_cpos,
                                         torch::Tensor hyperedge_list,
                                         torch::Tensor hyperedge_list_end,
                                         torch::Tensor net_mask,
                                         torch::Tensor hpwl_scale);

std::vector<torch::Tensor> wa_wirelength_cuda(torch::Tensor node_pos,
                                              torch::Tensor pin_id2node_id,
                                              torch::Tensor pin_rel_cpos,
                                              torch::Tensor node2pin_list,
                                              torch::Tensor node2pin_list_end,
                                              torch::Tensor hyperedge_list,
                                              torch::Tensor hyperedge_list_end,
                                              torch::Tensor net_mask,
                                              float gamma,
                                              bool deterministic);

std::vector<torch::Tensor> wa_wirelength_hpwl_cuda(torch::Tensor node_pos,
                                                   torch::Tensor pin_id2node_id,
                                                   torch::Tensor pin_rel_cpos,
                                                   torch::Tensor node2pin_list,
                                                   torch::Tensor node2pin_list_end,
                                                   torch::Tensor hyperedge_list,
                                                   torch::Tensor hyperedge_list_end,
                                                   torch::Tensor net_mask,
                                                   float gamma,
                                                   bool deterministic);

std::vector<torch::Tensor> wa_wirelength_masked_scale_hpwl_cuda(torch::Tensor node_pos,
                                                                torch::Tensor pin_id2node_id,
                                                                torch::Tensor pin_rel_cpos,
                                                                torch::Tensor node2pin_list,
                                                                torch::Tensor node2pin_list_end,
                                                                torch::Tensor hyperedge_list,
                                                                torch::Tensor hyperedge_list_end,
                                                                torch::Tensor net_mask,
                                                                torch::Tensor hpwl_scale,
                                                                float gamma,
                                                                bool deterministic);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor masked_scale_hpwl_sum(torch::Tensor node_pos,
                                    torch::Tensor pin_id2node_id,
                                    torch::Tensor pin_rel_cpos,
                                    torch::Tensor hyperedge_list,
                                    torch::Tensor hyperedge_list_end,
                                    torch::Tensor net_mask,
                                    torch::Tensor hpwl_scale) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(pin_rel_cpos);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);
    CHECK_INPUT(net_mask);
    CHECK_INPUT(hpwl_scale);
    return masked_scale_hpwl_sum_cuda(
        node_pos, pin_id2node_id, pin_rel_cpos, hyperedge_list, hyperedge_list_end, net_mask, hpwl_scale);
}

std::vector<torch::Tensor> wa_wirelength(torch::Tensor node_pos,
                                         torch::Tensor pin_id2node_id,
                                         torch::Tensor pin_rel_cpos,
                                         torch::Tensor node2pin_list,
                                         torch::Tensor node2pin_list_end,
                                         torch::Tensor hyperedge_list,
                                         torch::Tensor hyperedge_list_end,
                                         torch::Tensor net_mask,
                                         float gamma,
                                         bool deterministic) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(pin_rel_cpos);
    CHECK_INPUT(node2pin_list);
    CHECK_INPUT(node2pin_list_end);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);
    CHECK_INPUT(net_mask);

    return wa_wirelength_cuda(node_pos,
                              pin_id2node_id,
                              pin_rel_cpos,
                              node2pin_list,
                              node2pin_list_end,
                              hyperedge_list,
                              hyperedge_list_end,
                              net_mask,
                              gamma,
                              deterministic);
}

std::vector<torch::Tensor> wa_wirelength_hpwl(torch::Tensor node_pos,
                                              torch::Tensor pin_id2node_id,
                                              torch::Tensor pin_rel_cpos,
                                              torch::Tensor node2pin_list,
                                              torch::Tensor node2pin_list_end,
                                              torch::Tensor hyperedge_list,
                                              torch::Tensor hyperedge_list_end,
                                              torch::Tensor net_mask,
                                              float gamma,
                                              bool deterministic) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(pin_rel_cpos);
    CHECK_INPUT(node2pin_list);
    CHECK_INPUT(node2pin_list_end);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);
    CHECK_INPUT(net_mask);

    return wa_wirelength_hpwl_cuda(node_pos,
                                   pin_id2node_id,
                                   pin_rel_cpos,
                                   node2pin_list,
                                   node2pin_list_end,
                                   hyperedge_list,
                                   hyperedge_list_end,
                                   net_mask,
                                   gamma,
                                   deterministic);
}

std::vector<torch::Tensor> wa_wirelength_masked_scale_hpwl(torch::Tensor node_pos,
                                                           torch::Tensor pin_id2node_id,
                                                           torch::Tensor pin_rel_cpos,
                                                           torch::Tensor node2pin_list,
                                                           torch::Tensor node2pin_list_end,
                                                           torch::Tensor hyperedge_list,
                                                           torch::Tensor hyperedge_list_end,
                                                           torch::Tensor net_mask,
                                                           torch::Tensor hpwl_scale,
                                                           float gamma,
                                                           bool deterministic) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(pin_id2node_id);
    CHECK_INPUT(pin_rel_cpos);
    CHECK_INPUT(node2pin_list);
    CHECK_INPUT(node2pin_list_end);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);
    CHECK_INPUT(net_mask);
    CHECK_INPUT(hpwl_scale);

    return wa_wirelength_masked_scale_hpwl_cuda(node_pos,
                                                pin_id2node_id,
                                                pin_rel_cpos,
                                                node2pin_list,
                                                node2pin_list_end,
                                                hyperedge_list,
                                                hyperedge_list_end,
                                                net_mask,
                                                hpwl_scale,
                                                gamma,
                                                deterministic);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("masked_scale_hpwl_sum", &masked_scale_hpwl_sum, "calculate the sum of scaled HPWL");
    m.def("merged_forward_backward", &wa_wirelength, "calculate WA wirelength and pin grad");
    m.def("merged_forward_backward_with_hpwl", &wa_wirelength_hpwl, "calculate WA wirelength, pin grad and hpwl");
    m.def("merged_forward_backward_with_masked_scale_hpwl",
          &wa_wirelength_masked_scale_hpwl,
          "calculate WA wirelength, pin grad and the scaled hpwl");
}