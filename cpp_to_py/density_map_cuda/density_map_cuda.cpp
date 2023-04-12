#include <torch/extension.h>

torch::Tensor density_map_cuda_normalize_node(torch::Tensor node_pos,
                                              torch::Tensor node_size,
                                              torch::Tensor node_weight,
                                              torch::Tensor expand_ratio,
                                              torch::Tensor unit_len,
                                              torch::Tensor normalize_node_info,
                                              int num_bin_x,
                                              int num_bin_y,
                                              int num_nodes);

torch::Tensor density_map_cuda_forward(torch::Tensor normalize_node_info,
                                       torch::Tensor sorted_node_map,
                                       torch::Tensor aux_mat,
                                       int num_bin_x,
                                       int num_bin_y,
                                       int num_nodes,
                                       bool deterministic);

torch::Tensor density_map_cuda_backward(torch::Tensor normalize_node_info,
                                        torch::Tensor grad_mat,
                                        torch::Tensor sorted_node_map,
                                        torch::Tensor node_grad,
                                        float grad_weight,
                                        int num_bin_x,
                                        int num_bin_y,
                                        int num_nodes,
                                        bool deterministic);

torch::Tensor density_map_cuda_forward_naive(torch::Tensor node_pos,
                                             torch::Tensor node_size,
                                             torch::Tensor node_weight,
                                             torch::Tensor unit_len,
                                             torch::Tensor aux_mat,
                                             int num_bin_x,
                                             int num_bin_y,
                                             int num_nodes,
                                             float min_node_w,
                                             float min_node_h,
                                             float margin,
                                             bool clamp_node,
                                             bool deterministic);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor density_map_normalize_node(torch::Tensor node_pos,
                                         torch::Tensor node_size,
                                         torch::Tensor node_weight,
                                         torch::Tensor expand_ratio,
                                         torch::Tensor unit_len,
                                         torch::Tensor normalize_node_info,
                                         int num_bin_x,
                                         int num_bin_y,
                                         int num_nodes) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(node_size);
    CHECK_INPUT(node_weight);
    CHECK_INPUT(expand_ratio);
    CHECK_INPUT(unit_len);
    CHECK_INPUT(normalize_node_info);

    return density_map_cuda_normalize_node(
        node_pos, node_size, node_weight, expand_ratio, unit_len, normalize_node_info, num_bin_x, num_bin_y, num_nodes);
}
torch::Tensor density_map_forward(torch::Tensor normalize_node_info,
                                  torch::Tensor sorted_node_map,
                                  torch::Tensor aux_mat,
                                  int num_bin_x,
                                  int num_bin_y,
                                  int num_nodes,
                                  bool deterministic) {
    CHECK_INPUT(normalize_node_info);
    CHECK_INPUT(sorted_node_map);
    CHECK_INPUT(aux_mat);

    return density_map_cuda_forward(
        normalize_node_info, sorted_node_map, aux_mat, num_bin_x, num_bin_y, num_nodes, deterministic);
}

torch::Tensor density_map_backward(torch::Tensor normalize_node_info,
                                   torch::Tensor grad_mat,
                                   torch::Tensor sorted_node_map,
                                   torch::Tensor node_grad,
                                   float grad_weight,
                                   int num_bin_x,
                                   int num_bin_y,
                                   int num_nodes,
                                   bool deterministic) {
    CHECK_INPUT(normalize_node_info);
    CHECK_INPUT(grad_mat);
    CHECK_INPUT(sorted_node_map);
    CHECK_INPUT(node_grad);

    return density_map_cuda_backward(normalize_node_info,
                                     grad_mat,
                                     sorted_node_map,
                                     node_grad,
                                     grad_weight,
                                     num_bin_x,
                                     num_bin_y,
                                     num_nodes,
                                     deterministic);
}

torch::Tensor density_map_forward_naive(torch::Tensor node_pos,
                                        torch::Tensor node_size,
                                        torch::Tensor node_weight,
                                        torch::Tensor unit_len,
                                        torch::Tensor aux_mat,
                                        int num_bin_x,
                                        int num_bin_y,
                                        int num_nodes,
                                        float min_node_w,
                                        float min_node_h,
                                        float margin,
                                        bool clamp_node,
                                        bool deterministic) {
    CHECK_INPUT(node_pos);
    CHECK_INPUT(node_size);
    CHECK_INPUT(node_weight);
    CHECK_INPUT(unit_len);
    CHECK_INPUT(aux_mat);

    return density_map_cuda_forward_naive(node_pos,
                                          node_size,
                                          node_weight,
                                          unit_len,
                                          aux_mat,
                                          num_bin_x,
                                          num_bin_y,
                                          num_nodes,
                                          min_node_w,
                                          min_node_h,
                                          margin,
                                          clamp_node,
                                          deterministic);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pre_normalize", &density_map_normalize_node, "normalize bin size to 1");
    m.def("forward", &density_map_forward, "get density map from node information");
    m.def("forward_naive", &density_map_forward_naive, "calculate density map");
    m.def("backward", &density_map_backward, "calculate density gradient of each node");
}