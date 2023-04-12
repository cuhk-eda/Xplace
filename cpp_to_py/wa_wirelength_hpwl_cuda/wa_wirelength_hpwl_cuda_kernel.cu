#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

__global__ void node_pos_to_pin_pos_cuda_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> pin_id2node_id,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pin_pos,
    int num_pins) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = index >> 1;  // pin index
    if (i < num_pins) {
        const int c = index & 1;  // channel index
        int64_t node_id = pin_id2node_id[i];
        pin_pos[i][c] += node_pos[node_id][c];
    }
}

__global__ void calc_node_grad_deterministic_cuda_kernel(
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_grad,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pin_grad,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> node2pin_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> node2pin_list_end,
    int num_nodes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = index >> 1;  // node index
    if (i < num_nodes) {
        const int c = index & 1;  // channel index
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = node2pin_list_end[i - 1];
        }
        int64_t end_idx = node2pin_list_end[i];
        if (end_idx != start_idx) {
            node_grad[i][c] += pin_grad[node2pin_list[start_idx]][c];
            for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
                node_grad[i][c] += pin_grad[node2pin_list[idx]][c];
            }
        }
    }
}

__global__ void masked_scale_hpwl_cuda_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> net_mask,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> hpwl_scale,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> partial_hpwl,
    int num_nets) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = index >> 1;  // net index
    if (i < num_nets && net_mask[i]) {
        const int c = index & 1;  // channel index
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];
        if (end_idx != start_idx) {
            int64_t pin_id = hyperedge_list[start_idx];
            float x_min = pin_pos[pin_id][c];
            float x_max = pin_pos[pin_id][c];
            for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
                float xx = pin_pos[hyperedge_list[idx]][c];
                x_min = min(xx, x_min);
                x_max = max(xx, x_max);
            }
            partial_hpwl[i][c] = round(abs(x_max - x_min) * hpwl_scale[c]);
        }
    }
}

__global__ void wa_wirelength_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> net_mask,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> partial_wa_wl,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pin_grad,
    int num_nets,
    float inv_gamma) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = index >> 1;  // net index
    if (i < num_nets && net_mask[i]) {
        const int c = index & 1;  // channel index
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];
        if (end_idx != start_idx) {
            int64_t pin_id = hyperedge_list[start_idx];
            float x_min = pin_pos[pin_id][c];
            float x_max = pin_pos[pin_id][c];
            for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
                float xx = pin_pos[hyperedge_list[idx]][c];
                x_min = min(xx, x_min);
                x_max = max(xx, x_max);
            }

            float xexp_x_sum = 0;
            float xexp_nx_sum = 0;
            float exp_x_sum = 0;
            float exp_nx_sum = 0;

            for (int64_t idx = start_idx; idx < end_idx; idx++) {
                float xx = pin_pos[hyperedge_list[idx]][c];
                float exp_x = exp((xx - x_max) * inv_gamma);
                float exp_nx = exp((x_min - xx) * inv_gamma);

                xexp_x_sum += xx * exp_x;
                xexp_nx_sum += xx * exp_nx;
                exp_x_sum += exp_x;
                exp_nx_sum += exp_nx;
            }

            float wl = xexp_x_sum / exp_x_sum - xexp_nx_sum / exp_nx_sum;
            partial_wa_wl[i][c] = wl;

            float b_x = inv_gamma / (exp_x_sum);
            float a_x = (1.0 - b_x * xexp_x_sum) / exp_x_sum;
            float b_nx = -inv_gamma / (exp_nx_sum);
            float a_nx = (1.0 - b_nx * xexp_nx_sum) / exp_nx_sum;

            for (int64_t idx = start_idx; idx < end_idx; idx++) {
                float xx = pin_pos[hyperedge_list[idx]][c];
                float exp_x = exp((xx - x_max) * inv_gamma);
                float exp_nx = exp((x_min - xx) * inv_gamma);

                pin_grad[hyperedge_list[idx]][c] = (a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx;
            }
        }
    }
}

__global__ void wa_wirelength_hpwl_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> net_mask,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> partial_wa_wl,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> partial_hpwl,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pin_grad,
    int num_nets,
    float inv_gamma) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = index >> 1;  // net index
    if (i < num_nets && net_mask[i]) {
        const int c = index & 1;  // channel index
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];
        if (end_idx != start_idx) {
            int64_t pin_id = hyperedge_list[start_idx];
            float x_min = pin_pos[pin_id][c];
            float x_max = pin_pos[pin_id][c];
            for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
                float xx = pin_pos[hyperedge_list[idx]][c];
                x_min = min(xx, x_min);
                x_max = max(xx, x_max);
            }
            partial_hpwl[i][c] = abs(x_max - x_min);

            float xexp_x_sum = 0;
            float xexp_nx_sum = 0;
            float exp_x_sum = 0;
            float exp_nx_sum = 0;

            for (int64_t idx = start_idx; idx < end_idx; idx++) {
                float xx = pin_pos[hyperedge_list[idx]][c];
                float exp_x = exp((xx - x_max) * inv_gamma);
                float exp_nx = exp((x_min - xx) * inv_gamma);

                xexp_x_sum += xx * exp_x;
                xexp_nx_sum += xx * exp_nx;
                exp_x_sum += exp_x;
                exp_nx_sum += exp_nx;
            }

            float wl = xexp_x_sum / exp_x_sum - xexp_nx_sum / exp_nx_sum;
            partial_wa_wl[i][c] = wl;

            float b_x = inv_gamma / (exp_x_sum);
            float a_x = (1.0 - b_x * xexp_x_sum) / exp_x_sum;
            float b_nx = -inv_gamma / (exp_nx_sum);
            float a_nx = (1.0 - b_nx * xexp_nx_sum) / exp_nx_sum;

            for (int64_t idx = start_idx; idx < end_idx; idx++) {
                float xx = pin_pos[hyperedge_list[idx]][c];
                float exp_x = exp((xx - x_max) * inv_gamma);
                float exp_nx = exp((x_min - xx) * inv_gamma);

                pin_grad[hyperedge_list[idx]][c] = (a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx;
            }
        }
    }
}

__global__ void wa_wirelength_masked_scale_hpwl_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> net_mask,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> hpwl_scale,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> partial_wa_wl,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> partial_hpwl,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pin_grad,
    int num_nets,
    float inv_gamma) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = index >> 1;  // net index
    if (i < num_nets && net_mask[i]) {
        const int c = index & 1;  // channel index
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];
        if (end_idx != start_idx) {
            int64_t pin_id = hyperedge_list[start_idx];
            float x_min = pin_pos[pin_id][c];
            float x_max = pin_pos[pin_id][c];
            for (int64_t idx = start_idx + 1; idx < end_idx; idx++) {
                float cur_x = pin_pos[hyperedge_list[idx]][c];
                x_min = min(cur_x, x_min);
                x_max = max(cur_x, x_max);
            }
            partial_hpwl[i][c] = round((x_max - x_min) * hpwl_scale[c]);
            // partial_hpwl[i][c] = round(abs(x_max - x_min) * hpwl_scale[c]);

            float sum_x_exp_x = 0;
            float sum_x_exp_nx = 0;
            float sum_exp_x = 0;
            float sum_exp_nx = 0;

            for (int64_t idx = start_idx; idx < end_idx; idx++) {
                float cur_x = pin_pos[hyperedge_list[idx]][c];
                float recenter_exp_x = exp((cur_x - x_max) * inv_gamma);
                float recenter_exp_nx = exp((x_min - cur_x) * inv_gamma);

                sum_x_exp_x += cur_x * recenter_exp_x;
                sum_x_exp_nx += cur_x * recenter_exp_nx;
                sum_exp_x += recenter_exp_x;
                sum_exp_nx += recenter_exp_nx;
            }
            float inv_sum_exp_x = 1 / sum_exp_x;
            float inv_sum_exp_nx = 1 / sum_exp_nx;

            float s_x = sum_x_exp_x * inv_sum_exp_x;
            float ns_nx = sum_x_exp_nx * inv_sum_exp_nx;
            partial_wa_wl[i][c] = s_x - ns_nx;
            float x_coeff = inv_gamma * inv_sum_exp_x;
            float nx_coeff = -inv_gamma * inv_sum_exp_nx;
            float grad_const = (1 - inv_gamma * s_x) * inv_sum_exp_x;
            float grad_nconst = (1 + inv_gamma * ns_nx) * inv_sum_exp_nx;

            for (int64_t idx = start_idx; idx < end_idx; idx++) {
                int64_t pin_id = hyperedge_list[idx];
                float cur_x = pin_pos[pin_id][c];
                float recenter_exp_x = exp((cur_x - x_max) * inv_gamma);
                float recenter_exp_nx = exp((x_min - cur_x) * inv_gamma);

                pin_grad[pin_id][c] = (grad_const + x_coeff * cur_x) * recenter_exp_x -
                                      (grad_nconst + nx_coeff * cur_x) * recenter_exp_nx;
            }
        }
    }
}

void calc_node_grad_cuda(torch::Tensor node_grad,
                         torch::Tensor pin_id2node_id,
                         torch::Tensor pin_grad,
                         torch::Tensor node2pin_list,
                         torch::Tensor node2pin_list_end,
                         int num_nodes,
                         bool deterministic) {
    if (deterministic) {
        auto stream = at::cuda::getCurrentCUDAStream();
        const int threads = 128;
        const int blocks = (num_nodes * 2 + threads - 1) / threads;
        calc_node_grad_deterministic_cuda_kernel<<<blocks, threads, 0, stream>>>(
            node_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            pin_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            node2pin_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            node2pin_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            num_nodes);
    } else {
        const auto pin_id2node_id_view = pin_id2node_id.unsqueeze(1).expand({-1, 2});
        node_grad.scatter_add_(0, pin_id2node_id_view, pin_grad);
    }
}

std::vector<torch::Tensor> wa_wirelength_cuda(torch::Tensor node_pos,
                                              torch::Tensor pin_id2node_id,
                                              torch::Tensor pin_rel_cpos,
                                              torch::Tensor node2pin_list,
                                              torch::Tensor node2pin_list_end,
                                              torch::Tensor hyperedge_list,
                                              torch::Tensor hyperedge_list_end,
                                              torch::Tensor net_mask,
                                              float gamma,
                                              bool deterministic) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto num_nodes = node_pos.size(0);
    const auto num_pins = pin_id2node_id.size(0);
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_channels = 2;  // x, y

    auto pin_pos = pin_rel_cpos.clone();  // pin
    auto partial_wa_wl = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto pin_grad = torch::zeros({num_pins, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));

    const int threads = 128;
    const int blocks = (num_pins * 2 + threads - 1) / threads;

    node_pos_to_pin_pos_cuda_kernel<<<blocks, threads, 0, stream>>>(
        node_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        pin_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        num_pins);

    const int threads2 = 128;
    const int blocks2 = (num_nets * 2 + threads2 - 1) / threads2;

    float inv_gamma = 1 / gamma;
    wa_wirelength_kernel<<<blocks2, threads2, 0, stream>>>(
        pin_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        net_mask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
        partial_wa_wl.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pin_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        num_nets,
        inv_gamma);

    auto node_grad = torch::zeros({num_nodes, num_channels}, torch::dtype(pin_grad.dtype()).device(pin_grad.device()));
    calc_node_grad_cuda(
        node_grad, pin_id2node_id, pin_grad, node2pin_list, node2pin_list_end, num_nodes, deterministic);

    return {partial_wa_wl, node_grad};
}

torch::Tensor masked_scale_hpwl_sum_cuda(torch::Tensor node_pos,
                                         torch::Tensor pin_id2node_id,
                                         torch::Tensor pin_rel_cpos,
                                         torch::Tensor hyperedge_list,
                                         torch::Tensor hyperedge_list_end,
                                         torch::Tensor net_mask,
                                         torch::Tensor hpwl_scale) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto num_nodes = node_pos.size(0);
    const auto num_pins = pin_id2node_id.size(0);
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_channels = 2;  // x, y

    auto pin_pos = pin_rel_cpos.clone();  // pin
    auto partial_hpwl = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));

    const int threads = 128;
    const int blocks = (num_pins * 2 + threads - 1) / threads;

    node_pos_to_pin_pos_cuda_kernel<<<blocks, threads, 0, stream>>>(
        node_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        pin_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        num_pins);

    const int threads2 = 128;
    const int blocks2 = (num_nets * 2 + threads2 - 1) / threads2;

    masked_scale_hpwl_cuda_kernel<<<blocks2, threads2, 0, stream>>>(
        pin_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        net_mask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
        hpwl_scale.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        partial_hpwl.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        num_nets);
    const auto total_hpwl = partial_hpwl.sum();

    return total_hpwl;
}

std::vector<torch::Tensor> wa_wirelength_hpwl_cuda(torch::Tensor node_pos,
                                                   torch::Tensor pin_id2node_id,
                                                   torch::Tensor pin_rel_cpos,
                                                   torch::Tensor node2pin_list,
                                                   torch::Tensor node2pin_list_end,
                                                   torch::Tensor hyperedge_list,
                                                   torch::Tensor hyperedge_list_end,
                                                   torch::Tensor net_mask,
                                                   float gamma,
                                                   bool deterministic) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto num_nodes = node_pos.size(0);
    const auto num_pins = pin_id2node_id.size(0);
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_channels = 2;  // x, y

    auto pin_pos = pin_rel_cpos.clone();  // pin
    auto partial_wa_wl = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto partial_hpwl = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto pin_grad = torch::zeros({num_pins, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));

    const int threads = 128;
    const int blocks = (num_pins * 2 + threads - 1) / threads;

    node_pos_to_pin_pos_cuda_kernel<<<blocks, threads, 0, stream>>>(
        node_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        pin_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        num_pins);

    const int threads2 = 128;
    const int blocks2 = (num_nets * 2 + threads2 - 1) / threads2;

    float inv_gamma = 1 / gamma;
    wa_wirelength_hpwl_kernel<<<blocks2, threads2, 0, stream>>>(
        pin_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        net_mask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
        partial_wa_wl.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        partial_hpwl.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pin_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        num_nets,
        inv_gamma);

    auto node_grad = torch::zeros({num_nodes, num_channels}, torch::dtype(pin_grad.dtype()).device(pin_grad.device()));
    calc_node_grad_cuda(
        node_grad, pin_id2node_id, pin_grad, node2pin_list, node2pin_list_end, num_nodes, deterministic);

    return {partial_wa_wl, node_grad, partial_hpwl};
}

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
                                                                bool deterministic) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto num_nodes = node_pos.size(0);
    const auto num_pins = pin_id2node_id.size(0);
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_channels = 2;  // x, y

    auto pin_pos = pin_rel_cpos.clone();  // pin
    auto partial_wa_wl = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto partial_hpwl = torch::zeros({num_nets, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));
    auto pin_grad = torch::zeros({num_pins, num_channels}, torch::dtype(pin_pos.dtype()).device(pin_pos.device()));

    const int threads = 128;
    const int blocks = (num_pins * 2 + threads - 1) / threads;

    node_pos_to_pin_pos_cuda_kernel<<<blocks, threads, 0, stream>>>(
        node_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        pin_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        num_pins);

    const int threads2 = 128;
    const int blocks2 = (num_nets * 2 + threads2 - 1) / threads2;

    float inv_gamma = 1 / gamma;
    wa_wirelength_masked_scale_hpwl_kernel<<<blocks2, threads2, 0, stream>>>(
        pin_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        net_mask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
        hpwl_scale.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        partial_wa_wl.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        partial_hpwl.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pin_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        num_nets,
        inv_gamma);

    auto node_grad = torch::zeros({num_nodes, num_channels}, torch::dtype(pin_grad.dtype()).device(pin_grad.device()));
    calc_node_grad_cuda(
        node_grad, pin_id2node_id, pin_grad, node2pin_list, node2pin_list_end, num_nodes, deterministic);

    return {partial_wa_wl, node_grad, partial_hpwl};
}