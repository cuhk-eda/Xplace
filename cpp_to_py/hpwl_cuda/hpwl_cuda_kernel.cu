#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

__global__ void hpwl_cuda_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pin_pos,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> hyperedge_list_end,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> partial_hpwl,
    int num_nets) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = index >> 1;  // pin index
    if (i < num_nets) {
        const int c = index & 1;  // channel index
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];
        partial_hpwl[i][c] = 0;
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
        }
    }
}

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

torch::Tensor hpwl_cuda(torch::Tensor pos, torch::Tensor hyperedge_list, torch::Tensor hyperedge_list_end) {
    cudaSetDevice(pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto num_nets = hyperedge_list_end.size(0);
    const int num_channels = 2;
    auto partial_hpwl = torch::zeros({num_nets, num_channels}, torch::dtype(pos.dtype()).device(pos.device()));

    const int threads = 64;
    const int blocks = (num_nets * 2 + threads - 1) / threads;

    hpwl_cuda_kernel<<<blocks, threads, 0, stream>>>(
        pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        hyperedge_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        hyperedge_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        partial_hpwl.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        num_nets);

    return partial_hpwl;
}

torch::Tensor node_pos_to_pin_pos_cuda(torch::Tensor node_pos,
                                       torch::Tensor pin_id2node_id,
                                       torch::Tensor pin_rel_cpos) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();
    // pin_pos == node_pos + pin_rel_cpos

    const auto num_pins = pin_id2node_id.size(0);

    auto pin_pos = pin_rel_cpos.clone();  // pin

    const int threads = 64;
    const int blocks = (num_pins * 2 + threads - 1) / threads;

    node_pos_to_pin_pos_cuda_kernel<<<blocks, threads, 0, stream>>>(
        node_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        pin_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        num_pins);

    return pin_pos;
}
