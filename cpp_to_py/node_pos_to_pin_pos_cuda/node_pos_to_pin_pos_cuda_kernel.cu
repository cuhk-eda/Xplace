#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void node_pos_to_pin_pos_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> pin_id2node_id,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pin_pos,
    int num_pins) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = index >> 1;  // pin index
    if (i < num_pins) {
        const int c = index & 1;  // channel index
        int64_t node_id = pin_id2node_id[i];
        pin_pos[i][c] += node_pos[node_id][c];
        // scalar_t result = pin_pos[i][c] + node_pos[node_id][c];
        // pin_pos[i][c] = result;
        // gpuAtomicAdd(&pin_pos[i][c], node_pos[node_id][c]);
    }
}

torch::Tensor node_pos_to_pin_pos_cuda_forward(torch::Tensor node_pos,
                                               torch::Tensor pin_id2node_id,
                                               torch::Tensor pin_rel_cpos) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();
    // pin_pos == node_pos + pin_rel_cpos

    const auto num_pins = pin_id2node_id.size(0);

    auto pin_pos = pin_rel_cpos.clone();  // pin

    const int threads = 64;
    const int blocks = (num_pins * 2 + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "node_pos_to_pin_pos_cuda_forward", ([&] {
                              node_pos_to_pin_pos_cuda_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                  node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  pin_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  num_pins);
                          }));

    return pin_pos;
}
