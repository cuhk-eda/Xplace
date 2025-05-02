#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "utils.cuh"

namespace gt {

__global__ void explore_path_kernel(index_type* at_prefix_pin,
                                    index_type* at_prefix_arc,
                                    int* at_prefix_attr,
                                    float* pinAT,
                                    int* arc_types,
                                    float* arcDelay,
                                    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> indices,
                                    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> ep_i_indices,
                                    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> endpoints_index,
                                    float* from_pin_delay,
                                    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_visited,
                                    int K) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K) {
        index_type cur_id = endpoints_index[indices[k]];
        int to_i = ep_i_indices[indices[k]];
        while (cur_id != -1) {
            atomicAdd(&pin_visited[cur_id], 1);
            int prev_id = at_prefix_pin[cur_id * NUM_ATTR + to_i];
            int arc_id = at_prefix_arc[cur_id * NUM_ATTR + to_i];
            int from_i = at_prefix_attr[cur_id * NUM_ATTR + to_i];
            int arc_i = (from_i << 1) + (to_i & 0b1);
            float at = 0;
            float delay = 0;
            if (prev_id != -1) {
                at = pinAT[cur_id * NUM_ATTR + to_i];
                delay = arcDelay[arc_id * 2 * NUM_ATTR + arc_i] / pow(1 + k, 2);

                if (arc_types[arc_id] == 0) {
                    atomicAdd(&from_pin_delay[cur_id], delay);
                } else {
                    atomicAdd(&from_pin_delay[prev_id], delay);
                }
            }
            cur_id = prev_id;
            to_i = from_i;
        }
    }
}

__global__ void explore_path_deterministic_kernel(index_type* at_prefix_pin,
                                                  index_type* at_prefix_arc,
                                                  int* at_prefix_attr,
                                                  float* pinAT,
                                                  int* arc_types,
                                                  float* arcDelay,
                                                  torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> indices,
                                                  torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> ep_i_indices,
                                                  torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> endpoints_index,
                                                  unsigned long long* from_pin_delay,
                                                  torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> pin_visited,
                                                  int K,
                                                  unsigned long long scalar) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K) {
        index_type cur_id = endpoints_index[indices[k]];
        int to_i = ep_i_indices[indices[k]];
        while (cur_id != -1) {
            atomicAdd(&pin_visited[cur_id], 1);
            int prev_id = at_prefix_pin[cur_id * NUM_ATTR + to_i];
            int arc_id = at_prefix_arc[cur_id * NUM_ATTR + to_i];
            int from_i = at_prefix_attr[cur_id * NUM_ATTR + to_i];
            int arc_i = (from_i << 1) + (to_i & 0b1);
            float at = 0;
            float delay = 0;
            if (prev_id != -1) {
                at = pinAT[cur_id * NUM_ATTR + to_i];
                delay = arcDelay[arc_id * 2 * NUM_ATTR + arc_i] / pow(1 + k, 2);

                if (arc_types[arc_id] == 0) {
                    atomicAdd(&from_pin_delay[cur_id], static_cast<unsigned long long>(delay * scalar));
                } else {
                    atomicAdd(&from_pin_delay[prev_id], static_cast<unsigned long long>(delay * scalar));
                }
            }
            cur_id = prev_id;
            to_i = from_i;
        }
    }
}

__global__ void copyFromFloatAuxArray(
    unsigned long long* aux_array_uint64, float* aux_array, unsigned long long scalar, float inv_scalar, int num_pins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins) {
        aux_array_uint64[i] = static_cast<unsigned long long>(aux_array[i] * scalar);
    }
}

__global__ void copyToFloatAuxArray(
    unsigned long long* aux_array_uint64, float* aux_array, unsigned long long scalar, float inv_scalar, int num_pins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins) {
        aux_array[i] = static_cast<float>(inv_scalar * aux_array_uint64[i]);
    }
}

std::tuple<torch::Tensor, torch::Tensor> explore_path(index_type* at_prefix_pin,
                                                      index_type* at_prefix_arc,
                                                      int* at_prefix_attr,
                                                      float* pinAT,
                                                      int* arc_types,
                                                      float* arcDelay,
                                                      torch::Tensor indices,
                                                      torch::Tensor ep_i_indices,
                                                      torch::Tensor endpoints_index,
                                                      int num_pins,
                                                      int K,
                                                      bool deterministic) {
    cudaSetDevice(indices.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor from_pin_delay = torch::zeros({num_pins}, torch::dtype(torch::kFloat32).device(indices.device())).contiguous();
    torch::Tensor pin_visited = torch::zeros({num_pins}, torch::dtype(torch::kInt32).device(indices.device())).contiguous();

    int numThreads = 512;
    int numBlocks = (K + numThreads - 1) / numThreads;

    if (deterministic) {
        // use cache to save runtime
        int cp_threads = 512;
        int cp_blocks = (num_pins + cp_threads - 1) / cp_threads;

        // TODO: use a better way to determine the scalar
        int max_value_bits = 32;
        int scalar_bits = max(64 - max_value_bits, 0);
        unsigned long long scalar = (1UL << scalar_bits);
        float inv_scalar = 1.0 / static_cast<float>(scalar);
        static unsigned long long* aux_array_uint64_ptr = nullptr;
        static int aux_array_uint64_size = -1;
        if (aux_array_uint64_ptr == nullptr) {
            aux_array_uint64_size = num_pins;
            cudaMalloc(&aux_array_uint64_ptr, num_pins * sizeof(unsigned long long));
        } else if (num_pins != aux_array_uint64_size) {
            cudaFree(aux_array_uint64_ptr);
            aux_array_uint64_ptr = nullptr;
            aux_array_uint64_size = num_pins;
            cudaMalloc(&aux_array_uint64_ptr, aux_array_uint64_size * sizeof(unsigned long long));
        }
        copyFromFloatAuxArray<<<cp_blocks, cp_threads, 0, stream>>>(
            aux_array_uint64_ptr, from_pin_delay.data_ptr<float>(), scalar, inv_scalar, num_pins);
        explore_path_deterministic_kernel<<<numBlocks, numThreads, 0, stream>>>(
            at_prefix_pin,
            at_prefix_arc,
            at_prefix_attr,
            pinAT,
            arc_types,
            arcDelay,
            indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            ep_i_indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            endpoints_index.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            aux_array_uint64_ptr,
            pin_visited.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            K,
            scalar);
        copyToFloatAuxArray<<<cp_blocks, cp_threads, 0, stream>>>(
            aux_array_uint64_ptr, from_pin_delay.data_ptr<float>(), scalar, inv_scalar, num_pins);

    } else {
        explore_path_kernel<<<numBlocks, numThreads, 0, stream>>>(at_prefix_pin,
                                                                  at_prefix_arc,
                                                                  at_prefix_attr,
                                                                  pinAT,
                                                                  arc_types,
                                                                  arcDelay,
                                                                  indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                                                  ep_i_indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                                                  endpoints_index.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                  from_pin_delay.data_ptr<float>(),
                                                                  pin_visited.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                  K);
    }

    return {from_pin_delay, pin_visited};
}

}  // namespace gt