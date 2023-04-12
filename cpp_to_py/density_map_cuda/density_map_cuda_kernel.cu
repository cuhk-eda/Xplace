#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

template <typename T>
__device__ T overlap(T x_l, T x_h, T bin_x_l) {
    // bin_x_h == bin_x_l + 1
    return min(x_h, bin_x_l + 1) - max(x_l, bin_x_l);
}

__global__ void density_map_cuda_normalize_node_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_size,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> node_weight,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> expand_ratio,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> unit_len,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> normalize_node_info,
    int num_bin_x,
    int num_bin_y,
    int num_nodes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        // normalize bin_size_x and bin_size_y to 1
        normalize_node_info[i][0] = (node_pos[i][0] - node_size[i][0] / 2) / unit_len[0];  // x_l
        normalize_node_info[i][1] = (node_pos[i][0] + node_size[i][0] / 2) / unit_len[0];  // x_h
        normalize_node_info[i][2] = (node_pos[i][1] - node_size[i][1] / 2) / unit_len[1];  // y_l
        normalize_node_info[i][3] = (node_pos[i][1] + node_size[i][1] / 2) / unit_len[1];  // y_h
        normalize_node_info[i][4] = node_weight[i] * expand_ratio[i];                      // weight
        if (normalize_node_info[i][1] - normalize_node_info[i][0] < 0 ||
            normalize_node_info[i][3] - normalize_node_info[i][2] < 0 ||
            (node_size[i][0] < 1e-6 && node_size[i][1] < 1e-6)) {
            normalize_node_info[i][4] = -normalize_node_info[i][4];  // we should ignore node whose weight <= 0
        }
    }
}

__global__ void __launch_bounds__(256, 4) density_map_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> normalize_node_info,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    float *aux_mat,
    int num_nodes,
    int num_bin_x,
    int num_bin_y) {
    const int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const float weight = normalize_node_info[i][4];
        if (weight > 0) {
            const float x_l = normalize_node_info[i][0];
            const float x_h = normalize_node_info[i][1];
            const float y_l = normalize_node_info[i][2];
            const float y_h = normalize_node_info[i][3];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);

            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                float bin_x_l = static_cast<float>(j);
                float overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    float bin_y_l = static_cast<float>(k);
                    float overlap_y = overlap(y_l, y_h, bin_y_l);
                    float overlap_area = overlap_x * overlap_y;
                    atomicAdd(&aux_mat[j * num_bin_y + k], weight * overlap_area);
                }
            }
        }
    }
}

__global__ void __launch_bounds__(256, 4) density_map_cuda_deterministic_forward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> normalize_node_info,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    unsigned long long *aux_mat,
    int num_nodes,
    int num_bin_x,
    int num_bin_y,
    unsigned long long scalar) {
    const int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const float weight = normalize_node_info[i][4];
        if (weight > 0) {
            const float x_l = normalize_node_info[i][0];
            const float x_h = normalize_node_info[i][1];
            const float y_l = normalize_node_info[i][2];
            const float y_h = normalize_node_info[i][3];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);

            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                float bin_x_l = static_cast<float>(j);
                float overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    float bin_y_l = static_cast<float>(k);
                    float overlap_y = overlap(y_l, y_h, bin_y_l);
                    float overlap_area = overlap_x * overlap_y;
                    atomicAdd(&aux_mat[j * num_bin_y + k],
                              static_cast<unsigned long long>(weight * overlap_area * scalar));
                }
            }
        }
    }
}

__global__ void __launch_bounds__(256, 4) density_map_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> normalize_node_info,
    const float *grad_mat,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_grad,
    float grad_weight,
    int num_bin_x,
    int num_bin_y,
    int num_nodes) {
    const int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const float weight = normalize_node_info[i][4];
        if (weight > 0) {
            const float x_l = normalize_node_info[i][0];
            const float x_h = normalize_node_info[i][1];
            const float y_l = normalize_node_info[i][2];
            const float y_h = normalize_node_info[i][3];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);

            extern __shared__ unsigned char grad_xy[];
            float *grad_x = (float *)grad_xy;
            float *grad_y = grad_x + blockDim.z;
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                grad_x[threadIdx.z] = grad_y[threadIdx.z] = 0;
            }
            __syncthreads();

            float part_grad_x = 0;
            float part_grad_y = 0;

            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                float bin_x_l = static_cast<float>(j);
                float overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    float bin_y_l = static_cast<float>(k);
                    float overlap_y = overlap(y_l, y_h, bin_y_l);
                    float overlap_area = overlap_x * overlap_y;
                    float tmp_x = grad_mat[0 * num_bin_x * num_bin_y + j * num_bin_y + k];
                    float tmp_y = grad_mat[1 * num_bin_x * num_bin_y + j * num_bin_y + k];
                    part_grad_x += overlap_area * tmp_x;
                    part_grad_y += overlap_area * tmp_y;
                }
            }
            atomicAdd(&grad_x[threadIdx.z], part_grad_x);
            atomicAdd(&grad_y[threadIdx.z], part_grad_y);
            __syncthreads();

            if (threadIdx.x == 0 && threadIdx.y == 0) {
                node_grad[i][0] = grad_weight * weight * grad_x[threadIdx.z];
                node_grad[i][1] = grad_weight * weight * grad_y[threadIdx.z];
            }
        }
    }
}

__global__ void density_map_cuda_deterministic_backward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> normalize_node_info,
    const float *grad_mat,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_grad,
    float grad_weight,
    int num_bin_x,
    int num_bin_y,
    int num_nodes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const float weight = normalize_node_info[i][4];
        if (weight > 0) {
            const float x_l = normalize_node_info[i][0];
            const float x_h = normalize_node_info[i][1];
            const float y_l = normalize_node_info[i][2];
            const float y_h = normalize_node_info[i][3];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);

            float gradX = 0;
            float gradY = 0;
            for (int j = x_lf; j < x_hf + 1; j++) {
                float bin_x_l = static_cast<float>(j);
                float overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf; k < y_hf + 1; k++) {
                    float bin_y_l = static_cast<float>(k);
                    float overlap_y = overlap(y_l, y_h, bin_y_l);
                    float overlap_area = overlap_x * overlap_y;
                    gradX += grad_mat[0 * num_bin_x * num_bin_y + j * num_bin_y + k] * overlap_area;
                    gradY += grad_mat[1 * num_bin_x * num_bin_y + j * num_bin_y + k] * overlap_area;
                }
            }
            node_grad[i][0] = grad_weight * weight * gradX;
            node_grad[i][1] = grad_weight * weight * gradY;
        }
    }
}

__global__ void copyFromFloatAuxMat(
    unsigned long long *aux_mat_uint64, float *aux_mat, unsigned long long scalar, float inv_scalar, int num_bin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bin) {
        aux_mat_uint64[i] = static_cast<unsigned long long>(aux_mat[i] * scalar);
    }
}

__global__ void copyToFloatAuxMat(
    unsigned long long *aux_mat_uint64, float *aux_mat, unsigned long long scalar, float inv_scalar, int num_bin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bin) {
        aux_mat[i] = static_cast<float>(inv_scalar * aux_mat_uint64[i]);
    }
}

torch::Tensor density_map_cuda_normalize_node(torch::Tensor node_pos,
                                              torch::Tensor node_size,
                                              torch::Tensor node_weight,
                                              torch::Tensor expand_ratio,
                                              torch::Tensor unit_len,
                                              torch::Tensor normalize_node_info,
                                              int num_bin_x,
                                              int num_bin_y,
                                              int num_nodes) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int threads = 128;
    const int blocks = (num_nodes + threads - 1) / threads;

    density_map_cuda_normalize_node_kernel<<<blocks, threads, 0, stream>>>(
        node_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        node_size.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        node_weight.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        expand_ratio.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        unit_len.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        normalize_node_info.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        num_bin_x,
        num_bin_y,
        num_nodes);

    return normalize_node_info;
}

torch::Tensor density_map_cuda_forward(torch::Tensor normalize_node_info,
                                       torch::Tensor sorted_node_map,
                                       torch::Tensor aux_mat,
                                       int num_bin_x,
                                       int num_bin_y,
                                       int num_nodes,
                                       bool deterministic) {
    cudaSetDevice(normalize_node_info.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int thread_count = 64;
    dim3 blockSize(2, 2, thread_count);
    int block_count = (num_nodes - 1 + thread_count) / thread_count;

    if (deterministic) {
        // each bin size is pre-normalized to 1x1
        // max_value_bits -> #bits of the maximum density == (num_bin_x * num_bin_y)
        int max_value_bits = max(static_cast<int>(ceil(log2((num_bin_x + 0.1) * (num_bin_y + 0.1)))) + 1, 32);
        int scalar_bits = max(64 - max_value_bits, 0);
        unsigned long long scalar = (1UL << scalar_bits);
        float inv_scalar = 1.0 / static_cast<float>(scalar);
        int num_bin = num_bin_x * num_bin_y;

        // use cache to save runtime
        int cp_threads = 512;
        int cp_blocks = (num_bin + cp_threads - 1) / cp_threads;
        static unsigned long long *aux_mat_uint64_ptr = nullptr;
        static int aux_mat_uint64_size = -1;
        if (aux_mat_uint64_ptr == nullptr) {
            aux_mat_uint64_size = num_bin;
            cudaMalloc(&aux_mat_uint64_ptr, aux_mat_uint64_size * sizeof(unsigned long long));
        } else if (num_bin != aux_mat_uint64_size) {
            cudaFree(aux_mat_uint64_ptr);
            aux_mat_uint64_ptr = nullptr;
            aux_mat_uint64_size = num_bin;
            cudaMalloc(&aux_mat_uint64_ptr, aux_mat_uint64_size * sizeof(unsigned long long));
        }
        copyFromFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
            aux_mat_uint64_ptr, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        density_map_cuda_deterministic_forward_kernel<<<block_count, blockSize, 0, stream>>>(
            normalize_node_info.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            aux_mat_uint64_ptr,
            num_nodes,
            num_bin_x,
            num_bin_y,
            scalar);
        copyToFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
            aux_mat_uint64_ptr, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);

        // without cache, need a lot of cudaMallocAsync...
        // int cp_threads = 512;
        // int cp_blocks = (num_bin + cp_threads - 1) / cp_threads;
        // unsigned long long *aux_mat_uint64 = nullptr;
        // cudaMallocAsync(&aux_mat_uint64, num_bin * sizeof(unsigned long long), stream);
        // copyFromFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
        //     aux_mat_uint64, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        // density_map_cuda_deterministic_forward_kernel<<<block_count, blockSize, 0, stream>>>(
        //     normalize_node_info.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        //     sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        //     aux_mat_uint64,
        //     num_nodes,
        //     num_bin_x,
        //     num_bin_y,
        //     scalar);
        // copyToFloatAuxMat<<<cp_blocks, cp_threads, 0, stream>>>(
        //     aux_mat_uint64, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        // cudaFreeAsync(aux_mat_uint64, stream);
    } else {
        density_map_cuda_forward_kernel<<<block_count, blockSize, 0, stream>>>(
            normalize_node_info.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            aux_mat.data_ptr<float>(),
            num_nodes,
            num_bin_x,
            num_bin_y);
    }

    return aux_mat;
}

torch::Tensor density_map_cuda_backward(torch::Tensor normalize_node_info,
                                        torch::Tensor grad_mat,
                                        torch::Tensor sorted_node_map,
                                        torch::Tensor node_grad,
                                        float grad_weight,
                                        int num_bin_x,
                                        int num_bin_y,
                                        int num_nodes,
                                        bool deterministic) {
    cudaSetDevice(normalize_node_info.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    if (deterministic) {
        int threads = 64;
        int blocks = (num_nodes + threads - 1) / threads;
        density_map_cuda_deterministic_backward_kernel<<<blocks, threads, 0, stream>>>(
            normalize_node_info.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            grad_mat.data_ptr<float>(),
            sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            node_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            grad_weight,
            num_bin_x,
            num_bin_y,
            num_nodes);
    } else {
        int thread_count = 64;
        dim3 blockSize(2, 2, thread_count);
        int block_count = (num_nodes - 1 + thread_count) / thread_count;
        size_t shared_mem_size = sizeof(float) * thread_count * 2;
        density_map_cuda_backward_kernel<<<block_count, blockSize, shared_mem_size, stream>>>(
            normalize_node_info.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            grad_mat.data_ptr<float>(),
            sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            node_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            grad_weight,
            num_bin_x,
            num_bin_y,
            num_nodes);
    }

    return node_grad;
}