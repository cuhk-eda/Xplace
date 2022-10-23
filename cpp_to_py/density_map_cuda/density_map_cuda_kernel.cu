#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <vector>

template <typename scalar_t>
__device__ scalar_t overlap(scalar_t x_l, scalar_t x_h, scalar_t bin_x_l) {
    // bin_x_h == bin_x_l + 1
    return min(x_h, bin_x_l + 1) - max(x_l, bin_x_l);
}

template <typename scalar_t>
__global__ void density_map_cuda_normalize_node_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_size,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> node_weight,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> expand_ratio,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> unit_len,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalize_node_info,
    int num_bin_x,
    int num_bin_y,
    int num_nodes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        normalize_node_info[i][0] = (node_pos[i][0] - node_size[i][0] / 2) / unit_len[0];  // x_l
        normalize_node_info[i][1] = (node_pos[i][0] + node_size[i][0] / 2) / unit_len[0];  // x_h
        normalize_node_info[i][2] = (node_pos[i][1] - node_size[i][1] / 2) / unit_len[1];  // y_l
        normalize_node_info[i][3] = (node_pos[i][1] + node_size[i][1] / 2) / unit_len[1];  // y_h
        normalize_node_info[i][4] = node_weight[i] * expand_ratio[i];                      // weight
        if (normalize_node_info[i][1] - normalize_node_info[i][0] < 0 ||
            normalize_node_info[i][3] - normalize_node_info[i][2] < 0) {
            normalize_node_info[i][4] = -normalize_node_info[i][4];  // we should ignore node whose weight <= 0
        }
    }
}

template <typename scalar_t>
__global__ void __launch_bounds__(256, 4) density_map_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalize_node_info,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> aux_mat,
    int num_nodes,
    int num_bin_x,
    int num_bin_y) {
    const int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const scalar_t weight = normalize_node_info[i][4];
        if (weight > 0) {
            const scalar_t x_l = normalize_node_info[i][0];
            const scalar_t x_h = normalize_node_info[i][1];
            const scalar_t y_l = normalize_node_info[i][2];
            const scalar_t y_h = normalize_node_info[i][3];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);

            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
                    scalar_t overlap_area = overlap_x * overlap_y;
                    gpuAtomicAdd(&aux_mat[j][k], weight * overlap_area);
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void __launch_bounds__(256, 4) density_map_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> normalize_node_info,
    const scalar_t *grad_mat,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sorted_node_map,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_grad,
    float grad_weight,
    int num_bin_x,
    int num_bin_y,
    int num_nodes) {
    const int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_nodes) {
        const int i = (sorted_node_map[index] >= 0) ? sorted_node_map[index] : index;
        const scalar_t weight = normalize_node_info[i][4];
        if (weight > 0) {
            const scalar_t x_l = normalize_node_info[i][0];
            const scalar_t x_h = normalize_node_info[i][1];
            const scalar_t y_l = normalize_node_info[i][2];
            const scalar_t y_h = normalize_node_info[i][3];
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);

            extern __shared__ unsigned char grad_xy[];
            scalar_t *grad_x = (scalar_t *)grad_xy;
            scalar_t *grad_y = grad_x + blockDim.z;
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                grad_x[threadIdx.z] = grad_y[threadIdx.z] = 0;
            }
            __syncthreads();

            scalar_t part_grad_x = 0;
            scalar_t part_grad_y = 0;

            for (int j = x_lf + threadIdx.y; j < x_hf + 1; j += blockDim.y) {
                scalar_t bin_x_l = static_cast<scalar_t>(j);
                scalar_t overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf + threadIdx.x; k < y_hf + 1; k += blockDim.x) {
                    scalar_t bin_y_l = static_cast<scalar_t>(k);
                    scalar_t overlap_y = overlap(y_l, y_h, bin_y_l);
                    scalar_t overlap_area = overlap_x * overlap_y;
                    scalar_t tmp_x = grad_mat[0 * num_bin_x * num_bin_y + j * num_bin_y + k];
                    scalar_t tmp_y = grad_mat[1 * num_bin_x * num_bin_y + j * num_bin_y + k];
                    // part_grad_x += overlap_area * grad_mat[0][j][k];
                    // part_grad_y += overlap_area * grad_mat[1][j][k];
                    part_grad_x += overlap_area * tmp_x;
                    part_grad_y += overlap_area * tmp_y;
                }
            }
            gpuAtomicAdd(&grad_x[threadIdx.z], part_grad_x);
            gpuAtomicAdd(&grad_y[threadIdx.z], part_grad_y);
            __syncthreads();

            if (threadIdx.x == 0 && threadIdx.y == 0) {
                node_grad[i][0] = grad_weight * weight * grad_x[threadIdx.z];
                node_grad[i][1] = grad_weight * weight * grad_y[threadIdx.z];
            }
        }
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

    AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "density_map_cuda_normalize_node", ([&] {
                              density_map_cuda_normalize_node_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                  node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  node_size.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  node_weight.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  expand_ratio.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  unit_len.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  normalize_node_info.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  num_bin_x,
                                  num_bin_y,
                                  num_nodes);
                          }));

    return normalize_node_info;
}

torch::Tensor density_map_cuda_forward(torch::Tensor normalize_node_info,
                                       torch::Tensor sorted_node_map,
                                       torch::Tensor aux_mat,
                                       int num_bin_x,
                                       int num_bin_y,
                                       int num_nodes) {
    cudaSetDevice(normalize_node_info.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int thread_count = 64;
    dim3 blockSize(2, 2, thread_count);
    int block_count = (num_nodes - 1 + thread_count) / thread_count;

    AT_DISPATCH_ALL_TYPES(normalize_node_info.scalar_type(), "density_map_cuda_forward", ([&] {
                              density_map_cuda_forward_kernel<scalar_t><<<block_count, blockSize, 0, stream>>>(
                                  normalize_node_info.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                  aux_mat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  num_nodes,
                                  num_bin_x,
                                  num_bin_y);
                          }));

    return aux_mat;
}

torch::Tensor density_map_cuda_backward(torch::Tensor normalize_node_info,
                                        torch::Tensor grad_mat,
                                        torch::Tensor sorted_node_map,
                                        torch::Tensor node_grad,
                                        float grad_weight,
                                        int num_bin_x,
                                        int num_bin_y,
                                        int num_nodes) {
    cudaSetDevice(normalize_node_info.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int thread_count = 64;
    dim3 blockSize(2, 2, thread_count);
    int block_count = (num_nodes - 1 + thread_count) / thread_count;

    AT_DISPATCH_ALL_TYPES(normalize_node_info.scalar_type(), "density_map_cuda_backward", ([&] {
                              size_t shared_mem_size = sizeof(scalar_t) * thread_count * 2;
                              density_map_cuda_backward_kernel<scalar_t>
                                  <<<block_count, blockSize, shared_mem_size, stream>>>(
                                      normalize_node_info.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      grad_mat.data_ptr<scalar_t>(),
                                      sorted_node_map.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                      node_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                      grad_weight,
                                      num_bin_x,
                                      num_bin_y,
                                      num_nodes);
                          }));

    return node_grad;
}