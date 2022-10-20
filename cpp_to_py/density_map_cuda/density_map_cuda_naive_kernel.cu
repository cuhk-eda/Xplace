#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <THC/THCAtomics.cuh>
#include <vector>

template <typename scalar_t>
__global__ void density_map_cuda_forward_naive_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_size,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> node_weight,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> unit_len,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> aux_mat,
    int num_bin_x,
    int num_bin_y,
    int num_nodes,
    float min_node_w,
    float min_node_h,
    float margin,
    bool clamp_node) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        scalar_t node_w = node_size[i][0];
        scalar_t node_h = node_size[i][1];
        scalar_t ratio = 1.0;
        if (clamp_node) {
            const scalar_t node_area = node_w * node_h;
            node_w = max(node_w, static_cast<scalar_t>(min_node_w));
            node_h = max(node_h, static_cast<scalar_t>(min_node_h));
            ratio = node_area / (node_w * node_h);
        }
        const scalar_t mgn = static_cast<scalar_t>(margin);
        const scalar_t num_bin_x_minus_mgn = static_cast<scalar_t>(num_bin_x) - mgn;
        const scalar_t num_bin_y_minus_mgn = static_cast<scalar_t>(num_bin_y) - mgn;
        const scalar_t small_mgn = static_cast<scalar_t>(margin * 0.1);
        scalar_t x_l = max((node_pos[i][0] - node_w / 2) / unit_len[0], mgn);
        scalar_t x_h = min((node_pos[i][0] + node_w / 2) / unit_len[0], num_bin_x_minus_mgn);
        scalar_t y_l = max((node_pos[i][1] - node_h / 2) / unit_len[1], mgn);
        scalar_t y_h = min((node_pos[i][1] + node_h / 2) / unit_len[1], num_bin_y_minus_mgn);
        x_l = min(x_l, num_bin_x_minus_mgn);
        x_h = max(x_h, mgn);
        y_l = min(y_l, num_bin_y_minus_mgn);
        y_h = max(y_h, mgn);
        if (x_h - x_l < small_mgn || y_h - y_l < small_mgn) {
            return;
        }
        const scalar_t p_node_wght = node_weight[i] * ratio;

        const int x_lf = lround(floor(x_l));
        const int x_hf = lround(floor(x_h));
        const int y_lf = lround(floor(y_l));
        const int y_hf = lround(floor(y_h));

        for (int j = x_lf; j < x_hf + 1; j++) {
            const scalar_t bin_x_l = j;
            const scalar_t bin_x_h = j + 1;
            scalar_t overlap_x = min(x_h, bin_x_h) - max(x_l, bin_x_l);
            for (int k = y_lf; k < y_hf + 1; k++) {
                const scalar_t bin_y_l = k;
                const scalar_t bin_y_h = k + 1;
                scalar_t overlap_y = min(y_h, bin_y_h) - max(y_l, bin_y_l);
                scalar_t overlap_area = overlap_x * overlap_y;
                gpuAtomicAdd(&aux_mat[j][k], p_node_wght * overlap_area);
            }
        }
    }
}

template <typename scalar_t>
__global__ void density_map_cuda_backward_naive_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_size,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_mat,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> unit_len,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> node_weight,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> node_grad,
    float grad_weight,
    int num_bin_x,
    int num_bin_y,
    int num_nodes,
    float min_node_w,
    float min_node_h,
    float margin,
    bool clamp_node) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        scalar_t node_w = node_size[i][0];
        scalar_t node_h = node_size[i][1];
        scalar_t ratio = 1.0;
        if (clamp_node) {
            const scalar_t node_area = node_w * node_h;
            node_w = max(node_w, static_cast<scalar_t>(min_node_w));
            node_h = max(node_h, static_cast<scalar_t>(min_node_h));
            ratio = node_area / (node_w * node_h);
        }
        const scalar_t mgn = static_cast<scalar_t>(margin);
        const scalar_t num_bin_x_minus_mgn = static_cast<scalar_t>(num_bin_x) - mgn;
        const scalar_t num_bin_y_minus_mgn = static_cast<scalar_t>(num_bin_y) - mgn;
        const scalar_t small_mgn = static_cast<scalar_t>(margin * 0.1);
        scalar_t x_l = max((node_pos[i][0] - node_w / 2) / unit_len[0], mgn);
        scalar_t x_h = min((node_pos[i][0] + node_w / 2) / unit_len[0], num_bin_x_minus_mgn);
        scalar_t y_l = max((node_pos[i][1] - node_h / 2) / unit_len[1], mgn);
        scalar_t y_h = min((node_pos[i][1] + node_h / 2) / unit_len[1], num_bin_y_minus_mgn);
        x_l = min(x_l, num_bin_x_minus_mgn);
        x_h = max(x_h, mgn);
        y_l = min(y_l, num_bin_y_minus_mgn);
        y_h = max(y_h, mgn);
        if (x_h - x_l < small_mgn || y_h - y_l < small_mgn) {
            return;
        }

        const int x_lf = lround(floor(x_l));
        const int x_hf = lround(floor(x_h));
        const int y_lf = lround(floor(y_l));
        const int y_hf = lround(floor(y_h));

        scalar_t gradX = 0;
        scalar_t gradY = 0;
        for (int j = x_lf; j < x_hf + 1; j++) {
            const scalar_t bin_x_l = j;
            const scalar_t bin_x_h = j + 1;
            scalar_t overlap_x = min(x_h, bin_x_h) - max(x_l, bin_x_l);
            for (int k = y_lf; k < y_hf + 1; k++) {
                const scalar_t bin_y_l = k;
                const scalar_t bin_y_h = k + 1;
                scalar_t overlap_y = min(y_h, bin_y_h) - max(y_l, bin_y_l);
                scalar_t overlap_area = overlap_x * overlap_y;
                gradX += grad_mat[0][j][k] * overlap_area;
                gradY += grad_mat[1][j][k] * overlap_area;
            }
        }
        node_grad[i][0] = grad_weight * ratio * node_weight[i] * gradX;
        node_grad[i][1] = grad_weight * ratio * node_weight[i] * gradY;
    }
}

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
                                             bool clamp_node) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int threads = 64;
    const int blocks = (num_nodes + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "density_map_cuda_forward_naive", ([&] {
                              density_map_cuda_forward_naive_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                  node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  node_size.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  node_weight.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  unit_len.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  aux_mat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  num_bin_x,
                                  num_bin_y,
                                  num_nodes,
                                  min_node_w,
                                  min_node_h,
                                  margin,
                                  clamp_node);
                          }));

    return aux_mat;
}

torch::Tensor density_map_cuda_backward(torch::Tensor node_pos,
                                        torch::Tensor node_size,
                                        torch::Tensor grad_mat,
                                        torch::Tensor node_weight,
                                        torch::Tensor unit_len,
                                        torch::Tensor node_grad,
                                        float grad_weight,
                                        int num_bin_x,
                                        int num_bin_y,
                                        int num_nodes,
                                        float min_node_w,
                                        float min_node_h,
                                        float margin,
                                        bool clamp_node) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int threads = 64;
    const int blocks = (num_nodes + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(node_pos.scalar_type(), "density_map_cuda_backward_naive", ([&] {
                              density_map_cuda_backward_naive_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                  node_pos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  node_size.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  grad_mat.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                  unit_len.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  node_weight.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                  node_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                  grad_weight,
                                  num_bin_x,
                                  num_bin_y,
                                  num_nodes,
                                  min_node_w,
                                  min_node_h,
                                  margin,
                                  clamp_node);
                          }));

    return node_grad;
}
