#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

__global__ void density_map_cuda_forward_naive_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_size,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> node_weight,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> unit_len,
    float *aux_mat,
    int num_bin_x,
    int num_bin_y,
    int num_nodes,
    float min_node_w,
    float min_node_h,
    float margin,
    bool clamp_node) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        float node_w = node_size[i][0];
        float node_h = node_size[i][1];
        float ratio = 1.0;
        if (clamp_node) {
            const float node_area = node_w * node_h;
            node_w = max(node_w, static_cast<float>(min_node_w));
            node_h = max(node_h, static_cast<float>(min_node_h));
            ratio = node_area / (node_w * node_h);
        }
        const float mgn = static_cast<float>(margin);
        const float num_bin_x_minus_mgn = static_cast<float>(num_bin_x) - mgn;
        const float num_bin_y_minus_mgn = static_cast<float>(num_bin_y) - mgn;
        const float small_mgn = static_cast<float>(margin * 0.1);
        float x_l = max((node_pos[i][0] - node_w / 2) / unit_len[0], mgn);
        float x_h = min((node_pos[i][0] + node_w / 2) / unit_len[0], num_bin_x_minus_mgn);
        float y_l = max((node_pos[i][1] - node_h / 2) / unit_len[1], mgn);
        float y_h = min((node_pos[i][1] + node_h / 2) / unit_len[1], num_bin_y_minus_mgn);
        x_l = min(x_l, num_bin_x_minus_mgn);
        x_h = max(x_h, mgn);
        y_l = min(y_l, num_bin_y_minus_mgn);
        y_h = max(y_h, mgn);
        if (x_h - x_l < small_mgn || y_h - y_l < small_mgn) {
            return;
        }
        const float p_node_wght = node_weight[i] * ratio;

        const int x_lf = lround(floor(x_l));
        const int x_hf = lround(floor(x_h));
        const int y_lf = lround(floor(y_l));
        const int y_hf = lround(floor(y_h));

        for (int j = x_lf; j < x_hf + 1; j++) {
            const float bin_x_l = j;
            const float bin_x_h = j + 1;
            float overlap_x = min(x_h, bin_x_h) - max(x_l, bin_x_l);
            for (int k = y_lf; k < y_hf + 1; k++) {
                const float bin_y_l = k;
                const float bin_y_h = k + 1;
                float overlap_y = min(y_h, bin_y_h) - max(y_l, bin_y_l);
                float overlap_area = overlap_x * overlap_y;
                atomicAdd(&aux_mat[j * num_bin_y + k], p_node_wght * overlap_area);
            }
        }
    }
}

__global__ void density_map_cuda_deterministic_forward_naive_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_size,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> node_weight,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> unit_len,
    unsigned long long *aux_mat,
    int num_bin_x,
    int num_bin_y,
    int num_nodes,
    float min_node_w,
    float min_node_h,
    float margin,
    bool clamp_node,
    unsigned long long scalar) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        float node_w = node_size[i][0];
        float node_h = node_size[i][1];
        float ratio = 1.0;
        if (clamp_node) {
            const float node_area = node_w * node_h;
            node_w = max(node_w, static_cast<float>(min_node_w));
            node_h = max(node_h, static_cast<float>(min_node_h));
            ratio = node_area / (node_w * node_h);
        }
        const float mgn = static_cast<float>(margin);
        const float num_bin_x_minus_mgn = static_cast<float>(num_bin_x) - mgn;
        const float num_bin_y_minus_mgn = static_cast<float>(num_bin_y) - mgn;
        const float small_mgn = static_cast<float>(margin * 0.1);
        float x_l = max((node_pos[i][0] - node_w / 2) / unit_len[0], mgn);
        float x_h = min((node_pos[i][0] + node_w / 2) / unit_len[0], num_bin_x_minus_mgn);
        float y_l = max((node_pos[i][1] - node_h / 2) / unit_len[1], mgn);
        float y_h = min((node_pos[i][1] + node_h / 2) / unit_len[1], num_bin_y_minus_mgn);
        x_l = min(x_l, num_bin_x_minus_mgn);
        x_h = max(x_h, mgn);
        y_l = min(y_l, num_bin_y_minus_mgn);
        y_h = max(y_h, mgn);
        if (x_h - x_l < small_mgn || y_h - y_l < small_mgn) {
            return;
        }
        const float p_node_wght = node_weight[i] * ratio;

        const int x_lf = lround(floor(x_l));
        const int x_hf = lround(floor(x_h));
        const int y_lf = lround(floor(y_l));
        const int y_hf = lround(floor(y_h));

        for (int j = x_lf; j < x_hf + 1; j++) {
            const float bin_x_l = j;
            const float bin_x_h = j + 1;
            float overlap_x = min(x_h, bin_x_h) - max(x_l, bin_x_l);
            for (int k = y_lf; k < y_hf + 1; k++) {
                const float bin_y_l = k;
                const float bin_y_h = k + 1;
                float overlap_y = min(y_h, bin_y_h) - max(y_l, bin_y_l);
                float overlap_area = overlap_x * overlap_y;
                atomicAdd(&aux_mat[j * num_bin_y + k],
                          static_cast<unsigned long long>(p_node_wght * overlap_area * scalar));
            }
        }
    }
}

__global__ void density_map_cuda_backward_naive_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_size,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_mat,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> unit_len,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> node_weight,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_grad,
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
        float node_w = node_size[i][0];
        float node_h = node_size[i][1];
        float ratio = 1.0;
        if (clamp_node) {
            const float node_area = node_w * node_h;
            node_w = max(node_w, static_cast<float>(min_node_w));
            node_h = max(node_h, static_cast<float>(min_node_h));
            ratio = node_area / (node_w * node_h);
        }
        const float mgn = static_cast<float>(margin);
        const float num_bin_x_minus_mgn = static_cast<float>(num_bin_x) - mgn;
        const float num_bin_y_minus_mgn = static_cast<float>(num_bin_y) - mgn;
        const float small_mgn = static_cast<float>(margin * 0.1);
        float x_l = max((node_pos[i][0] - node_w / 2) / unit_len[0], mgn);
        float x_h = min((node_pos[i][0] + node_w / 2) / unit_len[0], num_bin_x_minus_mgn);
        float y_l = max((node_pos[i][1] - node_h / 2) / unit_len[1], mgn);
        float y_h = min((node_pos[i][1] + node_h / 2) / unit_len[1], num_bin_y_minus_mgn);
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

        float gradX = 0;
        float gradY = 0;
        for (int j = x_lf; j < x_hf + 1; j++) {
            const float bin_x_l = j;
            const float bin_x_h = j + 1;
            float overlap_x = min(x_h, bin_x_h) - max(x_l, bin_x_l);
            for (int k = y_lf; k < y_hf + 1; k++) {
                const float bin_y_l = k;
                const float bin_y_h = k + 1;
                float overlap_y = min(y_h, bin_y_h) - max(y_l, bin_y_l);
                float overlap_area = overlap_x * overlap_y;
                gradX += grad_mat[0][j][k] * overlap_area;
                gradY += grad_mat[1][j][k] * overlap_area;
            }
        }
        node_grad[i][0] = grad_weight * ratio * node_weight[i] * gradX;
        node_grad[i][1] = grad_weight * ratio * node_weight[i] * gradY;
    }
}

__global__ void copyFromFloatAuxMat2(
    unsigned long long *aux_mat_uint64, float *aux_mat, unsigned long long scalar, float inv_scalar, int num_bin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bin) {
        aux_mat_uint64[i] = static_cast<unsigned long long>(aux_mat[i] * scalar);
    }
}

__global__ void copyToFloatAuxMat2(
    unsigned long long *aux_mat_uint64, float *aux_mat, unsigned long long scalar, float inv_scalar, int num_bin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bin) {
        aux_mat[i] = static_cast<float>(inv_scalar * aux_mat_uint64[i]);
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
                                             bool clamp_node,
                                             bool deterministic) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int threads = 64;
    const int blocks = (num_nodes + threads - 1) / threads;

    if (deterministic) {
        // each bin size is internally normalized to 1x1
        // max_value_bits -> #bits of the maximum density == (num_bin_x * num_bin_y)
        int max_value_bits = max(static_cast<int>(ceil(log2((num_bin_x + 0.1) * (num_bin_y + 0.1)))) + 1, 32);
        int scalar_bits = max(64 - max_value_bits, 0);
        unsigned long long scalar = (1UL << scalar_bits);
        float inv_scalar = 1.0 / static_cast<float>(scalar);
        int num_bin = num_bin_x * num_bin_y;

        int cp_threads = 512;
        int cp_blocks = (num_bin + cp_threads - 1) / cp_threads;
        unsigned long long *aux_mat_uint64 = nullptr;
        cudaMallocAsync(&aux_mat_uint64, num_bin * sizeof(unsigned long long), stream);
        copyFromFloatAuxMat2<<<cp_blocks, cp_threads, 0, stream>>>(
            aux_mat_uint64, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        density_map_cuda_deterministic_forward_naive_kernel<<<blocks, threads, 0, stream>>>(
            node_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            node_size.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            node_weight.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            unit_len.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            aux_mat_uint64,
            num_bin_x,
            num_bin_y,
            num_nodes,
            min_node_w,
            min_node_h,
            margin,
            clamp_node,
            scalar);
        copyToFloatAuxMat2<<<cp_blocks, cp_threads, 0, stream>>>(
            aux_mat_uint64, aux_mat.data_ptr<float>(), scalar, inv_scalar, num_bin);
        cudaFreeAsync(aux_mat_uint64, stream);
    } else {
        density_map_cuda_forward_naive_kernel<<<blocks, threads, 0, stream>>>(
            node_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            node_size.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            node_weight.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            unit_len.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            aux_mat.data_ptr<float>(),
            num_bin_x,
            num_bin_y,
            num_nodes,
            min_node_w,
            min_node_h,
            margin,
            clamp_node);
    }

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
                                        bool clamp_node,
                                        bool deterministic) {
    cudaSetDevice(node_pos.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int threads = 64;
    const int blocks = (num_nodes + threads - 1) / threads;

    density_map_cuda_backward_naive_kernel<<<blocks, threads, 0, stream>>>(
        node_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        node_size.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        grad_mat.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        unit_len.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        node_weight.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        node_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        grad_weight,
        num_bin_x,
        num_bin_y,
        num_nodes,
        min_node_w,
        min_node_h,
        margin,
        clamp_node);

    return node_grad;
}
