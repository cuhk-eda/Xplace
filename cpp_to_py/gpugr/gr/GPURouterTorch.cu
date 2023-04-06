#include "GPURouter.h"
#include "InCellUsage.cuh"

namespace gr {

#define BLOCK_SIZE 512
#define BLOCK_NUMBER(n) (((n) + (BLOCK_SIZE) - 1) / BLOCK_SIZE)

__device__ void inline cudaSwapInt(int &a, int &b) {
    int c(a);
    a = b;
    b = c;
}

__device__ float overlap(float x_l, float x_h, float bin_x_l) {
    // bin_x_h == bin_x_l + 1
    return min(x_h, bin_x_l + 1) - max(x_l, bin_x_l);
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

__global__ void getDmdTensor(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> dmdMap,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> wireDmdMap,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> viaDmdMap,
    const float *capacity, int *wires, int *vias, float *fixed, int N, int LAYER, int xSize, int ySize, int DIRECTION) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < LAYER * N * N && idx % N + 1 < N) {
        int layer = idx / N / N, x = idx / N % N, y = idx % N;
        if (!(layer & 1) ^ DIRECTION) cudaSwapInt(x, y);
        if (layer < LAYER && x < xSize && y < ySize) {
            float wireDmd = wires[idx] + fixed[idx];
            float viaDmd = twoCellsViaUsage(idx, vias, N, LAYER);
            dmdMap[layer][x][y] = wireDmd + viaDmd;
            wireDmdMap[layer][x][y] = wireDmd;
            viaDmdMap[layer][x][y] = viaDmd;
        }
    }
}

__global__ void getCapTensor(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> capMap,
    const float *capacity, int *wires, int *vias, float *fixed, int N, int LAYER, int xSize, int ySize, int DIRECTION) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < LAYER * N * N && idx % N + 1 < N) {
        int layer = idx / N / N, x = idx / N % N, y = idx % N;
        if (!(layer & 1) ^ DIRECTION) cudaSwapInt(x, y);
        if (layer < LAYER && x < xSize && y < ySize) {
            capMap[layer][x][y] = capacity[idx];
        }
    }
}

__global__ void compGcellRouteForce(
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> gbpin_grad,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> mask_map,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> wire_dmd_map_2d,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> via_dmd_map_2d,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> cap_map_2d,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> dist_weights,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> wirelength_weights,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> route_gradmat,
    float grad_weight, float unit_wire_cost, float unit_via_cost,
    int *gbpinRoutes, int *gbpin2netId, int *routes, int *routesOffset,
    int numGbPin, int N, int LAYER, int xSize, int ySize, int DIRECTION
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numGbPin) {
        int netId = gbpin2netId[idx];
        routes += routesOffset[netId];
        if (routes[0] == -1) {
            // FIXME: failed PR, use floating cost instead of int cost
            return;
        }
        gbpinRoutes += idx * 6;
        int numGbpinRoutes = gbpinRoutes[0];
        // int numGbpinVias = gbpinRoutes[5]; // TODO: consider #Vias in route grad
        for (int i = 1; i < 1 + numGbpinRoutes; i++) {
            int routeId = gbpinRoutes[i];
            bool reverseRoute = false;
            if (routeId < 0) {
                // for a segment (lx, hx), if reverseRoute is true, the gbpin is located at hx, else at lx
                routeId = -routeId;
                reverseRoute = true;
            }
            int p = routes[routeId];
            int l = p / N / N, x = p % (N * N) / N, y = p % N;
            if (!(l & 1) ^ DIRECTION) cudaSwapInt(x, y);
            int lx = x, hx = x, ly = y, hy = y;
            if ((l & 1) ^ DIRECTION) {
                hy += routes[routeId + 1];
            } else {
                hx += routes[routeId + 1];
            }
            if (lx != hx) {
                // x direction route segement
                float grad = 0, total_weight = 0;
                for (int j = lx; j <= hx; j++) {
                    float cur_dist_weight;
                    if (reverseRoute) {
                        cur_dist_weight = dist_weights[hx - j];
                    } else {
                        cur_dist_weight = dist_weights[j - lx];
                    }
                    // TODO: 1) should we also consider X direction? 
                    //       2) consider via cost?
                    float cost = unit_wire_cost / min(cap_map_2d[j][ly], 0.2);
                    grad += cost * route_gradmat[1][j][ly] * mask_map[j][ly] * cur_dist_weight;
                    total_weight += cur_dist_weight;
                }
                gbpin_grad[idx][1] = grad_weight * grad / total_weight * wirelength_weights[hx - lx + 1];
            } else if (ly != hy) {
                // y direction route segement
                float grad = 0, total_weight = 0;
                for (int j = ly; j<= hy; j++) {
                    float cur_dist_weight;
                    if (reverseRoute) {
                        cur_dist_weight = dist_weights[hy - j];
                    } else {
                        cur_dist_weight = dist_weights[j - ly];
                    }
                    float cost = unit_wire_cost / min(cap_map_2d[lx][j], 0.2);
                    grad += cost * route_gradmat[0][lx][j] * mask_map[lx][j] * cur_dist_weight;
                    total_weight += cur_dist_weight;
                }
                gbpin_grad[idx][0] = grad_weight * grad / total_weight * wirelength_weights[hy - ly + 1];
            }
        }
    }
}

__global__ void assignRouteForceToPlPin(
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> plpin_grad,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> gbpin_grad,
    int *plPinId2gbPinId, int numPlPin
) {
    int plPinId = blockIdx.x * blockDim.x + threadIdx.x;
    if (plPinId < numPlPin) {
        int gbPinId = plPinId2gbPinId[plPinId];
        plpin_grad[plPinId][0] = gbpin_grad[gbPinId][0];
        plpin_grad[plPinId][1] = gbpin_grad[gbPinId][1];
    }
}

__global__ void fillerRouteForce(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> filler_pos,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> filler_size,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> filler_weight,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> expand_ratio,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> filler_grad,
    const float *grad_mat,
    float grad_weight,
    float unit_len_x,
    float unit_len_y,
    int num_bin_x,
    int num_bin_y,
    int num_fillers
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_fillers) {
        float weight = filler_weight[i];
        if (weight > 0) {
            const float x_l = (filler_pos[i][0] - filler_size[i][0] / 2) / unit_len_x;
            const float x_h = (filler_pos[i][0] + filler_size[i][0] / 2) / unit_len_x;
            const float y_l = (filler_pos[i][1] - filler_size[i][1] / 2) / unit_len_y;
            const float y_h = (filler_pos[i][1] + filler_size[i][1] / 2) / unit_len_y;
            if (x_h - x_l < 0 || y_h - y_l < 0) return;
            weight *= expand_ratio[i];
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
            filler_grad[i][0] = grad_weight * weight * gradX;
            filler_grad[i][1] = grad_weight * weight * gradY;
        }
    }
}

__global__ void inflateNodeRatioWeighted(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_size,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> node_weight,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> expand_ratio,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> inflate_mat,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_inflate_ratio,
    float grad_weight,
    float unit_len_x,
    float unit_len_y,
    int num_bin_x,
    int num_bin_y,
    int num_nodes
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        float weight = node_weight[i] * expand_ratio[i];
        if (weight > 0) {
            const float x_l = (node_pos[i][0] - node_size[i][0] / 2) / unit_len_x;
            const float x_h = (node_pos[i][0] + node_size[i][0] / 2) / unit_len_x;
            const float y_l = (node_pos[i][1] - node_size[i][1] / 2) / unit_len_y;
            const float y_h = (node_pos[i][1] + node_size[i][1] / 2) / unit_len_y;
            if (x_h - x_l < 0 || y_h - y_l < 0) return;
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);
            const float node_area = (x_h - x_l) * (y_h - y_l);

            float inflate_x = 0, inflate_y = 0;
            for (int j = x_lf; j < x_hf + 1; j++) {
                float bin_x_l = static_cast<float>(j);
                float overlap_x = overlap(x_l, x_h, bin_x_l);
                for (int k = y_lf; k < y_hf + 1; k++) {
                    float bin_y_l = static_cast<float>(k);
                    float overlap_y = overlap(y_l, y_h, bin_y_l);
                    float overlap_area_ratio = overlap_x * overlap_y / node_area;
                    inflate_x += overlap_area_ratio * inflate_mat[0][j][k];
                    inflate_y += overlap_area_ratio * inflate_mat[1][j][k];
                }
            }
            node_inflate_ratio[i][0] = grad_weight * weight * inflate_x;
            node_inflate_ratio[i][1] = grad_weight * weight * inflate_y;
        }
    }
}

__global__ void inflateNodeRatioMax(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_size,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> node_weight,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> expand_ratio,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> inflate_mat,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_inflate_ratio,
    float grad_weight,
    float unit_len_x,
    float unit_len_y,
    int num_bin_x,
    int num_bin_y,
    int num_nodes
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        float weight = node_weight[i] * expand_ratio[i];
        if (weight > 0) {
            const float x_l = (node_pos[i][0] - node_size[i][0] / 2) / unit_len_x;
            const float x_h = (node_pos[i][0] + node_size[i][0] / 2) / unit_len_x;
            const float y_l = (node_pos[i][1] - node_size[i][1] / 2) / unit_len_y;
            const float y_h = (node_pos[i][1] + node_size[i][1] / 2) / unit_len_y;
            if (x_h - x_l < 0 || y_h - y_l < 0) return;
            int x_lf = lround(floor(x_l));
            int x_hf = lround(floor(x_h));
            int y_lf = lround(floor(y_l));
            int y_hf = lround(floor(y_h));
            x_lf = max(x_lf, 0);
            x_hf = min(x_hf, num_bin_x - 1);
            y_lf = max(y_lf, 0);
            y_hf = min(y_hf, num_bin_y - 1);

            float inflate_x = 0, inflate_y = 0;
            for (int j = x_lf; j < x_hf + 1; j++) {
                for (int k = y_lf; k < y_hf + 1; k++) {
                    inflate_x = max(inflate_x, inflate_mat[0][j][k]);
                    inflate_y = max(inflate_y, inflate_mat[1][j][k]);
                }
            }
            node_inflate_ratio[i][0] = grad_weight * weight * inflate_x;
            node_inflate_ratio[i][1] = grad_weight * weight * inflate_y;
        }
    }
}

__global__ void inflatePinRelCpos(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_inflate_ratio,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> old_pin_rel_cpos,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> pin_id2node_id,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> new_pin_rel_cpos,
    int num_movable_nodes,
    int num_pins
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins) {
        int64_t node_id = pin_id2node_id[i];
        if (node_id < num_movable_nodes) {
            new_pin_rel_cpos[i][0] = old_pin_rel_cpos[i][0] * node_inflate_ratio[node_id][0];
            new_pin_rel_cpos[i][1] = old_pin_rel_cpos[i][1] * node_inflate_ratio[node_id][1];
        }
    }
}

__global__ void pseudoPinForce(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_pos,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pseudo_pin_pos,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_grad,
    int num_nodes,
    float inv_gamma) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_id = index >> 1;  // net index
    if (pin_id < num_nodes) {
        const int c = index & 1;  // channel index

        float x1 = node_pos[pin_id][c];
        float x2 = pseudo_pin_pos[pin_id][c];
        float x_max = max(x1, x2);
        float x_min = min(x1, x2);

        float sum_x_exp_x = 0;
        float sum_x_exp_nx = 0;
        float sum_exp_x = 0;
        float sum_exp_nx = 0;

        for (int i = 0; i < 2; i++) {
            float cur_x = i == 0 ? x1 : x2;
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
        float x_coeff = inv_gamma * inv_sum_exp_x;
        float nx_coeff = -inv_gamma * inv_sum_exp_nx;
        float grad_const = (1 - inv_gamma * s_x) * inv_sum_exp_x;
        float grad_nconst = (1 + inv_gamma * ns_nx) * inv_sum_exp_nx;

        // calc x1 (original pin)'s gradient
        float recenter_exp_x = exp((x1 - x_max) * inv_gamma);
        float recenter_exp_nx = exp((x_min - x1) * inv_gamma);
        node_grad[pin_id][c] = (grad_const + x_coeff * x1) * recenter_exp_x -
                                (grad_nconst + nx_coeff * x1) * recenter_exp_nx;
        
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GPURouter::getDemandMap() {
    int xSize = X;
    int ySize = Y;
    torch::Tensor dmdMap = torch::zeros({LAYER, xSize, ySize}, 
                        torch::dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, DEVICE_ID)));
    torch::Tensor wireDmdMap = torch::zeros({LAYER, xSize, ySize}, 
                        torch::dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, DEVICE_ID)));
    torch::Tensor viaDmdMap = torch::zeros({LAYER, xSize, ySize}, 
                        torch::dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, DEVICE_ID)));
    getDmdTensor<<<BLOCK_NUMBER(LAYER * N * N), BLOCK_SIZE>>>(
        dmdMap.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        wireDmdMap.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        viaDmdMap.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        capacity, wires, vias, fixed, N, LAYER, xSize, ySize, DIRECTION);
    return {dmdMap, wireDmdMap, viaDmdMap};
}

torch::Tensor GPURouter::getCapacityMap() {
    int xSize = X;
    int ySize = Y;
    torch::Tensor capMap = torch::zeros({LAYER, xSize, ySize}, 
                        torch::dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, DEVICE_ID)));
    getCapTensor<<<BLOCK_NUMBER(LAYER * N * N), BLOCK_SIZE>>>(
        capMap.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        capacity, wires, vias, fixed, N, LAYER, xSize, ySize, DIRECTION);
    return capMap;
}

torch::Tensor GPURouter::calcRouteGrad(torch::Tensor mask_map,
                                       torch::Tensor wire_dmd_map_2d,
                                       torch::Tensor via_dmd_map_2d,
                                       torch::Tensor cap_map_2d,
                                       torch::Tensor dist_weights,
                                       torch::Tensor wirelength_weights,
                                       torch::Tensor route_gradmat,
                                       torch::Tensor node2pin_list,
                                       torch::Tensor node2pin_list_end,
                                       float grad_weight,
                                       float unit_wire_cost,
                                       float unit_via_cost,
                                       int num_nodes) {
    // 1. compute demand map and capacity map
    // 2. compute route_gradmat for each gcell (use torchDCT)

    // 3. compute route force for each global pin
    torch::Tensor gbpin_grad = torch::zeros({numGbPin, 2}, 
                        torch::dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, DEVICE_ID)));
    compGcellRouteForce<<<BLOCK_NUMBER(numGbPin), BLOCK_SIZE>>>(
        gbpin_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        mask_map.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        wire_dmd_map_2d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        via_dmd_map_2d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        cap_map_2d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        dist_weights.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        wirelength_weights.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        route_gradmat.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        grad_weight, unit_wire_cost, unit_via_cost,
        gbpinRoutes, gbpin2netId, routes, routesOffset,
        numGbPin, N, LAYER, X, Y, DIRECTION
    );

    // 4. assign route force to placement pins (node's pins)
    torch::Tensor plpin_grad = torch::zeros({numPlPin, 2}, 
                        torch::dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, DEVICE_ID)));
    assignRouteForceToPlPin<<<BLOCK_NUMBER(numPlPin), BLOCK_SIZE>>>(
        plpin_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        gbpin_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        plPinId2gbPinId, numPlPin
    );

    // 5. calc node grad (use deterministic summation)
    auto node_grad = torch::zeros({num_nodes, 2}, torch::dtype(plpin_grad.dtype()).device(plpin_grad.device()));
    const int threads = 128;
    const int blocks = (num_nodes * 2 + threads - 1) / threads;
    calc_node_grad_deterministic_cuda_kernel<<<blocks, threads, 0>>>(
        node_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        plpin_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        node2pin_list.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        node2pin_list_end.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        num_nodes);

    return node_grad;
}

torch::Tensor GPURouter::calcFillerRouteGrad(torch::Tensor filler_pos,
                                             torch::Tensor filler_size,
                                             torch::Tensor filler_weight,
                                             torch::Tensor expand_ratio,
                                             torch::Tensor grad_mat,
                                             float grad_weight,
                                             float unit_len_x,
                                             float unit_len_y,
                                             int num_bin_x,
                                             int num_bin_y,
                                             int num_fillers) {
    int threads = 64;
    int blocks = (num_fillers + threads - 1) / threads;

    auto filler_grad = torch::zeros({num_fillers, 2}, torch::dtype(filler_pos.dtype()).device(filler_pos.device()));

    fillerRouteForce<<<blocks, threads, 0>>>(
        filler_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        filler_size.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        filler_weight.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        expand_ratio.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        filler_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        grad_mat.data_ptr<float>(),
        grad_weight, unit_len_x, unit_len_y, num_bin_x, num_bin_y, num_fillers
    );

    return filler_grad;
}

torch::Tensor GPURouter::calcPseudoPinGrad(torch::Tensor node_pos, torch::Tensor pseudo_pin_pos, float gamma) {
    const auto num_nodes = node_pos.size(0);

    const int threads = 128;
    const int blocks = (num_nodes * 2 + threads - 1) / threads;
    float inv_gamma = 1 / gamma;

    auto node_grad = torch::zeros({num_nodes, 2}, torch::dtype(node_pos.dtype()).device(node_pos.device()));
    pseudoPinForce<<<blocks, threads, 0>>>(
        node_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pseudo_pin_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        node_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        num_nodes, inv_gamma
    );

    return node_grad;
}

torch::Tensor GPURouter::calcNodeInflateRatio(torch::Tensor node_pos,
                                              torch::Tensor node_size,
                                              torch::Tensor node_weight,
                                              torch::Tensor expand_ratio,
                                              torch::Tensor inflate_mat,
                                              float grad_weight,
                                              float unit_len_x,
                                              float unit_len_y,
                                              int num_bin_x,
                                              int num_bin_y,
                                              bool use_weighted_inflation) {
    const auto num_nodes = node_pos.size(0);

    const int threads = 128;
    const int blocks = (num_nodes + threads - 1) / threads;

    auto node_inflate_ratio = torch::ones({num_nodes, 2}, torch::dtype(node_pos.dtype()).device(node_pos.device()));
    if (use_weighted_inflation) {
        inflateNodeRatioWeighted<<<blocks, threads, 0>>>(
            node_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            node_size.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            node_weight.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            expand_ratio.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            inflate_mat.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            node_inflate_ratio.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            grad_weight, unit_len_x, unit_len_y, num_bin_x, num_bin_y, num_nodes
        );
    } else {
        inflateNodeRatioMax<<<blocks, threads, 0>>>(
            node_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            node_size.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            node_weight.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            expand_ratio.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            inflate_mat.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            node_inflate_ratio.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            grad_weight, unit_len_x, unit_len_y, num_bin_x, num_bin_y, num_nodes
        );
    }

    return node_inflate_ratio;
}

torch::Tensor GPURouter::calcInflatedPinRelCpos(torch::Tensor node_inflate_ratio,
                                                torch::Tensor old_pin_rel_cpos,
                                                torch::Tensor pin_id2node_id,
                                                int num_movable_nodes) {
    const auto num_pins = old_pin_rel_cpos.size(0);

    const int threads = 128;
    const int blocks = (num_pins + threads - 1) / threads;

    auto new_pin_rel_cpos = old_pin_rel_cpos.clone();
    inflatePinRelCpos<<<blocks, threads, 0>>>(
        node_inflate_ratio.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        old_pin_rel_cpos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        pin_id2node_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        new_pin_rel_cpos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        num_movable_nodes, num_pins
    );

    return new_pin_rel_cpos;
}

}  // namespace gr