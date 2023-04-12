#pragma once

#include "gpudp/dp/detailed_place_db.cuh"

namespace dp {

__global__ void collect_kernel(const int* d_flags, int* d_sums, int* d_results, const int length) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_flags[tid] == 1 && tid < length) {
        d_results[d_sums[tid]] = tid;
    }
}

template <typename T, typename V>
__global__ void select_kernel_add(const T* a, const V* b, int* c) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *c = (int)(*a) + (int)(*b);
    }
}

void select(const int* d_flags, int* d_results, const int length, int* scratch, int* num_collected) {
    size_t temp_storage_bytes = 0;
    void* d_temp_storage = NULL;  // need this NULL pointer to get temp_storage_bytes
    int* prefix_sum = scratch;

    checkCuda(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_flags, prefix_sum, length));

    // Run exclusive prefix sum
    checkCuda(cub::DeviceScan::ExclusiveSum((void*)d_results, temp_storage_bytes, d_flags, prefix_sum, length));
    // cudaDeviceSynchronize();

    select_kernel_add<<<1, 1>>>(prefix_sum + (length - 1), d_flags + (length - 1), num_collected);

    collect_kernel<<<(length + 256 - 1) / 256, 256>>>(d_flags, prefix_sum, d_results, length);
    cudaDeviceSynchronize();
}

/// @brief for each node, check its first level neighbors, if they are selected, mark itself as dependent
template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__device__ void mark_dependent_nodes_self(const DetailedPlaceDBType& db,
                                          IndependentSetMatchingStateType& state,
                                          int node_id) {
    if (state.selected_markers[node_id]) {
        state.dependent_markers[node_id] = 1;
        return;
    }
    typename DetailedPlaceDBType::type node_xl = db.x[node_id];
    typename DetailedPlaceDBType::type node_yl = db.y[node_id];
    // in case all nets are masked
    int node2pin_start = db.flat_node2pin_start_map[node_id];
    int node2pin_end = db.flat_node2pin_start_map[node_id + 1];
    for (int node2pin_id = node2pin_start; node2pin_id < node2pin_end; ++node2pin_id) {
        int node_pin_id = db.flat_node2pin_map[node2pin_id];
        int net_id = db.pin2net_map[node_pin_id];
        if (db.net_mask[net_id]) {
            int net2pin_start = db.flat_net2pin_start_map[net_id];
            int net2pin_end = db.flat_net2pin_start_map[net_id + 1];
            for (int net2pin_id = net2pin_start; net2pin_id < net2pin_end; ++net2pin_id) {
                int net_pin_id = db.flat_net2pin_map[net2pin_id];
                int other_node_id = db.pin2node_map[net_pin_id];
                typename DetailedPlaceDBType::type other_node_xl = db.x[other_node_id];
                typename DetailedPlaceDBType::type other_node_yl = db.y[other_node_id];
                if (std::abs(node_xl - other_node_xl) + std::abs(node_yl - other_node_yl) < state.skip_threshold) {
                    if (other_node_id < db.num_movable_nodes && state.selected_markers[other_node_id]) {
                        state.dependent_markers[node_id] = 1;
                        return;
                    }
                }
            }
        }
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void maximal_independent_set_kernel(DetailedPlaceDBType db,
                                               IndependentSetMatchingStateType state,
                                               int* empty) {
    const int from = blockIdx.x * blockDim.x + threadIdx.x;
    const int incr = gridDim.x * blockDim.x;

    // do
    //{
    // empty = true;
    for (int node_id = from; node_id < db.num_movable_nodes; node_id += incr) {
        if (!state.dependent_markers[node_id]) {
            if (*empty) {
                atomicExch(empty, false);
            }
            // empty = false;
            bool min_node_flag = true;
            {
                typename DetailedPlaceDBType::type node_xl = db.x[node_id];
                typename DetailedPlaceDBType::type node_yl = db.y[node_id];
                int node_rank = state.ordered_nodes[node_id];
                // in case all nets are masked
                int node2pin_start = db.flat_node2pin_start_map[node_id];
                int node2pin_end = db.flat_node2pin_start_map[node_id + 1];
                for (int node2pin_id = node2pin_start; node2pin_id < node2pin_end; ++node2pin_id) {
                    int node_pin_id = db.flat_node2pin_map[node2pin_id];
                    int net_id = db.pin2net_map[node_pin_id];
                    if (db.net_mask[net_id]) {
                        int net2pin_start = db.flat_net2pin_start_map[net_id];
                        int net2pin_end = db.flat_net2pin_start_map[net_id + 1];
                        for (int net2pin_id = net2pin_start; net2pin_id < net2pin_end; ++net2pin_id) {
                            int net_pin_id = db.flat_net2pin_map[net2pin_id];
                            int other_node_id = db.pin2node_map[net_pin_id];
                            typename DetailedPlaceDBType::type other_node_xl = db.x[other_node_id];
                            typename DetailedPlaceDBType::type other_node_yl = db.y[other_node_id];
                            typename DetailedPlaceDBType::type distance =
                                abs(node_xl - other_node_xl) + abs(node_yl - other_node_yl);
                            if (other_node_id < db.num_movable_nodes && (distance < state.skip_threshold) &&
                                (state.selected_markers[other_node_id] ||
                                 (state.dependent_markers[other_node_id] == 0 &&
                                  state.ordered_nodes[other_node_id] < node_rank))) {
                                min_node_flag = false;
                                break;
                            }
                        }
                        if (!min_node_flag) {
                            break;
                        }
                    }
                }
            }
            if (min_node_flag) {
                state.selected_markers[node_id] = 1;
            }
        }
    }
    //} while (!empty);
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void mark_dependent_nodes_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state) {
    for (int node_id = blockIdx.x * blockDim.x + threadIdx.x; node_id < db.num_movable_nodes;
         node_id += blockDim.x * gridDim.x) {
        if (!state.dependent_markers[node_id]) {
            mark_dependent_nodes_self(db, state, node_id);
        }
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void init_markers_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < db.num_nodes; i += blockDim.x * gridDim.x) {
        state.selected_markers[i] = 0;
        // make sure multi-row height cells are not selected
        state.dependent_markers[i] = (db.node_size_y[i] > db.row_height);
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void maximal_independent_set(DetailedPlaceDBType const& db, IndependentSetMatchingStateType& state) {
    // if dependent_markers is 1, it means "cannot be selected"
    // if selected_markers is 1, it means "already selected"
    init_markers_kernel<<<ceilDiv(db.num_nodes, 256), 256>>>(db, state);

    int host_empty;

    int iteration = 0;
    do {
        host_empty = true;
        checkCuda(cudaMemcpy(state.independent_set_empty_flag, &host_empty, sizeof(int), cudaMemcpyHostToDevice));
        maximal_independent_set_kernel<<<ceilDiv(db.num_movable_nodes, 256), 256>>>(
            db, state, state.independent_set_empty_flag);
        mark_dependent_nodes_kernel<<<ceilDiv(db.num_movable_nodes, 256), 256>>>(db, state);
        checkCuda(cudaMemcpy(&host_empty, state.independent_set_empty_flag, sizeof(int), cudaMemcpyDeviceToHost));
        ++iteration;
    } while (!host_empty && iteration < 10);

    select(state.selected_markers,
           state.selected_maximal_independent_set,
           db.num_movable_nodes,
           state.select_scratch,
           state.device_num_selected);
    checkCuda(cudaMemcpy(&state.num_selected, state.device_num_selected, sizeof(int), cudaMemcpyDeviceToHost));
}

}  // namespace dp