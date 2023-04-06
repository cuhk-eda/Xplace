#pragma once

#include "gpudp/dp/detailed_place_db.cuh"

namespace dp {

#define BIG_NEGATIVE -9999999
#define MAX_MINIBATCH 64

template <typename T>
inline void init_auction(const int num_graphs, const int num_nodes, char*& scratch, char*& stop_flags) {
    checkCuda(cudaMalloc(
        &scratch,
        num_graphs * (3 * num_nodes + 1) * sizeof(int) + num_graphs * (num_nodes * num_nodes + num_nodes) * sizeof(T)));
    allocateCuda(stop_flags, num_graphs, char);
}

inline void destroy_auction(char* scratch, char* stop_flags) {
    cudaFree(scratch);
    cudaFree(stop_flags);
}

template <typename T>
__global__ void __launch_bounds__(256, 4) linear_assignment_auction_kernel(const int num_nodes,
                                                                             const T* __restrict__ data_ptr,
                                                                             int* person2item_ptr,
                                                                             int* item2person_ptr,
                                                                             T* bids_ptr,
                                                                             T* prices_ptr,
                                                                             int* sbids_ptr,
                                                                             char* stop_flag_ptr,
                                                                             const float auction_max_eps,
                                                                             const float auction_min_eps,
                                                                             const float auction_factor,
                                                                             const int max_iterations) {
    const int batch_id = blockIdx.x;
    const int node_id = threadIdx.x;
    __shared__ float auction_eps;
    __shared__ int num_iteration;
    __shared__ int num_assigned;
    extern __shared__ T s_prices[];

    if (node_id == 0) {
        auction_eps = auction_max_eps;
        num_iteration = 0;
    }

    const T* __restrict__ data = data_ptr + batch_id * num_nodes * num_nodes;
    int* person2item = person2item_ptr + batch_id * num_nodes;
    int* item2person = item2person_ptr + batch_id * num_nodes;
    T* bids = bids_ptr + batch_id * num_nodes * num_nodes;
    int* sbids = sbids_ptr + batch_id * num_nodes;
    T* prices = prices_ptr + batch_id * num_nodes;
    char* stop_flag = stop_flag_ptr + batch_id;

    __syncthreads();

    while (auction_eps >= auction_min_eps && num_iteration < max_iterations) {
        // clear num_assigned
        if (node_id == 0) {
            num_assigned = 0;
        }

        // pre-init
        for (int i = node_id; i < num_nodes; i += blockDim.x) {
            person2item[i] = -1;
            item2person[i] = -1;
        }
        __syncthreads();

        // start iterative solving
        while (num_assigned < num_nodes && num_iteration < max_iterations) {
            // phase 1: init bid and bids
            for (int i = node_id; i < num_nodes; i += blockDim.x) {
                sbids[i] = 0;
            }
            for (int i = node_id; i < num_nodes * num_nodes; i += blockDim.x) {
                bids[i] = 0;
            }

            // preload price
            s_prices[node_id] = prices[node_id];
            __syncthreads();

            // phase 2: bidding
            if (person2item[node_id] == -1) {
                T top1_val = BIG_NEGATIVE;
                T top2_val = BIG_NEGATIVE;
                int top1_col;
                T tmp_val;

                for (int col = 0; col < num_nodes; col++) {
                    tmp_val = data[node_id * num_nodes + col];
                    if (tmp_val < 0) {
                        continue;
                    }
                    tmp_val = tmp_val - s_prices[col];
                    if (tmp_val >= top1_val) {
                        top2_val = top1_val;
                        top1_col = col;
                        top1_val = tmp_val;
                    } else if (tmp_val > top2_val) {
                        top2_val = tmp_val;
                    }
                }
                if (top2_val == BIG_NEGATIVE) {
                    top2_val = top1_val;
                }
                T bid = top1_val - top2_val + auction_eps;
                bids[num_nodes * top1_col + node_id] = bid;
                atomicMax(sbids + top1_col, 1);
            }

            __syncthreads();

            // phase 3 : assignment
            if (sbids[node_id] != 0) {
                T high_bid = 0;
                int high_bidder = -1;

                T tmp_bid = -1;
                for (int i = 0; i < num_nodes; i++) {
                    tmp_bid = bids[node_id * num_nodes + i];
                    if (tmp_bid > high_bid) {
                        high_bid = tmp_bid;
                        high_bidder = i;
                    }
                }

                int current_person = item2person[node_id];
                if (current_person >= 0) {
                    person2item[current_person] = -1;
                } else {
                    atomicAdd(&num_assigned, 1);
                }

                prices[node_id] += high_bid;
                person2item[high_bidder] = node_id;
                item2person[node_id] = high_bidder;
            }
            __syncthreads();

            // update iteration
            if (node_id == 0) {
                num_iteration++;
            }
            __syncthreads();
        }
        // scale auction_eps
        if (node_id == 0) {
            auction_eps *= auction_factor;
        }
        __syncthreads();
    }
    __syncthreads();
    // report whether finish solving
    if (node_id == 0) {
        *stop_flag = (num_assigned == num_nodes);
    }
}

template <typename T>
void linear_assignment_auction(const T* cost_matrics,
                               int* solutions,
                               const int num_graphs,
                               const int num_nodes,
                               char* scratch,
                               char* stop_flags,
                               const float auction_max_eps,
                               const float auction_min_eps,
                               const float auction_factor,
                               const int max_iterations) {
    // get pointers from scratch, size of scratch: num_graphs * (4*num_nodes + num_nodes*num_nodes) * 4 bytes
    int* person2item = (int*)scratch;
    int* item2person = person2item + num_graphs * num_nodes;
    int* sbids = item2person + num_graphs * num_nodes;
    T* prices = (T*)(sbids + num_graphs * num_nodes);
    T* bids = prices + num_graphs * num_nodes;

    // init
    cudaMemsetAsync(prices, 0, num_graphs * num_nodes * sizeof(T));

    // launch solver
    linear_assignment_auction_kernel<T><<<num_graphs, num_nodes, num_nodes * sizeof(T)>>>(num_nodes,
                                                                                          cost_matrics,
                                                                                          person2item,
                                                                                          item2person,
                                                                                          bids,
                                                                                          prices,
                                                                                          sbids,
                                                                                          stop_flags,
                                                                                          auction_max_eps,
                                                                                          auction_min_eps,
                                                                                          auction_factor,
                                                                                          max_iterations);
    cudaDeviceSynchronize();

    // copy solutions
    cudaMemcpy(solutions, person2item, num_graphs * num_nodes * sizeof(int), cudaMemcpyDeviceToDevice);
}

}  // namespace dp