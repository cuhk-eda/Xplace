#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "utils.cuh"

namespace gt {

__global__ void RCTreeNet(float *x,
                          float *y,
                          const float *pin_offset_x,
                          const float *pin_offset_y,
                          const int *pin2node_map,
                          const int *flat_net2pin_start_map,
                          const int *flat_net2pin_map,
                          float *pinLoad,
                          float *pinImpulse,
                          float *pinCap,
                          float *pinWireCap,
                          float *pinRootDelay,
                          float *pinRootRes,
                          int num_nets,
                          float unit_to_micron,
                          int *net_is_clock,
                          float cf,
                          float rf) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nets) {
        int start_idx = flat_net2pin_start_map[idx];
        int end_idx = flat_net2pin_start_map[idx + 1];
        int root = flat_net2pin_map[start_idx];
        float x_root = x[pin2node_map[root]] + pin_offset_x[root];
        float y_root = y[pin2node_map[root]] + pin_offset_y[root];
        float root_cap = 0;

        // Load
        for (int i = start_idx + 1; i < end_idx; i++) {
            int pin_id = flat_net2pin_map[i];
            float x_pin = x[pin2node_map[pin_id]] + pin_offset_x[pin_id];
            float y_pin = y[pin2node_map[pin_id]] + pin_offset_y[pin_id];
            float dist = abs(x_pin - x_root) + abs(y_pin - y_root);
            float wl = dist / unit_to_micron;
            if (net_is_clock[idx]) wl = 0;
            float pin_cap = cf * wl * 0.5;
            float pin_res = rf * wl;
            root_cap += pin_cap;

            for (int j = 0; j < NUM_ATTR; j++) {
                float pin_cap_lib =
                    isnan(pinCap[pin_id * (NUM_ATTR + 2) + j]) ? pinCap[pin_id * (NUM_ATTR + 2) + 4 + (j >> 1)] : pinCap[pin_id * (NUM_ATTR + 2) + j];
                float load = pinLoad[pin_id * NUM_ATTR + j];

                pinLoad[pin_id * NUM_ATTR + j] = isnan(load) ? pin_cap + pin_cap_lib : load + pin_cap + pin_cap_lib;
                pinRootRes[pin_id * NUM_ATTR + j] = pin_res;
                pinLoad[root * NUM_ATTR + j] = isnan(pinLoad[root * NUM_ATTR + j]) ? pinLoad[pin_id * NUM_ATTR + j]
                                                                                   : pinLoad[root * NUM_ATTR + j] + pinLoad[pin_id * NUM_ATTR + j];
            }
        }
        // Root
        for (int j = 0; j < NUM_ATTR; j++) {
            float pin_cap_lib =
                isnan(pinCap[root * (NUM_ATTR + 2) + j]) ? pinCap[root * (NUM_ATTR + 2) + 4 + (j >> 1)] : pinCap[root * (NUM_ATTR + 2) + j];
            float load = pinLoad[root * NUM_ATTR + j];
            pinLoad[root * NUM_ATTR + j] = isnan(load) ? root_cap + pin_cap_lib : load + root_cap + pin_cap_lib;
        }
        // Delay
        for (int i = start_idx + 1; i < end_idx; i++) {
            int pin_id = flat_net2pin_map[i];
            for (int j = 0; j < NUM_ATTR; j++) {
                pinRootDelay[pin_id * NUM_ATTR + j] = pinRootRes[pin_id * NUM_ATTR + j] * pinLoad[pin_id * NUM_ATTR + j];
                pinImpulse[pin_id * NUM_ATTR + j] = 0;
            }
        }
        // Impulse
        for (int i = start_idx + 1; i < end_idx; i++) {
            int pin_id = flat_net2pin_map[i];
            for (int j = 0; j < NUM_ATTR; j++) {
                float pin_cap_lib =
                    isnan(pinCap[pin_id * (NUM_ATTR + 2) + j]) ? pinCap[pin_id * (NUM_ATTR + 2) + 4 + (j >> 1)] : pinCap[pin_id * (NUM_ATTR + 2) + j];
                float res = pinRootRes[pin_id * NUM_ATTR + j];
                float cap = pinLoad[pin_id * NUM_ATTR + j];
                float delay = pinRootDelay[pin_id * NUM_ATTR + j];
                pinImpulse[pin_id * NUM_ATTR + j] = sqrt(2 * res * cap * delay - delay * delay);
            }
        }
        if (end_idx - start_idx == 1) {
            for (int j = 0; j < NUM_ATTR; j++) {
                pinRootDelay[root * NUM_ATTR + j] = 0;
                pinImpulse[root * NUM_ATTR + j] = 0;
            }
        }
    }
}

void update_rc_timing_cuda(float *x,
                           float *y,
                           const float *pin_offset_x,
                           const float *pin_offset_y,
                           const int *pin2node_map,
                           const int *flat_net2pin_start_map,
                           const int *flat_net2pin_map,
                           float *pinLoad,
                           float *pinImpulse,
                           float *pinCap,
                           float *pinWireCap,
                           float *pinRootDelay,
                           float *pinRootRes,
                           int num_nets,
                           int num_pins,
                           float unit_to_micron,
                           int *net_is_clock,
                           float cf,
                           float rf) {
    RCTreeNet<<<BLOCK_NUMBER(num_nets), BLOCK_SIZE>>>(x,
                                                      y,
                                                      pin_offset_x,
                                                      pin_offset_y,
                                                      pin2node_map,
                                                      flat_net2pin_start_map,
                                                      flat_net2pin_map,
                                                      pinLoad,
                                                      pinImpulse,
                                                      pinCap,
                                                      pinWireCap,
                                                      pinRootDelay,
                                                      pinRootRes,
                                                      num_nets,
                                                      unit_to_micron,
                                                      net_is_clock,
                                                      cf,
                                                      rf);
}

__global__ void flatten_rc_kernel(const int *edge_from,
                                  const int *edge_to,
                                  const int *flat_net2node_start_map,
                                  const int *flat_net2edge_start_map,
                                  float *edge_res,
                                  float *res_parent,
                                  int *parent_node,
                                  int *root_dist,
                                  int *cnts,
                                  int *edge_cnts,
                                  int *node_order,
                                  int *edge_order,
                                  int num_nets,
                                  int num_nodes,
                                  int num_edges) {
    const int idx = blockIdx.x;
    if (idx < num_nets) {
        int nst = flat_net2node_start_map[idx];
        int nend = flat_net2node_start_map[idx + 1];
        int root = nst;

        int est = flat_net2edge_start_map[idx];
        int eend = flat_net2edge_start_map[idx + 1];

        if (threadIdx.x == 0) {
            parent_node[root] = -1;
            root_dist[root] = 0;
        }

        __syncthreads();

        for (int d = 0; d < nend - nst; d++) {
            for (int i = est + threadIdx.x; i < eend; i += blockDim.x) {
                int from = edge_from[i];
                int to = edge_to[i];
                float res = edge_res[i];
                if ((root_dist[from] == d) && (root_dist[to] == -1)) {
                    parent_node[to] = from;
                    root_dist[to] = d + 1;
                    atomicAdd(&cnts[d + nst], 1);
                    for (int j = 0; j < NUM_ATTR; j++) {
                        atomicAdd(&res_parent[to * NUM_ATTR + j], res);
                    }
                } else if ((root_dist[to] == d) && (root_dist[from] == -1)) {
                    parent_node[from] = to;
                    root_dist[from] = d + 1;
                    atomicAdd(&cnts[d + nst], 1);
                    for (int j = 0; j < NUM_ATTR; j++) {
                        atomicAdd(&res_parent[from * NUM_ATTR + j], res);
                    }
                }
            }
            __syncthreads();
            if (cnts[d + nst] == 0) break;
        }

        if (threadIdx.x == 0) {
            const int num_edges_local = eend - est;

            // calculate accumulation
            int edge_count = 0;
            for (int i = 0; i < num_edges_local; i++) {
                edge_count += cnts[i + nst];  // FIXME:
                cnts[i + nst] = edge_count;
            }

            // calculate order
            for (int i = 0; i < num_edges_local; i++) {
                int from = edge_from[i + est];
                int to = edge_to[i + est];
                int min_d = min(root_dist[from], root_dist[to]);

                int start = min_d == 0 ? 0 : cnts[min_d - 1 + nst];
                edge_order[est + start + edge_cnts[min_d + est]] = i + est;
                atomicAdd(&edge_cnts[min_d + est], 1);
            }
        }

        __syncthreads();

        // sort according to dist and cnts
        extern __shared__ int offset;
        if (threadIdx.x == 0) {
            offset = 0;
        }
        __syncthreads();
        for (int d = 0; d < nend - nst; d++) {
            // if (threadIdx.x == 0) {
            //     offset += cnts[d + nst];
            // }
            __syncthreads();
            for (int i = nst + threadIdx.x; i < nend; i += blockDim.x) {
                if (root_dist[i] == d) {
                    int pos = atomicAdd(&cnts[d + nst], -1);
                    int off = atomicAdd(&offset, 1);
                    // order[nst + offset - pos] = i;
                    node_order[nst + off] = i;
                }
            }
            __syncthreads();
        }
    }
}

__global__ void propagate_rc_kernel(const int *flat_net2node_start_map,
                                    const int *parent_node,
                                    const int *node_order,
                                    const int *node2pin_map,
                                    const float *res_parent,
                                    const float *pinCap,
                                    const float *pinLoad,
                                    const float *node_cap,
                                    float *node_load,
                                    float *node_delay,
                                    float *node_ldelay,
                                    float *node_impulse,
                                    float *node_beta,
                                    int num_nets,
                                    int num_nodes) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cond = threadIdx.y;
    if (idx < num_nets) {
        int nst = flat_net2node_start_map[idx];
        int nend = flat_net2node_start_map[idx + 1];

        for (int i = nend - 1; i >= nst; i--) {
            int node = node_order[i];
            int pnode = parent_node[node];
            int pin = node2pin_map[node];
            float wire_cap = node_cap[node * NUM_ATTR + cond];
            if (pin != -1) {
                float pin_cap_lib =
                    isnan(pinCap[pin * (NUM_ATTR + 2) + cond]) ? pinCap[pin * (NUM_ATTR + 2) + 4 + (cond >> 1)] : pinCap[pin * (NUM_ATTR + 2) + cond];
                float pin_load = pinLoad[pin * NUM_ATTR + cond];
                wire_cap = wire_cap + pin_cap_lib + pin_load;
            }
            atomicAdd(&node_load[node * NUM_ATTR + cond], wire_cap);
            if (pnode != -1) atomicAdd(&node_load[pnode * NUM_ATTR + cond], node_load[node * NUM_ATTR + cond]);
        }
        for (int i = nst + 1; i < nend; i++) {
            int node = node_order[i];
            int pnode = parent_node[node];
            int pin = node2pin_map[node];
            float t = node_load[node * NUM_ATTR + cond] * res_parent[node * NUM_ATTR + cond];
            node_delay[node * NUM_ATTR + cond] = node_delay[pnode * NUM_ATTR + cond] + t;
        }
        for (int i = nend - 1; i >= nst; i--) {
            int node = node_order[i];
            int pnode = parent_node[node];
            int pin = node2pin_map[node];
            float wire_cap = node_cap[node * NUM_ATTR + cond];
            if (pin != -1) {
                float pin_cap_lib =
                    isnan(pinCap[pin * (NUM_ATTR + 2) + cond]) ? pinCap[pin * (NUM_ATTR + 2) + 4 + (cond >> 1)] : pinCap[pin * (NUM_ATTR + 2) + cond];
                float pin_load = pinLoad[pin * NUM_ATTR + cond];
                wire_cap = wire_cap + pin_cap_lib + pin_load;
            }
            float l = wire_cap * node_delay[node * NUM_ATTR + cond];
            atomicAdd(&node_ldelay[node * NUM_ATTR + cond], l);
            if (pnode != -1) atomicAdd(&node_ldelay[pnode * NUM_ATTR + cond], node_ldelay[node * NUM_ATTR + cond]);
        }
        for (int i = nst + 1; i < nend; i++) {
            int node = node_order[i];
            int pnode = parent_node[node];
            int pin = node2pin_map[node];
            float t = node_ldelay[node * NUM_ATTR + cond] * res_parent[node * NUM_ATTR + cond];
            node_beta[node * NUM_ATTR + cond] = node_beta[pnode * NUM_ATTR + cond] + t;
            node_impulse[node * NUM_ATTR + cond] =
                sqrt(2 * node_beta[node * NUM_ATTR + cond] - node_delay[node * NUM_ATTR + cond] * node_delay[node * NUM_ATTR + cond]);
        }
    }
}

__global__ void move_to_timing_graph(const int *flat_net2node_start_map,
                                     const int *node2pin_map,
                                     const float *node_load,
                                     const float *node_delay,
                                     const float *node_impulse,
                                     float *pinLoad,
                                     float *pinImpulse,
                                     float *pinRootDelay,
                                     int num_nets) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cond = threadIdx.y;
    if (idx < num_nets) {
        int nst = flat_net2node_start_map[idx];
        int nend = flat_net2node_start_map[idx + 1];
        for (int i = nst; i < nend; i++) {
            int node = i;
            int pin = node2pin_map[node];
            if (pin != -1) {
                pinLoad[pin * NUM_ATTR + cond] = node_load[node * NUM_ATTR + cond];
                pinRootDelay[pin * NUM_ATTR + cond] = node_delay[node * NUM_ATTR + cond];
                pinImpulse[pin * NUM_ATTR + cond] = node_impulse[node * NUM_ATTR + cond];
            }
        }
    }
}

void flatten_rc_tree(std::vector<int> host_edge_from,
                     std::vector<int> host_edge_to,
                     float *edge_res,
                     float *node_cap,
                     std::vector<int> host_flat_net2node_start_map,
                     std::vector<int> host_flat_net2edge_start_map,
                     std::vector<int> host_node2pin_map,
                     int *node_order,
                     int *edge_order,
                     int *parent_node,
                     float *res_parent,
                     float *pinLoad,
                     float *pinImpulse,
                     float *pinCap,
                     float *pinWireCap,
                     float *pinRootDelay,
                     float *pinRootRes,
                     int num_nets,
                     int num_pins,
                     int num_nodes,
                     int num_edges) {
    int *edge_from, *edge_to, *flat_net2node_start_map, *flat_net2edge_start_map, *node2pin_map;

    cudaMalloc(&edge_from, host_edge_from.size() * sizeof(int));
    cudaMalloc(&edge_to, host_edge_to.size() * sizeof(int));
    cudaMalloc(&flat_net2node_start_map, host_flat_net2node_start_map.size() * sizeof(int));
    cudaMalloc(&flat_net2edge_start_map, host_flat_net2edge_start_map.size() * sizeof(int));
    cudaMalloc(&node2pin_map, host_node2pin_map.size() * sizeof(int));

    cudaMemcpy(edge_from, host_edge_from.data(), host_edge_from.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(edge_to, host_edge_to.data(), host_edge_to.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(flat_net2node_start_map, host_flat_net2node_start_map.data(), host_flat_net2node_start_map.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(flat_net2edge_start_map, host_flat_net2edge_start_map.data(), host_flat_net2edge_start_map.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(node2pin_map, host_node2pin_map.data(), host_node2pin_map.size() * sizeof(int), cudaMemcpyHostToDevice);

    int *root_dist, *cnts;
    int *edge_cnts;
    cudaMalloc(&root_dist, num_nodes * sizeof(int));
    cudaMalloc(&cnts, num_nodes * sizeof(int));
    cudaMalloc(&edge_cnts, num_edges * sizeof(int));

    reset_val<int><<<BLOCK_NUMBER(num_nodes), BLOCK_SIZE>>>(root_dist, num_nodes);
    cudaMemset(cnts, 0, num_nodes * sizeof(int));
    cudaMemset(edge_cnts, 0, num_edges * sizeof(int));

    int thread_count = 64;
    int numBlocks = num_nets;
    flatten_rc_kernel<<<numBlocks, thread_count>>>(edge_from,
                                                   edge_to,
                                                   flat_net2node_start_map,
                                                   flat_net2edge_start_map,
                                                   edge_res,
                                                   res_parent,
                                                   parent_node,
                                                   root_dist,
                                                   cnts,
                                                   edge_cnts,
                                                   node_order,
                                                   edge_order,
                                                   num_nets,
                                                   num_nodes,
                                                   num_edges);
    cudaFree(edge_from);
    cudaFree(edge_to);
    cudaFree(flat_net2node_start_map);
    cudaFree(flat_net2edge_start_map);
    cudaFree(node2pin_map);
    cudaFree(root_dist);
    cudaFree(cnts);
    cudaFree(edge_cnts);

    // device sync
    cudaDeviceSynchronize();
}

void propagate_rc_tree(std::vector<int> host_edge_from,
                       std::vector<int> host_edge_to,
                       float *edge_res,
                       float *node_cap,
                       std::vector<int> host_flat_net2node_start_map,
                       std::vector<int> host_flat_net2edge_start_map,
                       std::vector<int> host_node2pin_map,
                       int *node_order,
                       int *parent_node,
                       float *res_parent,
                       float *pinLoad,
                       float *pinImpulse,
                       float *pinCap,
                       float *pinWireCap,
                       float *pinRootDelay,
                       float *pinRootRes,
                       int num_nets,
                       int num_pins,
                       int num_nodes,
                       int num_edges) {
    int *edge_from, *edge_to, *flat_net2node_start_map, *flat_net2edge_start_map, *node2pin_map;
    cudaMalloc(&edge_from, host_edge_from.size() * sizeof(int));
    cudaMalloc(&edge_to, host_edge_to.size() * sizeof(int));
    cudaMalloc(&flat_net2node_start_map, host_flat_net2node_start_map.size() * sizeof(int));
    cudaMalloc(&flat_net2edge_start_map, host_flat_net2edge_start_map.size() * sizeof(int));
    cudaMalloc(&node2pin_map, host_node2pin_map.size() * sizeof(int));
    cudaMemcpy(edge_from, host_edge_from.data(), host_edge_from.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(edge_to, host_edge_to.data(), host_edge_to.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(flat_net2node_start_map, host_flat_net2node_start_map.data(), host_flat_net2node_start_map.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(flat_net2edge_start_map, host_flat_net2edge_start_map.data(), host_flat_net2edge_start_map.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(node2pin_map, host_node2pin_map.data(), host_node2pin_map.size() * sizeof(int), cudaMemcpyHostToDevice);

    float *node_load, *node_delay, *node_ldelay, *node_impulse, *node_beta;

    cudaMalloc(&node_load, num_nodes * NUM_ATTR * sizeof(float));
    cudaMalloc(&node_delay, num_nodes * NUM_ATTR * sizeof(float));
    cudaMalloc(&node_ldelay, num_nodes * NUM_ATTR * sizeof(float));
    cudaMalloc(&node_impulse, num_nodes * NUM_ATTR * sizeof(float));
    cudaMalloc(&node_beta, num_nodes * NUM_ATTR * sizeof(float));

    cudaMemset(node_load, 0, num_nodes * NUM_ATTR * sizeof(float));
    cudaMemset(node_delay, 0, num_nodes * NUM_ATTR * sizeof(float));
    cudaMemset(node_ldelay, 0, num_nodes * NUM_ATTR * sizeof(float));
    cudaMemset(node_impulse, 0, num_nodes * NUM_ATTR * sizeof(float));
    cudaMemset(node_beta, 0, num_nodes * NUM_ATTR * sizeof(float));

    int thread_count2 = 64;
    dim3 block_size(thread_count2, NUM_ATTR);
    int numBlocks2 = num_nets - 1 + thread_count2 / thread_count2;

    propagate_rc_kernel<<<numBlocks2, block_size>>>(flat_net2node_start_map,
                                                    parent_node,
                                                    node_order,
                                                    node2pin_map,
                                                    res_parent,
                                                    pinCap,
                                                    pinLoad,
                                                    node_cap,
                                                    node_load,
                                                    node_delay,
                                                    node_ldelay,
                                                    node_impulse,
                                                    node_beta,
                                                    num_nets,
                                                    num_nodes);

    move_to_timing_graph<<<numBlocks2, block_size>>>(flat_net2node_start_map, node2pin_map, node_load, node_delay, node_impulse, pinLoad, pinImpulse, pinRootDelay, num_nets);

    cudaFree(edge_from);
    cudaFree(edge_to);
    cudaFree(flat_net2node_start_map);
    cudaFree(flat_net2edge_start_map);
    cudaFree(node2pin_map);

    cudaFree(node_load);
    cudaFree(node_delay);
    cudaFree(node_ldelay);
    cudaFree(node_impulse);
    cudaFree(node_beta);
    cudaDeviceSynchronize();
}

__global__ void calc_rc_kernel(const int *edge_from,
                               const int *edge_to,
                               const int *flat_net2node_start_map,
                               const int *flat_net2edge_start_map,
                               int *root_dist,
                               int *cnts,
                               const int *edge_order,
                               const float *edge_wl,
                               float *node_cap,
                               float *edge_res,
                               int num_nets,
                               int num_edges,
                               int *net_is_clock,
                               float unit_to_micron,
                               float rf,
                               float cf) {
    const int idx = blockIdx.x;
    if (idx < num_nets) {
        int nst = flat_net2node_start_map[idx];
        int nend = flat_net2node_start_map[idx + 1];
        int root = nst;

        int est = flat_net2edge_start_map[idx];
        int eend = flat_net2edge_start_map[idx + 1];

        if (threadIdx.x == 0) {
            root_dist[root] = 0;
        }
        __syncthreads();

        for (int d = 0; d < nend - nst; d++) {
            for (int i = est + threadIdx.x; i < eend; i += blockDim.x) {
                int from = edge_from[i];
                int to = edge_to[i];
                float wl = edge_wl[i];
                if (net_is_clock[idx] == 1) wl = 0;
                float cap = wl * cf * 0.5 / unit_to_micron;
                float res = wl * rf / unit_to_micron;
                if ((root_dist[from] == d) && (root_dist[to] == -1)) {
                    root_dist[to] = d + 1;
                    atomicAdd(&cnts[d + nst], 1);
                    atomicAdd(&edge_res[i], res);
                    for (int j = 0; j < NUM_ATTR; j++) {
                        atomicAdd(&node_cap[to * NUM_ATTR + j], cap);
                        atomicAdd(&node_cap[from * NUM_ATTR + j], cap);
                    }
                } else if ((root_dist[to] == d) && (root_dist[from] == -1)) {
                    root_dist[from] = d + 1;
                    atomicAdd(&cnts[d + nst], 1);
                    atomicAdd(&edge_res[i], res);
                    for (int j = 0; j < NUM_ATTR; j++) {
                        atomicAdd(&node_cap[to * NUM_ATTR + j], cap);
                        atomicAdd(&node_cap[from * NUM_ATTR + j], cap);
                    }
                }
            }
            __syncthreads();
            if (cnts[d + nst] == 0) break;
        }
    }
}

void calc_res_cap(std::vector<int> host_edge_from,
                  std::vector<int> host_edge_to,
                  int *edge_order,
                  float *edge_res,
                  float *node_cap,
                  std::vector<int> host_flat_net2node_start_map,
                  std::vector<int> host_flat_net2edge_start_map,
                  std::vector<int> host_node2pin_map,
                  std::vector<float> host_edge_wl,
                  int num_nets,
                  int num_edges,
                  int num_nodes,
                  int *net_is_clock,
                  float unit_to_micron,
                  float rf,
                  float cf) {
    int *edge_from, *edge_to, *flat_net2node_start_map, *flat_net2edge_start_map, *node2pin_map;
    float *edge_wl;
    cudaMalloc(&edge_from, host_edge_from.size() * sizeof(int));
    cudaMalloc(&edge_to, host_edge_to.size() * sizeof(int));
    cudaMalloc(&flat_net2node_start_map, host_flat_net2node_start_map.size() * sizeof(int));
    cudaMalloc(&flat_net2edge_start_map, host_flat_net2edge_start_map.size() * sizeof(int));
    cudaMalloc(&node2pin_map, host_node2pin_map.size() * sizeof(int));
    cudaMalloc(&edge_wl, host_edge_wl.size() * sizeof(float));

    cudaMemcpy(edge_from, host_edge_from.data(), host_edge_from.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(edge_to, host_edge_to.data(), host_edge_to.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(flat_net2node_start_map, host_flat_net2node_start_map.data(), host_flat_net2node_start_map.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(flat_net2edge_start_map, host_flat_net2edge_start_map.data(), host_flat_net2edge_start_map.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(node2pin_map, host_node2pin_map.data(), host_node2pin_map.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(edge_wl, host_edge_wl.data(), host_edge_wl.size() * sizeof(float), cudaMemcpyHostToDevice);

    int *root_dist, *cnts;
    cudaMalloc(&root_dist, num_nodes * sizeof(int));
    cudaMalloc(&cnts, num_nodes * sizeof(int));

    reset_val<int><<<BLOCK_NUMBER(num_nodes), BLOCK_SIZE>>>(root_dist, num_nodes);
    cudaMemset(cnts, 0, num_nodes * sizeof(int));

    int thread_count = 64;
    int numBlocks = num_nets;
    calc_rc_kernel<<<numBlocks, thread_count>>>(edge_from,
                                                edge_to,
                                                flat_net2node_start_map,
                                                flat_net2edge_start_map,
                                                root_dist,
                                                cnts,
                                                edge_order,
                                                edge_wl,
                                                node_cap,
                                                edge_res,
                                                num_nets,
                                                num_edges,
                                                net_is_clock,
                                                unit_to_micron,
                                                rf,
                                                cf);

    cudaFree(edge_from);
    cudaFree(edge_to);
    cudaFree(flat_net2node_start_map);
    cudaFree(flat_net2edge_start_map);
    cudaFree(node2pin_map);
    cudaFree(root_dist);
    cudaFree(cnts);
    cudaFree(edge_wl);
}

}  // namespace gt