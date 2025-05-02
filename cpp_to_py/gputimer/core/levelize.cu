
#include "GPUTimer.h"
#include "utils.cuh"
#include "gputimer/db/GTDatabase.h"

namespace gt {

__global__ void advanceLevel(index_type *frontiers,
                             index_type *next_frontiers,
                             index_type *level_list,
                             index_type *pin_fanout_list_end,
                             index_type *pin_fanout_list,
                             int *pin_num_fanin,
                             int num_frontiers,
                             int *next_num_frontiers,
                             int *last_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_frontiers) {
        index_type pin_id = frontiers[idx];
        index_type ptr = atomicAdd(last_idx, 1);
        level_list[ptr] = pin_id;
        for (index_type i = pin_fanout_list_end[pin_id]; i < pin_fanout_list_end[pin_id + 1]; i++) {
            index_type fo_pin_id = pin_fanout_list[i];
            int prev_num = atomicAdd(&pin_num_fanin[fo_pin_id], -1);
            if (prev_num == 1) {
                index_type end = atomicAdd(next_num_frontiers, 1);
                next_frontiers[end] = fo_pin_id;
            }
        }
    }
}

void checkTimingGraph(index_type* level_list_cpu, int num_pins, vector<std::string> pin_names) {
    // check which pins are not in timing graph
    std::set<index_type> pins;
    for (int i = 0; i < num_pins; i++) {
        pins.insert(i);
    }
    for (int i = 0; i < num_pins; i++) {
        index_type pin_id = level_list_cpu[i];
        pins.erase(pin_id);
    }
    for (auto pin_id : pins) {
        printf("Unconnected pin_id: %d, name: %s\n", pin_id, pin_names[pin_id].c_str());
    }
}


void GPUTimer::levelize() {
    index_type *frontiers, *next_frontiers;
    int *next_num_frontiers, *last_idx;
    int num_frontiers = gtdb.pin_frontiers.size();
    cudaMalloc(&frontiers, num_pins * sizeof(index_type));
    cudaMalloc(&next_frontiers, num_pins * sizeof(index_type));
    cudaMalloc(&next_num_frontiers, sizeof(int));
    cudaMalloc(&last_idx, sizeof(int));
    cudaMemset(next_num_frontiers, 0, sizeof(int));
    cudaMemset(last_idx, 0, sizeof(int));
    cudaMemcpy(frontiers, gtdb.pin_frontiers.data(), num_frontiers * sizeof(index_type), cudaMemcpyHostToDevice);

    level_list_end_cpu.clear();
    level_list_end_cpu.push_back(0);
    int total_num_frontiers = 0;
    while (num_frontiers) {
        total_num_frontiers += num_frontiers;
        level_list_end_cpu.push_back(total_num_frontiers);
        advanceLevel<<<BLOCK_NUMBER(num_pins), BLOCK_SIZE>>>(
            frontiers, next_frontiers, level_list, pin_fanout_list_end, pin_fanout_list, pin_num_fanin, num_frontiers, next_num_frontiers, last_idx);
        cudaMemcpy(&num_frontiers, next_num_frontiers, sizeof(int), cudaMemcpyDeviceToHost);
        device_copy<index_type><<<1, 1>>>(next_frontiers, frontiers, num_frontiers);
        cudaMemset(next_num_frontiers, 0, sizeof(int));
        // debugPrint<int><<<1, 1>>>(next_num_frontiers, 1);
    }
    cudaMalloc(&level_list_end, level_list_end_cpu.size() * sizeof(index_type));
    cudaMemcpy(level_list_end, level_list_end_cpu.data(), level_list_end_cpu.size() * sizeof(index_type), cudaMemcpyHostToDevice);
    index_type *level_list_cpu = new index_type[total_num_frontiers];
    cudaMemcpy(level_list_cpu, level_list, total_num_frontiers * sizeof(index_type), cudaMemcpyDeviceToHost);

    // checkTimingGraph(level_list_cpu, num_pins, gtdb.pin_names);
}

} // namespace gt