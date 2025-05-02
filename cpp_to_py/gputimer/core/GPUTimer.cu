

#include "GPUTimer.h"
#include "gputimer/db/GTDatabase.h"
#include "gputiming.h"
#include "utils.cuh"

namespace gt {

void GPUTimer::initialize() {
    cudaMalloc(&pinCap, num_pins * (NUM_ATTR + 2) * sizeof(float));
    cudaMalloc(&pinWireCap, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&testRelatedAT, num_tests * NUM_ATTR * sizeof(float));
    cudaMalloc(&testRAT, num_tests * NUM_ATTR * sizeof(float));
    cudaMalloc(&testConstraint, num_tests * NUM_ATTR * sizeof(float));
    cudaMalloc(&pinRootRes, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&arcSlew, num_arcs * 2 * NUM_ATTR * sizeof(float));

    cudaMalloc(&net_is_clock, num_nets * sizeof(int));
    cudaMalloc(&level_list, num_pins * sizeof(int));
    cudaMalloc(&primary_outputs, num_POs * sizeof(index_type));

    cudaMemcpy(pinCap, gtdb.pin_capacitance.data(), num_pins * (NUM_ATTR + 2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(net_is_clock, gtdb.net_is_clock.data(), num_nets * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(primary_outputs, gtdb.primary_outputs.data(), gtdb.primary_outputs.size() * sizeof(index_type), cudaMemcpyHostToDevice);


    allocator = new GPULutAllocator();
    allocator->AllocateBatch(gtdb.liberty_timing_arcs);
    allocator->CopyToGPU();
    cudaMalloc((void **)&d_allocator, sizeof(GPULutAllocator));
    cudaMemcpy(d_allocator, allocator, sizeof(GPULutAllocator), cudaMemcpyHostToDevice);
    allocator->CopyToGPU(d_allocator);

    logger.info("GPUTimer initialized");

    cudaMalloc(&__pinSlew__, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&__pinLoad__, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&__pinRAT__, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&__pinAT__, num_pins * NUM_ATTR * sizeof(float));

    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinSlew, __pinSlew__, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinLoad, __pinLoad__, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinRAT, __pinRAT__, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinAT, __pinAT__, num_pins * NUM_ATTR);
}

GPUTimer::~GPUTimer() {
    logger.info("destruct GPUTimer");

    cudaFree(pinCap);
    cudaFree(pinWireCap);
    cudaFree(testRelatedAT);
    cudaFree(testRAT);
    cudaFree(testConstraint);
    cudaFree(pinRootRes);
    cudaFree(arcSlew);

    cudaFree(net_is_clock);
    cudaFree(level_list);
    cudaFree(primary_outputs);

    cudaFree(__pinSlew__);
    cudaFree(__pinLoad__);
    cudaFree(__pinRAT__);
    cudaFree(__pinAT__);

    allocator->~GPULutAllocator();
    cudaFree(d_allocator);
}

void GPUTimer::update_states() {
    cudaMemset(pinImpulse, 0, num_pins * NUM_ATTR * sizeof(float));
    cudaMemset(pinRootRes, 0, num_pins * NUM_ATTR * sizeof(float));
    cudaMemset(pinRootDelay, 0, num_pins * NUM_ATTR * sizeof(float));
    cudaMemset(pinWireCap, 0, num_pins * NUM_ATTR * sizeof(float));

    reset_val<float><<<BLOCK_NUMBER(2 * num_arcs * NUM_ATTR), BLOCK_SIZE>>>(arcDelay, 2 * num_arcs * NUM_ATTR);
    reset_val<float><<<BLOCK_NUMBER(2 * num_arcs * NUM_ATTR), BLOCK_SIZE>>>(arcSlew, 2 * num_arcs * NUM_ATTR);
    reset_val<float><<<BLOCK_NUMBER(num_tests * NUM_ATTR), BLOCK_SIZE>>>(testRelatedAT, num_tests * NUM_ATTR);
    reset_val<float><<<BLOCK_NUMBER(num_tests * NUM_ATTR), BLOCK_SIZE>>>(testRAT, num_tests * NUM_ATTR);
    reset_val<float><<<BLOCK_NUMBER(num_tests * NUM_ATTR), BLOCK_SIZE>>>(testConstraint, num_tests * NUM_ATTR);

    reset_val<index_type><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(at_prefix_pin, num_pins * NUM_ATTR);
    reset_val<index_type><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(at_prefix_arc, num_pins * NUM_ATTR);
    reset_val<index_type><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(at_prefix_attr, num_pins * NUM_ATTR);

    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(__pinSlew__, pinSlew, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(__pinLoad__, pinLoad, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(__pinRAT__, pinRAT, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(__pinAT__, pinAT, num_pins * NUM_ATTR);
    cudaDeviceSynchronize();
}

__global__ void update_endpoints_kernel0(float *pinAT, float *testRAT, int *test_id2_arc_id, index_type *timing_arc_from_pin_id, index_type *timing_arc_to_pin_id, float *endpoints0, int num_tests) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int test_idx = idx >> 2;
    const int i = idx & 0b11;
    const int el = i >> 1;
    const int rf = i & 1;
    if (test_idx < num_tests) {
        const int arc_id = test_id2_arc_id[test_idx];
        const int from_pin_id = timing_arc_from_pin_id[arc_id];
        const int to_pin_id = timing_arc_to_pin_id[arc_id];
        if (isnan(pinAT[to_pin_id * NUM_ATTR + i]) || isnan(testRAT[test_idx * NUM_ATTR + i])) return;
        if (el == 0) {
            endpoints0[test_idx * NUM_ATTR + i] = pinAT[to_pin_id * NUM_ATTR + i] - testRAT[test_idx * NUM_ATTR + i];
        } else {
            endpoints0[test_idx * NUM_ATTR + i] = testRAT[test_idx * NUM_ATTR + i] - pinAT[to_pin_id * NUM_ATTR + i];
        }
    }
}

__global__ void update_endpoints_kernel1(float *pinAT, float *pinRAT, index_type *primary_outputs, float *endpoints1, int num_POs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int po_idx = idx >> 2;
    const int i = idx & 0b11;
    const int el = i >> 1;
    if (po_idx < num_POs) {
        const int pin_idx = primary_outputs[po_idx];
        if (isnan(pinAT[pin_idx * NUM_ATTR + i]) || isnan(pinRAT[pin_idx * NUM_ATTR + i])) return;
        if (el == 0) {
            endpoints1[po_idx * NUM_ATTR + i] = pinAT[pin_idx * NUM_ATTR + i] - pinRAT[pin_idx * NUM_ATTR + i];
        } else {
            endpoints1[po_idx * NUM_ATTR + i] = pinRAT[pin_idx * NUM_ATTR + i] - pinAT[pin_idx * NUM_ATTR + i];
        }
    }
}

void GPUTimer::update_endpoints() {
    torch::Tensor endpoints0 = torch::zeros({num_tests, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::kCUDA)).contiguous();
    torch::Tensor endpoints1 = torch::zeros({num_POs, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::kCUDA)).contiguous();
    torch::fill_(endpoints0, nanf(""));
    torch::fill_(endpoints1, nanf(""));

    update_endpoints_kernel0<<<BLOCK_NUMBER(num_tests * NUM_ATTR), BLOCK_SIZE>>>(pinAT, testRAT, test_id2_arc_id, timing_arc_from_pin_id, timing_arc_to_pin_id, endpoints0.data_ptr<float>(), num_tests);
    update_endpoints_kernel1<<<BLOCK_NUMBER(num_POs * NUM_ATTR), BLOCK_SIZE>>>(pinAT, pinRAT, primary_outputs, endpoints1.data_ptr<float>(), num_POs);

    endpoint_slacks = torch::cat({endpoints0, endpoints1}, 0).contiguous();
}

}  // namespace gt
