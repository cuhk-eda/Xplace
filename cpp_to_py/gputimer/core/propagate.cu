
#include "gputiming.h"
#include "utils.cuh"

namespace gt {

__device__ void propagateSlew(index_type arc_id,
                              index_type from_pin_id,
                              index_type to_pin_id,
                              float *pinSlew,
                              float *pinLoad,
                              float *pinImpulse,
                              float *pinRootDelay,
                              float *arcDelay,
                              int arc_type,
                              int *timing_arc_id_map,
                              GPULutAllocator *d_allocator) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;
    if ((arc_type == 0) && (i < NUM_ATTR)) {
        float si = pinSlew[from_pin_id * NUM_ATTR + i];
        if (isnan(si)) return;
        float imp = pinImpulse[to_pin_id * NUM_ATTR + i];
        float so = si < 0.0 ? -sqrt(si * si + imp * imp) : sqrt(si * si + imp * imp);
        pinSlew[to_pin_id * NUM_ATTR + i] = so;
    } else if (arc_type == 1) {
        int el = i >> 2;
        int fel_rf = i >> 1;
        int tel_rf = ((i & 0b100) >> 1) + (i & 1);
        int irf = fel_rf & 1;
        int orf = tel_rf & 1;
        if ((timing_arc_id_map[arc_id * 2 + el] == -1) || isnan(pinSlew[from_pin_id * NUM_ATTR + fel_rf])) return;
        float si = pinSlew[from_pin_id * NUM_ATTR + fel_rf];
        float lc = pinLoad[to_pin_id * NUM_ATTR + tel_rf];
        int timing_id = timing_arc_id_map[arc_id * 2 + el];
        float so = d_allocator->query(timing_id, irf, orf, si, lc, 1);
        if (isnan(so)) return;
        if (isnan(pinSlew[to_pin_id * NUM_ATTR + tel_rf]) || ((pinSlew[to_pin_id * NUM_ATTR + tel_rf] > so) ^ el)) {
            atomicExch(&pinSlew[to_pin_id * NUM_ATTR + tel_rf], so);
        }
    }
}

__device__ void propagateDelay(index_type arc_id,
                               index_type from_pin_id,
                               index_type to_pin_id,
                               float *pinSlew,
                               float *pinLoad,
                               float *pinImpulse,
                               float *pinRootDelay,
                               float *arcDelay,
                               int arc_type,
                               int *timing_arc_id_map,
                               GPULutAllocator *d_allocator) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;
    if ((arc_type == 0) && (i < NUM_ATTR)) {
        float delay = pinRootDelay[to_pin_id * NUM_ATTR + i];
        int el_rf_rf = (i << 1) + (i & 1);
        arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf] = delay;
    } else if (arc_type == 1) {
        int el = i >> 2;
        int fel_rf = i >> 1;
        int tel_rf = ((i & 0b100) >> 1) + (i & 1);
        int irf = fel_rf & 1;
        int orf = tel_rf & 1;
        if ((timing_arc_id_map[arc_id * 2 + el] == -1) || isnan(pinSlew[from_pin_id * NUM_ATTR + fel_rf])) return;
        float si = pinSlew[from_pin_id * NUM_ATTR + fel_rf];
        float lc = pinLoad[to_pin_id * NUM_ATTR + tel_rf];
        int timing_id = timing_arc_id_map[arc_id * 2 + el];
        float delay = d_allocator->query(timing_id, irf, orf, si, lc, 0);
        if (isnan(delay)) return;
        arcDelay[arc_id * 2 * NUM_ATTR + i] = delay;
    }
}

__device__ void propagateAT(index_type arc_id,
                            index_type from_pin_id,
                            index_type to_pin_id,
                            float *pinAt,
                            float *arcDelay,
                            index_type *at_prefix_pin,
                            index_type *at_prefix_arc,
                            index_type *at_prefix_attr) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;
    int el = i >> 2;
    int fel_rf = i >> 1;
    int tel_rf = ((i & 0b100) >> 1) + (i & 1);
    int irf = fel_rf & 1;
    int orf = tel_rf & 1;
    if (isnan(pinAt[from_pin_id * NUM_ATTR + fel_rf]) || isnan(arcDelay[arc_id * 2 * NUM_ATTR + i])) return;
    float delay = arcDelay[arc_id * 2 * NUM_ATTR + i];
    float at = pinAt[from_pin_id * NUM_ATTR + fel_rf] + delay;

    // FIXME: conflict
    if (isnan(pinAt[to_pin_id * NUM_ATTR + tel_rf]) || ((pinAt[to_pin_id * NUM_ATTR + tel_rf] > at) ^ el)) {
        atomicExch(&pinAt[to_pin_id * NUM_ATTR + tel_rf], at);
        at_prefix_pin[to_pin_id * NUM_ATTR + tel_rf] = from_pin_id;
        at_prefix_arc[to_pin_id * NUM_ATTR + tel_rf] = arc_id;
        at_prefix_attr[to_pin_id * NUM_ATTR + tel_rf] = fel_rf;
    }
}

__device__ void propagateTest(index_type arc_id,
                              index_type test_id,
                              index_type from_pin_id,
                              index_type to_pin_id,
                              int *timing_arc_id_map,
                              float *pinSlew,
                              float *pinAt,
                              float *pinRat,
                              float *testRelatedAT,
                              float *testRAT,
                              float *testConstraint,
                              float clock_period,
                              GPULutAllocator *d_allocator) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;
    if (i < NUM_ATTR) {
        const int el = i >> 1;
        const int rf = i & 1;
        const int el_rf_rf = (i << 1) + (i & 1);
        if ((timing_arc_id_map[arc_id * 2 + el] == -1) || (isnan(pinSlew[to_pin_id * NUM_ATTR + i]))) return;
        int fel = el ^ 1;
        int timing_id = timing_arc_id_map[arc_id * 2 + el];
        int frf = d_allocator->d_is_rising_edge_triggered[timing_id] ? 0 : 1;
        if (frf && !d_allocator->d_is_falling_edge_triggered[timing_id]) {
            return;
        }
        const int fel_rf = (fel << 1) + frf;
        if (isnan(pinAt[from_pin_id * NUM_ATTR + fel_rf]) || isnan(pinSlew[from_pin_id * NUM_ATTR + fel_rf])) return;

        if (el == 0) {
            testRelatedAT[test_id * NUM_ATTR + i] = pinAt[from_pin_id * NUM_ATTR + fel_rf];
        } else {
            testRelatedAT[test_id * NUM_ATTR + i] = pinAt[from_pin_id * NUM_ATTR + fel_rf] + clock_period;
        }

        float sr = pinSlew[from_pin_id * NUM_ATTR + fel_rf];
        float sc = pinSlew[to_pin_id * NUM_ATTR + i];
        testConstraint[test_id * NUM_ATTR + i] = d_allocator->query(timing_id, frf, rf, sr, sc, 2);

        if (!isnan(testConstraint[test_id * NUM_ATTR + i]) && !isnan(testRelatedAT[test_id * NUM_ATTR + i])) {
            if (el == 0) {
                pinRat[to_pin_id * NUM_ATTR + i] = testRelatedAT[test_id * NUM_ATTR + i] + testConstraint[test_id * NUM_ATTR + i];
            } else {
                pinRat[to_pin_id * NUM_ATTR + i] = testRelatedAT[test_id * NUM_ATTR + i] - testConstraint[test_id * NUM_ATTR + i];
            }
            testRAT[test_id * NUM_ATTR + i] = pinRat[to_pin_id * NUM_ATTR + i];
        }
    }
}

__global__ void propagatePin(index_type *level_list,
                             index_type *pin_backward_arc_list_end,
                             index_type *pin_backward_arc_list,
                             index_type *timing_arc_from_pin_id,
                             int *arc_types,
                             int *arc_id2test_id,
                             float *pinSlew,
                             float *pinLoad,
                             float *pinImpulse,
                             float *pinRootDelay,
                             float *pinAt,
                             float *pinRat,
                             float *testRelatedAT,
                             float *testRAT,
                             float *testConstraint,
                             float *arcDelay,
                             int *timing_arc_id_map,
                             index_type *at_prefix_pin,
                             index_type *at_prefix_arc,
                             index_type *at_prefix_attr,
                             index_type level_start_offset,
                             int num_pins_level,
                             float clock_period,
                             GPULutAllocator *d_allocator) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_idx = idx >> 3;
    if (pin_idx < num_pins_level) {
        index_type to_pin_id = level_list[level_start_offset + pin_idx];
        for (index_type i = pin_backward_arc_list_end[to_pin_id]; i < pin_backward_arc_list_end[to_pin_id + 1]; i++) {
            index_type arc_id = pin_backward_arc_list[i];
            index_type from_pin_id = timing_arc_from_pin_id[arc_id];
            int arc_type = arc_types[arc_id];
            propagateSlew(arc_id, from_pin_id, to_pin_id, pinSlew, pinLoad, pinImpulse, pinRootDelay, arcDelay, arc_type, timing_arc_id_map, d_allocator);
            propagateDelay(arc_id, from_pin_id, to_pin_id, pinSlew, pinLoad, pinImpulse, pinRootDelay, arcDelay, arc_type, timing_arc_id_map, d_allocator);
            propagateAT(arc_id, from_pin_id, to_pin_id, pinAt, arcDelay, at_prefix_pin, at_prefix_arc, at_prefix_attr);
            int test_id = arc_id2test_id[arc_id];
            if (clock_period > 0 && test_id != -1) {
                propagateTest(arc_id, test_id, from_pin_id, to_pin_id, timing_arc_id_map, pinSlew, pinAt, pinRat, testRelatedAT, testRAT, testConstraint, clock_period, d_allocator);
            }
        }
    }
}

__device__ void propagateRAT(index_type arc_id,
                             int arc_type,
                             index_type from_pin_id,
                             index_type to_pin_id,
                             float *pinAt,
                             float *pinRat,
                             float *arcDelay,
                             int *timing_arc_id_map,
                             float *from_rats,
                             GPULutAllocator *d_allocator) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;
    if ((arc_type == 0) && (i < NUM_ATTR)) {
        const int el_rf_rf = (i << 1) + (i & 1);
        const int el = i >> 1;
        if (isnan(pinRat[to_pin_id * NUM_ATTR + i]) || isnan(arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf])) return;
        float delay = arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf];
        float rat = pinRat[to_pin_id * NUM_ATTR + i] - delay;
        if (isnan(pinRat[from_pin_id * NUM_ATTR + i]) || ((pinRat[from_pin_id * NUM_ATTR + i] < rat) ^ el)) {
            atomicExch(&pinRat[from_pin_id * NUM_ATTR + i], rat);
        }
    } else if (arc_type == 1) {
        int el = i >> 2;
        int fel_rf = i >> 1;
        int tel_rf = ((i & 0b100) >> 1) + (i & 1);
        int irf = fel_rf & 1;
        int orf = tel_rf & 1;
        if (timing_arc_id_map[arc_id * 2 + el] == -1) return;
        int timing_id = timing_arc_id_map[arc_id * 2 + el];
        if (!d_allocator->d_is_constraint[timing_id]) {
            if (isnan(pinRat[to_pin_id * NUM_ATTR + tel_rf]) || isnan(arcDelay[arc_id * 2 * NUM_ATTR + i])) return;
            float delay = arcDelay[arc_id * 2 * NUM_ATTR + i];
            float rat = pinRat[to_pin_id * NUM_ATTR + tel_rf] - delay;
            from_rats[threadIdx.x] = rat;
        } else {
            if (!d_allocator->is_transition_defined(timing_id, irf, orf)) return;
            if (el == 0) {
                const int fel_rf = 2 + irf;
                const int tel_rf = orf;
                float at = pinAt[from_pin_id * NUM_ATTR + fel_rf];
                if (isnan(pinRat[to_pin_id * NUM_ATTR + tel_rf]) || isnan(pinAt[to_pin_id * NUM_ATTR + tel_rf]) || isnan(at)) return;
                float slack = (pinRat[to_pin_id * NUM_ATTR + tel_rf] - pinAt[to_pin_id * NUM_ATTR + tel_rf]) * -1;
                float rat = at + slack;
                from_rats[threadIdx.x] = rat;
            } else {
                const int fel_rf = irf;
                const int tel_rf = 2 + orf;
                float at = pinAt[from_pin_id * NUM_ATTR + fel_rf];
                if (isnan(pinRat[to_pin_id * NUM_ATTR + tel_rf]) || isnan(pinAt[to_pin_id * NUM_ATTR + tel_rf]) || isnan(at)) return;
                float slack = (pinRat[to_pin_id * NUM_ATTR + tel_rf] - pinAt[to_pin_id * NUM_ATTR + tel_rf]);
                float rat = at - slack;
                from_rats[threadIdx.x] = rat;
            }
        }
    }
}

__global__ void propagatePinBack(index_type *level_list,
                                 index_type *pin_forward_arc_list_end,
                                 index_type *pin_forward_arc_list,
                                 index_type *timing_arc_to_pin_id,
                                 int *arc_types,
                                 int *arc_id2test_id,
                                 float *pinSlew,
                                 float *pinLoad,
                                 float *pinImpulse,
                                 float *pinRootDelay,
                                 float *pinAt,
                                 float *pinRat,
                                 float *testRelatedAT,
                                 float *testConstraint,
                                 float *arcDelay,
                                 int *timing_arc_id_map,
                                 index_type level_start_offset,
                                 int num_pins_level,
                                 float clock_period,
                                 GPULutAllocator *d_allocator) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_idx = idx >> 3;
    extern __shared__ float from_rats[];

    if (pin_idx < num_pins_level) {
        index_type from_pin_id = level_list[level_start_offset + pin_idx];
        for (index_type i = pin_forward_arc_list_end[from_pin_id]; i < pin_forward_arc_list_end[from_pin_id + 1]; i++) {
            index_type arc_id = pin_forward_arc_list[i];
            index_type to_pin_id = timing_arc_to_pin_id[arc_id];
            int arc_type = arc_types[arc_id];
            if ((threadIdx.x % (2 * NUM_ATTR)) == 0) {
                for (int i = threadIdx.x; i < threadIdx.x + 2 * NUM_ATTR; i++) from_rats[i] = nanf("");
            }
            __syncthreads();

            propagateRAT(arc_id, arc_type, from_pin_id, to_pin_id, pinAt, pinRat, arcDelay, timing_arc_id_map, from_rats, d_allocator);

            __syncthreads();
            if ((threadIdx.x % (2 * NUM_ATTR)) == 0) {
                for (int ti = threadIdx.x; ti < threadIdx.x + 2 * NUM_ATTR; ti++) {
                    const int i = ti & 0b111;
                    if (isnan(from_rats[ti])) continue;
                    int el = i >> 2;
                    int fel_rf = i >> 1;
                    int tel_rf = ((i & 0b100) >> 1) + (i & 1);
                    int irf = fel_rf & 1;
                    int orf = tel_rf & 1;
                    int timing_id = timing_arc_id_map[arc_id * 2 + el];
                    float rat = from_rats[ti];
                    if (!d_allocator->d_is_constraint[timing_id]) {
                        if (isnan(pinRat[from_pin_id * NUM_ATTR + fel_rf]) || ((pinRat[from_pin_id * NUM_ATTR + fel_rf] < rat) ^ el)) {
                            atomicExch(&pinRat[from_pin_id * NUM_ATTR + fel_rf], rat);
                        }
                    } else {
                        if (el == 0) {
                            const int fel_rf = 2 + irf;
                            const int tel_rf = orf;
                            if (isnan(pinRat[from_pin_id * NUM_ATTR + fel_rf]) || (pinRat[from_pin_id * NUM_ATTR + fel_rf] > rat)) {
                                atomicExch(&pinRat[from_pin_id * NUM_ATTR + fel_rf], rat);
                            }
                        } else {
                            const int fel_rf = irf;
                            const int tel_rf = 2 + orf;
                            if (isnan(pinRat[from_pin_id * NUM_ATTR + fel_rf]) || (pinRat[from_pin_id * NUM_ATTR + fel_rf] < rat)) {
                                atomicExch(&pinRat[from_pin_id * NUM_ATTR + fel_rf], rat);
                            }
                        }
                    }
                }
            }
        }
    }
}

void update_timing_cuda(index_type *level_list,
                        vector<int> level_list_end_cpu,
                        index_type *pin_forward_arc_list_end,
                        index_type *pin_forward_arc_list,
                        index_type *timing_arc_to_pin_id,
                        index_type *pin_backward_arc_list_end,
                        index_type *pin_backward_arc_list,
                        index_type *timing_arc_from_pin_id,
                        int *arc_types,
                        int *arc_id2test_id,
                        float *pinSlew,
                        float *pinLoad,
                        float *pinImpulse,
                        float *pinRootDelay,
                        float *pinAt,
                        float *pinRat,
                        float *testRelatedAT,
                        float *testRAT,
                        float *testConstraint,
                        float *arcDelay,
                        int *timing_arc_id_map,
                        index_type *at_prefix_pin,
                        index_type *at_prefix_arc,
                        index_type *at_prefix_attr,
                        float clock_period,
                        GPULutAllocator *d_allocator,
                        int num_pins,
                        bool deterministic) {
    for (int i = 1; i < level_list_end_cpu.size() - 1; i++) {
        int num_pins_level = level_list_end_cpu[i + 1] - level_list_end_cpu[i];
        index_type level_start_offset = level_list_end_cpu[i];
        // printf("==== level %d ======= %d \n", i, num_pins_level);
        propagatePin<<<BLOCK_NUMBER(num_pins_level * 2 * NUM_ATTR), BLOCK_SIZE>>>(level_list,
                                                                                  pin_backward_arc_list_end,
                                                                                  pin_backward_arc_list,
                                                                                  timing_arc_from_pin_id,
                                                                                  arc_types,
                                                                                  arc_id2test_id,
                                                                                  pinSlew,
                                                                                  pinLoad,
                                                                                  pinImpulse,
                                                                                  pinRootDelay,
                                                                                  pinAt,
                                                                                  pinRat,
                                                                                  testRelatedAT,
                                                                                  testRAT,
                                                                                  testConstraint,
                                                                                  arcDelay,
                                                                                  timing_arc_id_map,
                                                                                  at_prefix_pin,
                                                                                  at_prefix_arc,
                                                                                  at_prefix_attr,
                                                                                  level_start_offset,
                                                                                  num_pins_level,
                                                                                  clock_period,
                                                                                  d_allocator);

        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();

    for (int i = level_list_end_cpu.size() - 3; i >= 0; i--) {
        int num_pins_level = level_list_end_cpu[i + 1] - level_list_end_cpu[i];
        index_type level_start_offset = level_list_end_cpu[i];
        // printf("==== level %d ======= %d \n", i, num_pins_level);
        propagatePinBack<<<BLOCK_NUMBER(num_pins_level * 2 * NUM_ATTR), BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(level_list,
                                                                                                                  pin_forward_arc_list_end,
                                                                                                                  pin_forward_arc_list,
                                                                                                                  timing_arc_to_pin_id,
                                                                                                                  arc_types,
                                                                                                                  arc_id2test_id,
                                                                                                                  pinSlew,
                                                                                                                  pinLoad,
                                                                                                                  pinImpulse,
                                                                                                                  pinRootDelay,
                                                                                                                  pinAt,
                                                                                                                  pinRat,
                                                                                                                  testRelatedAT,
                                                                                                                  testConstraint,
                                                                                                                  arcDelay,
                                                                                                                  timing_arc_id_map,
                                                                                                                  level_start_offset,
                                                                                                                  num_pins_level,
                                                                                                                  clock_period,
                                                                                                                  d_allocator);

        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
}

}  // namespace gt
