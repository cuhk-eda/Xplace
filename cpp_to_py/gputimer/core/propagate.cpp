
#include "GPUTimer.h"

namespace gt {

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
                        float *pinAT,
                        float *pinRAT,
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
                        bool deterministic);


void GPUTimer::update_timing() {
    update_timing_cuda(level_list,
                       level_list_end_cpu,
                       pin_forward_arc_list_end,
                       pin_forward_arc_list,
                       timing_arc_to_pin_id,
                       pin_backward_arc_list_end,
                       pin_backward_arc_list,
                       timing_arc_from_pin_id,
                       arc_types,
                       arc_id2test_id,
                       pinSlew,
                       pinLoad,
                       pinImpulse,
                       pinRootDelay,
                       pinAT,
                       pinRAT,
                       testRelatedAT,
                       testRAT,
                       testConstraint,
                       arcDelay,
                       timing_arc_id_map,
                       at_prefix_pin,
                       at_prefix_arc,
                       at_prefix_attr,
                       clock_period,
                       d_allocator,
                       num_pins,
                       true);
}

}  // namespace gt