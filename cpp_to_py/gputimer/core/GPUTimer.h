#pragma once

#include <torch/extension.h>
#include "common/common.h"
#include "common/lib/spef/parser-spef.hpp"
#include "gputimer/base.h"

using std::tuple;
using std::vector;
using std::shared_ptr;

namespace gt {

class TimingArc;
class TimingTorchRawDB;
class GTDatabase;
class GPULutAllocator;

class GPUTimer {
public:
    GPULutAllocator *allocator;
    GPULutAllocator *d_allocator;
    GTDatabase& gtdb;
    TimingTorchRawDB& timing_raw_db;
    shared_ptr<GTDatabase> gtdb_holder;
    shared_ptr<TimingTorchRawDB> timing_raw_db_holder;
    GPUTimer(shared_ptr<GTDatabase> gtdb_, shared_ptr<TimingTorchRawDB> timing_raw_db_);
    ~GPUTimer();
    spef::Spef spef;
    void read_spef(const std::string& file);

    // === functions ===
    void initialize();
    void levelize();
    void update_rc_timing(torch::Tensor node_lpos, bool record = false, bool load = false, bool conpensation = false);
    void update_rc_timing_flute(torch::Tensor node_lpos, bool record = false);
    void update_rc_timing_spef();
    void update_states();
    void update_timing();
    void update_endpoints();
    
    float report_wns(int el);
    float report_tns_elw(int el);
    tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> report_wns_and_tns();
    torch::Tensor report_pin_slack();
    torch::Tensor endpoints_index();
    torch::Tensor report_endpoint_slack();
    torch::Tensor report_pin_at();
    torch::Tensor report_pin_rat();
    torch::Tensor report_pin_slew();
    torch::Tensor report_pin_load();

    tuple<vector<int64_t>, vector<float>, vector<float>> report_path(int ep_idx = -1, int el = -1, bool verbose = false);
    vector<vector<int64_t>> report_K_path(int K, bool verbose = false);
    tuple<torch::Tensor, torch::Tensor> report_criticality(int K, bool verbose = false, bool deterministic = true);
    tuple<torch::Tensor, torch::Tensor> report_criticality_threshold(float thrs, bool verbose = false, bool deterministic = true);

public:
    float time_unit() const;

public:
    int num_pins, num_arcs, num_timings, num_tests, num_POs, total_num_fanouts;
    float *pinSlew, *pinLoad, *pinRAT, *pinAT;
    float *pinImpulse, *pinRootDelay, *pinRootRes;
    float *arcDelay, *arcSlew;
    float *pinCap, *pinWireCap;
    float *testRelatedAT, *testConstraint, *testRAT;

    float *__pinSlew__, *__pinLoad__, *__pinRAT__, *__pinAT__;
    float *pinImpulse_ref, *pinLoad_ref, *pinRootDelay_ref;
    float *pinLoad_ratio, *pinRootDelay_ratio;

    int* pin_num_fanin;
    index_type *pin_fanout_list_end, *pin_fanout_list;
    index_type *pin_forward_arc_list_end, *pin_forward_arc_list;
    index_type *pin_backward_arc_list_end, *pin_backward_arc_list;
    index_type *timing_arc_from_pin_id, *timing_arc_to_pin_id;
    int *arc_types, *timing_arc_id_map, *arc_id2test_id;
    int* test_id2_arc_id;

    index_type* primary_outputs;
    TimingArc* liberty_timing_arcs;
    index_type *level_list_end, *level_list;
    vector<int> level_list_end_cpu;
    int* net_is_clock;

    float clock_period;

public:
    float* x;
    float* y;
    const float* init_x;
    const float* init_y;
    const float* node_size_x;
    const float* node_size_y;

    const float* pin_offset_x;
    const float* pin_offset_y;
    index_type *at_prefix_pin;
    index_type *at_prefix_arc;
    index_type *at_prefix_attr;

    const int* flat_node2pin_start_map;
    const int* flat_node2pin_map;
    const int* pin2node_map;

    const int* flat_net2pin_start_map;
    const int* flat_net2pin_map;
    const int* pin2net_map;
    const bool* net_mask;

    /* row info */
    int num_nets;
    int num_movable_nodes;
    int num_nodes;

    int num_threads;

    float wire_resistance_per_micron;
    float wire_capacitance_per_micron;
    int microns;
    float scale_factor;
    float res_unit;
    float cap_unit;

    torch::Tensor pin_slacks;
    torch::Tensor endpoint_slacks;
};

}  // namespace gt
