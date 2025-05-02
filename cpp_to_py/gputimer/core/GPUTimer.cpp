


#include "common/common.h"
#include "common/db/Database.h"
#include "io_parser/gp/GPDatabase.h"
#include "gputimer/db/GTDatabase.h"
#include "GPUTimer.h"

namespace gt {

GPUTimer::GPUTimer(std::shared_ptr<GTDatabase> gtdb_, shared_ptr<TimingTorchRawDB> timing_raw_db_)
    : gtdb(*gtdb_),
      timing_raw_db(*timing_raw_db_),
      x(timing_raw_db.x.data_ptr<float>()),
      y(timing_raw_db.y.data_ptr<float>()),
      init_x(timing_raw_db.init_x.data_ptr<float>()),
      init_y(timing_raw_db.init_y.data_ptr<float>()),
      node_size_x(timing_raw_db.node_size_x.data_ptr<float>()),
      node_size_y(timing_raw_db.node_size_y.data_ptr<float>()),
      pin_offset_x(timing_raw_db.pin_offset_x.data_ptr<float>()),
      pin_offset_y(timing_raw_db.pin_offset_y.data_ptr<float>()),
      // GPU pin attributes array
      pinSlew(timing_raw_db.pinSlew.data_ptr<float>()),
      pinLoad(timing_raw_db.pinLoad.data_ptr<float>()),
      pinRAT(timing_raw_db.pinRAT.data_ptr<float>()),
      pinAT(timing_raw_db.pinAT.data_ptr<float>()),
      pinImpulse(timing_raw_db.pinImpulse.data_ptr<float>()),
      pinRootDelay(timing_raw_db.pinRootDelay.data_ptr<float>()),
      arcDelay(timing_raw_db.arcDelay.data_ptr<float>()),
      // Critical path prefix info
      at_prefix_pin(timing_raw_db.at_prefix_pin.data_ptr<index_type>()),
      at_prefix_arc(timing_raw_db.at_prefix_arc.data_ptr<index_type>()),
      at_prefix_attr(timing_raw_db.at_prefix_attr.data_ptr<index_type>()),
      // Timing graph topology
      pin_forward_arc_list(timing_raw_db.pin_forward_arc_list.data_ptr<index_type>()),
      pin_forward_arc_list_end(timing_raw_db.pin_forward_arc_list_end.data_ptr<index_type>()),
      pin_backward_arc_list(timing_raw_db.pin_backward_arc_list.data_ptr<index_type>()),
      pin_backward_arc_list_end(timing_raw_db.pin_backward_arc_list_end.data_ptr<index_type>()),
      timing_arc_from_pin_id(timing_raw_db.timing_arc_from_pin_id.data_ptr<index_type>()),
      timing_arc_to_pin_id(timing_raw_db.timing_arc_to_pin_id.data_ptr<index_type>()),
      pin_num_fanin(timing_raw_db.pin_num_fanin.data_ptr<int>()),
      pin_fanout_list(timing_raw_db.pin_fanout_list.data_ptr<index_type>()),
      pin_fanout_list_end(timing_raw_db.pin_fanout_list_end.data_ptr<index_type>()),
      // Timer timing liberty variables
      timing_arc_id_map(timing_raw_db.timing_arc_id_map.data_ptr<int>()),
      arc_types(timing_raw_db.arc_types.data_ptr<int>()),
      arc_id2test_id(timing_raw_db.arc_id2test_id.data_ptr<int>()),
      test_id2_arc_id(timing_raw_db.test_id2_arc_id.data_ptr<int>()),
      // Circuit info
      flat_node2pin_start_map(timing_raw_db.flat_node2pin_start_map.data_ptr<int>()),
      flat_node2pin_map(timing_raw_db.flat_node2pin_map.data_ptr<int>()),
      pin2node_map(timing_raw_db.pin2node_map.data_ptr<int>()),
      flat_net2pin_start_map(timing_raw_db.flat_net2pin_start_map.data_ptr<int>()),
      flat_net2pin_map(timing_raw_db.flat_net2pin_map.data_ptr<int>()),
      pin2net_map(timing_raw_db.pin2net_map.data_ptr<int>()),
      net_mask(timing_raw_db.net_mask.data_ptr<bool>()),
      num_threads(timing_raw_db.num_threads),
      num_nodes(timing_raw_db.num_nodes),
      num_movable_nodes(timing_raw_db.num_movable_nodes),
      num_nets(timing_raw_db.num_nets),
      num_pins(timing_raw_db.num_pins),
      scale_factor(timing_raw_db.scale_factor) {
    num_arcs = gtdb.num_arcs;
    num_timings = gtdb.num_timings;
    total_num_fanouts = gtdb.total_num_fanouts;
    num_tests = gtdb.num_tests;
    num_POs = gtdb.num_POs;
    wire_resistance_per_micron = timing_raw_db.wire_resistance_per_micron;
    wire_capacitance_per_micron = timing_raw_db.wire_capacitance_per_micron;
    microns = timing_raw_db.microns;
    res_unit = gtdb.res_unit;
    cap_unit = gtdb.cap_unit;
    if (gtdb.clocks.empty())
        clock_period = 0;
    else
        clock_period = gtdb.clocks.begin()->second.period();
    gtdb_holder = gtdb_;
    timing_raw_db_holder = timing_raw_db_;
}

torch::Tensor GPUTimer::report_pin_at() {return timing_raw_db.pinAT;}
torch::Tensor GPUTimer::report_pin_rat() { return timing_raw_db.pinRAT; }
torch::Tensor GPUTimer::report_pin_slew() { return timing_raw_db.pinSlew; }
torch::Tensor GPUTimer::report_pin_load() { return timing_raw_db.pinLoad; }
torch::Tensor GPUTimer::report_endpoint_slack() { return endpoint_slacks; }
torch::Tensor GPUTimer::endpoints_index(){ return timing_raw_db.endpoints_id;}
float GPUTimer::time_unit() const { return gtdb.time_unit; }

float GPUTimer::report_wns(int el) {
    auto ep_slacks = torch::nan_to_num(endpoint_slacks, FLT_MAX);
    return torch::min(ep_slacks.index({"...", torch::indexing::Slice(2 * el, 2 * (el + 1))})).item<float>();
}

float GPUTimer::report_tns_elw(int el) {
    auto ep_slacks = torch::nan_to_num(endpoint_slacks);
    auto [slack_elw, order] = torch::min(ep_slacks.index({"...", torch::indexing::Slice(2 * el, 2 * (el + 1))}), 1);
    slack_elw.clamp_max_(0);

    return torch::sum(slack_elw, 0).item<float>();
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> GPUTimer::report_wns_and_tns() {
    auto ep_slacks = torch::nan_to_num(endpoint_slacks, FLT_MAX);
    auto [slack_e, order_e] = torch::min(ep_slacks.index({"...", torch::indexing::Slice(0, 2)}), 1);
    slack_e.clamp_max_(0);

    auto [slack_l, order_l] = torch::min(ep_slacks.index({"...", torch::indexing::Slice(2, 4)}), 1);
    slack_l.clamp_max_(0);

    return {torch::min(ep_slacks.index({"...", torch::indexing::Slice(0, 2)})),
            torch::sum(slack_e, 0),
            torch::min(ep_slacks.index({"...", torch::indexing::Slice(2, 4)})),
            torch::sum(slack_l, 0)};
}

torch::Tensor GPUTimer::report_pin_slack() {
    pin_slacks = torch::zeros_like(timing_raw_db.pinAT, torch::dtype(torch::kFloat32).device(timing_raw_db.pinAT.device()));
    auto s1 = timing_raw_db.pinAT - timing_raw_db.pinRAT;
    auto s2 = timing_raw_db.pinRAT - timing_raw_db.pinAT;
    pin_slacks.index({"...", torch::indexing::Slice(0, 2)}).data().copy_(s1.index({"...", torch::indexing::Slice(0, 2)}));
    pin_slacks.index({"...", torch::indexing::Slice(2, 4)}).data().copy_(s2.index({"...", torch::indexing::Slice(2, 4)}));

    return pin_slacks.contiguous();
}

// void GPUTimer::update_endpoints() {
//     pin_slacks = torch::zeros_like(timing_raw_db.pinAT, torch::dtype(torch::kFloat32).device(timing_raw_db.pinAT.device()));
//     auto s1 = timing_raw_db.pinAT - timing_raw_db.pinRAT;
//     auto s2 = timing_raw_db.pinRAT - timing_raw_db.pinAT;
//     pin_slacks.index({"...", torch::indexing::Slice(0, 2)}).data().copy_(s1.index({"...", torch::indexing::Slice(0, 2)}));
//     pin_slacks.index({"...", torch::indexing::Slice(2, 4)}).data().copy_(s2.index({"...", torch::indexing::Slice(2, 4)}));
    
//     auto [endpoints_id, tmp1] = torch::_unique(timing_raw_db.endpoints_id);
//     endpoint_slacks = torch::nan_to_num(pin_slacks.index_select(0, endpoints_id));
// }

}  // namespace gt
