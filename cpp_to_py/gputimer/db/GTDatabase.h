

#pragma once
#include <torch/extension.h>

#include <memory>

#include "common/common.h"
#include "common/lib/sdc/sdc.h"
#include "gputimer/base.h"

using std::set;
using std::shared_ptr;
using std::string;
using std::unordered_map;
using std::vector;
using std::array;

namespace gp {
class GPDatabase;
class GPPin;
class GPNet;
};  // namespace gp

namespace db {
class Database;
};  // namespace db

namespace gt {
class CellLib;
class LibertyCell;
class LibertyPort;
class TimingArc;
class Lut;
};  // namespace gt

namespace gt {
class TimingTorchRawDB;
class STAPin;
class clock;
class Net;
class Arc;
class cellpin;

class Clock {
private:
    std::string _name;
    float _period = .0f;
    int _source_id = -1;
public:
    Clock(const std::string& name, float period) : _name(name), _source_id(-1), _period(period) {};
    Clock(const std::string& name, int source_id, float period) : _name(name), _source_id(source_id), _period(period) {};
    inline const std::string& name() const;
    inline float period() const { return _period; }
    inline int source_id() { return _source_id; }
};

class STAPin {
public:
    vector<index_type> timing_arc_in;
    vector<index_type> timing_arc_out;
    set<index_type> fanin_pin_ids;
    set<index_type> fanout_pin_ids;
};

class GTDatabase {
public:
    db::Database& rawdb;
    gp::GPDatabase& gpdb;
    TimingTorchRawDB& timing_raw_db;
    std::array<std::shared_ptr<gt::CellLib>, MAX_SPLIT> cell_libs_;

    GTDatabase(shared_ptr<db::Database> rawdb_, shared_ptr<gp::GPDatabase> gpdb_, shared_ptr<TimingTorchRawDB> timing_raw_db_);
    ~GTDatabase() { logger.info("destruct gtdb"); }

public:
    void ExtractTimingGraph();
    void readSpef(const std::string& file);
    void readSdc(sdc::SDC& sdc);
    void _read_sdc(sdc::SetInputDelay&);
    void _read_sdc(sdc::SetDrivingCell&);
    void _read_sdc(sdc::SetInputTransition&);
    void _read_sdc(sdc::SetOutputDelay&);
    void _read_sdc(sdc::SetLoad&);
    void _read_sdc(sdc::CreateClock&);
    void _read_sdc(sdc::SetUnits&);
    bool is_redundant_timing(const TimingArc* timing_arc, Split el);

    // Units
    float res_unit;
    float cap_unit;
    float time_unit;

    std::optional<float> sdc_res_unit;
    std::optional<float> sdc_cap_unit;
    std::optional<float> sdc_time_unit;

    std::optional<float> spef_res_unit;
    std::optional<float> spef_cap_unit;
    std::optional<float> spef_time_unit;

public:
    vector<string> pin_names;
    vector<string> net_names;
    unordered_map<std::string, Clock> clocks;
    unordered_map<string, index_type> primary_input2pin_id;
    unordered_map<string, index_type> primary_output2pin_id;

    vector<STAPin*> STA_pins;
    vector<int> endpoints_id;

    vector<int> liberty_cell_type2port_list_end = {0};
    vector<int> liberty_port2timing_list_end = {0};
    vector<float> liberty_port_capacitance;
    vector<TimingArc*> liberty_timing_arcs;

    vector<int> pin_id2cell_type_id;
    vector<int> pin_id2port_offset_id;
    vector<int> cell_node_type_map;
    vector<float> pin_capacitance;

    int num_arcs = 0;
    int num_tests = 0;
    int num_timings = 0;
    int total_num_fanouts = 0;
    int num_pins;
    int num_POs;

    vector<int> timing_arc_from_pin_id, timing_arc_to_pin_id;
    vector<int> timing_arc_id_map;
    vector<int> arc_types, arc_id2test_id;
    vector<int> test_id2_arc_id;
    vector<int> net_is_clock;

    // Timing Graph
    /// @param primary_inputs                 primary input pins
    /// @param primary_outputs                primary output pins
    /// @param pin_frontiers                  frontier pins
    /// @param pin_fanout_list_end            pin_fanout start and end index
    /// @param pin_fanout_list                pin_fanout list of index
    /// @param pin_num_fanin                  number of fanin pins
    /// @param pin_forward_arc_list_end       pin_forward_arc start and end index
    /// @param pin_forward_arc_list           pin_forward_arc list of index
    /// @param pin_backward_arc_list_end      pin_backward_arc start and end index
    /// @param pin_backward_arc_list          pin_backward_arc list of index
    vector<index_type> primary_inputs, primary_outputs;
    vector<index_type> pin_frontiers;
    vector<index_type> pin_fanout_list_end, pin_fanout_list;
    vector<int> pin_num_fanin;
    vector<index_type> pin_forward_arc_list_end, pin_forward_arc_list;
    vector<index_type> pin_backward_arc_list_end, pin_backward_arc_list;

};

class TimingTorchRawDB {
public:
    TimingTorchRawDB(torch::Tensor node_lpos_init_,
                     torch::Tensor node_size_,
                     torch::Tensor pin_rel_lpos_,
                     torch::Tensor pin_id2node_id_,
                     torch::Tensor pin_id2net_id_,
                     torch::Tensor node2pin_list_,
                     torch::Tensor node2pin_list_end_,
                     torch::Tensor hyperedge_list_,
                     torch::Tensor hyperedge_list_end_,
                     torch::Tensor net_mask_,
                     int num_movable_nodes_,
                     float scale_factor_,
                     int microns_,
                     float wire_resistance_per_micron_,
                     float wire_capacitance_per_micron_);

    void commit_from(torch::Tensor x_, torch::Tensor y_);
    torch::Tensor get_curr_cposx();
    torch::Tensor get_curr_cposy();
    torch::Tensor get_curr_lposx();
    torch::Tensor get_curr_lposy();

public:
    /* node info */
    // for backup
    torch::Tensor node_lpos_init;
    torch::Tensor node_size;
    torch::Tensor pin_rel_lpos;

    torch::Tensor init_x;  // original pos (keep it const except committing)
    torch::Tensor init_y;  // original pos (keep it const except committing)
    torch::Tensor x;       // mutable/cached pos (current)
    torch::Tensor y;       // mutable/cached pos (current)
    torch::Tensor node_size_x;
    torch::Tensor node_size_y;

    /* pin info */
    torch::Tensor pin_offset_x;
    torch::Tensor pin_offset_y;

    // gputimer api tensors
    torch::Tensor at_prefix_pin;
    torch::Tensor at_prefix_arc;
    torch::Tensor at_prefix_attr;

    torch::Tensor flat_node2pin_start_map;
    torch::Tensor flat_node2pin_map;
    torch::Tensor pin2node_map;

    /* net info */
    torch::Tensor flat_net2pin_start_map;
    torch::Tensor flat_net2pin_map;
    torch::Tensor pin2net_map;
    torch::Tensor net_mask;

    /* chip info */
    int num_pins;
    int num_nets;
    int num_nodes;
    int num_movable_nodes;

    int num_threads;

public:
    float scale_factor;
    float wire_resistance_per_micron;
    float wire_capacitance_per_micron;
    int microns;

    // Timer model variables
    /// @param pinSlew                  Slew value on a pin
    /// @param pinLoad                  Load value on a pin
    /// @param pinRAT                   Required arrival time on a pin
    /// @param pinAT                    Arrival time on a pin
    /// @param pinImpulse               Impulse value on a pin
    /// @param pinRootDelay             Root delay value on a sink pin
public:
    torch::Tensor pinSlew;
    torch::Tensor pinLoad;
    torch::Tensor pinRAT;
    torch::Tensor pinAT;
    torch::Tensor pinImpulse;
    torch::Tensor pinRootDelay;

    // Timer RC Tree variables
    /// @param endpoints_id             Index of the endpoints
    /// @param arcDelay                 Delay value of an arc
    /// @param pinImpulse_ref           Reference impulse value of a accurate Timer
    /// @param pinLoad_ref              Reference load value of a accurate Timer
    /// @param pinLoad_ratio            Load ratio value of a accurate Timer
    /// @param pinRootDelay_ref         Reference root delay value of a accurate Timer
    /// @param pinRootDelay_ratio       Root delay ratio value of a accurate Timer
    /// @param pinRootDelay_compensation Root delay compensation value compared to a accurate Timer
public:
    // vector<float> arcDelay;
    torch::Tensor endpoints_id;
    torch::Tensor arcDelay;
    torch::Tensor pinImpulse_ref;
    torch::Tensor pinLoad_ref;
    torch::Tensor pinLoad_ratio;
    torch::Tensor pinRootDelay_ref;
    torch::Tensor pinRootDelay_ratio;
    torch::Tensor pinRootDelay_compensation;

    // Timer graph topology variables
    /// @param pin_forward_arc_list             List of forward arcs of a pin
    /// @param pin_forward_arc_list_end         Star & End index of the forward arcs lists
    /// @param pin_backward_arc_list            List of backward arcs of a pin
    /// @param pin_backward_arc_list_end        Star & End index of the backward arcs lists
    /// @param timing_arc_from_pin_id           From pin index of an arc
    /// @param timing_arc_to_pin_id             To pin index of an arc
    /// @param pin_num_fanin                    Number of fanin pins of a pin
    /// @param pin_fanout_list                  List of fanout pins of a pin
    /// @param pin_fanout_list_end              Star & End index of the fanout pins lists
public:
    torch::Tensor pin_forward_arc_list;
    torch::Tensor pin_forward_arc_list_end;
    torch::Tensor pin_backward_arc_list;
    torch::Tensor pin_backward_arc_list_end;
    torch::Tensor timing_arc_from_pin_id;
    torch::Tensor timing_arc_to_pin_id;
    torch::Tensor pin_num_fanin;
    torch::Tensor pin_fanout_list;
    torch::Tensor pin_fanout_list_end;

    // Timer timing liberty variables
    /// @param arc_types                        Types of an arc: 0/1
    /// @param timing_arc_id_map                Timing liberty index of an arc
    /// @param arc_id2test_id                   Timing test index of an arc: -1 for non-test arcs
    /// @param test_id2_arc_id                  Timing arc index of a test
public:
    torch::Tensor arc_types;
    torch::Tensor timing_arc_id_map;
    torch::Tensor arc_id2test_id;
    torch::Tensor test_id2_arc_id;
};

}  // namespace gt