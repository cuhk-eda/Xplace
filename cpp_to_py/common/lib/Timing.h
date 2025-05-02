#pragma once
#include <string>

#include "EnumNameMap.h"
#include "Helper.h"
using std::string;

namespace gt {

class LibertyCell;
class LibertyPort;
class TimingArc;
struct LutTemplate;
class Lut;

enum class TimingSense { non_unate, positive_unate, negative_unate, unknown };

enum class TimingType {
    clear,
    combinational,
    combinational_fall,
    combinational_rise,
    falling_edge,
    hold_falling,
    hold_rising,
    min_pulse_width,
    minimum_period,
    nochange_high_high,
    nochange_high_low,
    nochange_low_high,
    nochange_low_low,
    non_seq_hold_falling,
    non_seq_hold_rising,
    non_seq_setup_falling,
    non_seq_setup_rising,
    preset,
    recovery_falling,
    recovery_rising,
    removal_falling,
    removal_rising,
    retaining_time,
    rising_edge,
    setup_falling,
    setup_rising,
    skew_falling,
    skew_rising,
    three_state_disable,
    three_state_disable_fall,
    three_state_disable_rise,
    three_state_enable,
    three_state_enable_fall,
    three_state_enable_rise,
    min_clock_tree_path,
    max_clock_tree_path,
    unknown
};

TimingSense findTimingSense(const std::string sense_name);
TimingType findTimingType(const std::string type_name);

class TimingArc {
public:
    TimingArc();
    
    // TimingArc(TimingArcSet* set, LibertyPort* from, LibertyPort* to, Tran irf, Tran orf);
    int id = -1;
    string encode_str_;

    LibertyCell* cell_;
    LibertyPort* liberty_port_;

    TimingType timing_type_ = TimingType::unknown;
    TimingSense timing_sense_ = TimingSense::unknown;

    string related_port_name_;
    LibertyPort* from_port_;
    LibertyPort* to_port_;

    string sdf_cond_;
    bool is_cond_ = false;
    // TimingArcSet* timing_arc_set_;

    Lut* cell_delay_[MAX_TRAN];
    Lut* transition_[MAX_TRAN];
    Lut* constraint_[MAX_TRAN];

    string encode_arc();
    bool is_constraint() const;
    bool is_min_constraint() const;
    bool is_max_constraint() const;
    bool is_rising_edge_triggered() const;
    bool is_falling_edge_triggered() const;
};

};  // namespace gt
