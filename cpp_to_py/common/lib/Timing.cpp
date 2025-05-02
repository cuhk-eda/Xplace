#include "Timing.h"
#include "Lut.h"
#include "Liberty.h"

namespace gt {

TimingArc::TimingArc() {
    for (int i = 0; i < MAX_TRAN; i++) {
        cell_delay_[i] = new Lut();
        transition_[i] = new Lut();
        constraint_[i] = new Lut();
    }
}

bool TimingArc::is_constraint() const {
    switch (timing_type_) {
        case TimingType::removal_rising:
        case TimingType::removal_falling:
        case TimingType::recovery_rising:
        case TimingType::recovery_falling:
        case TimingType::setup_rising:
        case TimingType::setup_falling:
        case TimingType::hold_rising:
        case TimingType::hold_falling:
        case TimingType::non_seq_setup_rising:
        case TimingType::non_seq_setup_falling:
        case TimingType::non_seq_hold_rising:
        case TimingType::non_seq_hold_falling:
            return true;
            break;
        default:
            return false;
            break;
    }
}

bool TimingArc::is_min_constraint() const {
    switch (timing_type_) {
        case TimingType::hold_rising:
        case TimingType::hold_falling:
        case TimingType::non_seq_hold_rising:
        case TimingType::non_seq_hold_falling:
        case TimingType::removal_rising:
        case TimingType::removal_falling:
            return true;
            break;

        default:
            return false;
            break;
    }
}

bool TimingArc::is_max_constraint() const {
    switch (timing_type_) {
        case TimingType::setup_rising:
        case TimingType::setup_falling:
        case TimingType::non_seq_setup_rising:
        case TimingType::non_seq_setup_falling:
        case TimingType::recovery_rising:
        case TimingType::recovery_falling:
            return true;
            break;

        default:
            return false;
            break;
    }
}

bool TimingArc::is_falling_edge_triggered() const {
    switch (timing_type_) {
        case TimingType::setup_falling:
        case TimingType::hold_falling:
        case TimingType::removal_falling:
        case TimingType::recovery_falling:
        case TimingType::falling_edge:
            return true;
            break;

        default:
            return false;
            break;
    };
}

bool TimingArc::is_rising_edge_triggered() const {
    switch (timing_type_) {
        case TimingType::setup_rising:
        case TimingType::hold_rising:
        case TimingType::removal_rising:
        case TimingType::recovery_rising:
        case TimingType::rising_edge:
            return true;
            break;

        default:
            return false;
            break;
    };
}

// encode TimingType TimingSense from_port_name to_port_name to remove duplicate
string TimingArc::encode_arc() {
    int max_type_shift = std::ceil(std::log2(static_cast<int>(TimingType::unknown)));
    int encode = (static_cast<int>(timing_sense_) << max_type_shift) | static_cast<int>(timing_type_);
    string encode_str = std::to_string(encode);
    encode_str +=  "_" + from_port_->name + "_" + to_port_->name; 
    return encode_str;
}


EnumNameMap<TimingSense> timing_sense_name_map = {{TimingSense::non_unate, "non_unate"},
                                                  {TimingSense::positive_unate, "positive_unate"},
                                                  {TimingSense::negative_unate, "negative_unate"},
                                                  {TimingSense::unknown, "unknown"}};

TimingSense findTimingSense(const std::string sense_name) {
    return timing_sense_name_map.find(sense_name, TimingSense::unknown);
}

EnumNameMap<TimingType> timing_type_name_map = {{TimingType::clear, "clear"},
                                                {TimingType::combinational, "combinational"},
                                                {TimingType::combinational_fall, "combinational_fall"},
                                                {TimingType::combinational_rise, "combinational_rise"},
                                                {TimingType::falling_edge, "falling_edge"},
                                                {TimingType::hold_falling, "hold_falling"},
                                                {TimingType::hold_rising, "hold_rising"},
                                                {TimingType::min_pulse_width, "min_pulse_width"},
                                                {TimingType::minimum_period, "minimum_period"},
                                                {TimingType::nochange_high_high, "nochange_high_high"},
                                                {TimingType::nochange_high_low, "nochange_high_low"},
                                                {TimingType::nochange_low_high, "nochange_low_high"},
                                                {TimingType::nochange_low_low, "nochange_low_low"},
                                                {TimingType::non_seq_hold_falling, "non_seq_hold_falling"},
                                                {TimingType::non_seq_hold_rising, "non_seq_hold_rising"},
                                                {TimingType::non_seq_setup_falling, "non_seq_setup_falling"},
                                                {TimingType::non_seq_setup_rising, "non_seq_setup_rising"},
                                                {TimingType::preset, "preset"},
                                                {TimingType::recovery_falling, "recovery_falling"},
                                                {TimingType::recovery_rising, "recovery_rising"},
                                                {TimingType::removal_falling, "removal_falling"},
                                                {TimingType::removal_rising, "removal_rising"},
                                                {TimingType::retaining_time, "retaining_time"},
                                                {TimingType::rising_edge, "rising_edge"},
                                                {TimingType::setup_falling, "setup_falling"},
                                                {TimingType::setup_rising, "setup_rising"},
                                                {TimingType::skew_falling, "skew_falling"},
                                                {TimingType::skew_rising, "skew_rising"},
                                                {TimingType::three_state_disable, "three_state_disable"},
                                                {TimingType::three_state_disable_fall, "three_state_disable_fall"},
                                                {TimingType::three_state_disable_rise, "three_state_disable_rise"},
                                                {TimingType::three_state_enable, "three_state_enable"},
                                                {TimingType::three_state_enable_fall, "three_state_enable_fall"},
                                                {TimingType::three_state_enable_rise, "three_state_enable_rise"},
                                                {TimingType::min_clock_tree_path, "min_clock_tree_path"},
                                                {TimingType::max_clock_tree_path, "max_clock_tree_path"},
                                                {TimingType::unknown, "unknown"}};

TimingType findTimingType(const std::string type_name) {
    return timing_type_name_map.find(type_name, TimingType::unknown);
}

};  // namespace gt
