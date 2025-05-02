
#pragma once
#include "common/common.h"
#include "common/lib/Helper.h"

namespace gt {

enum class LutVar {
    TOTAL_OUTPUT_NET_CAPACITANCE = 0,
    INPUT_NET_TRANSITION = 1,
    CONSTRAINED_PIN_TRANSITION = 2,
    RELATED_PIN_TRANSITION = 3,
    INPUT_TRANSITION_TIME = 4,
};

inline const std::unordered_map<std::string_view, LutVar> lut_vars{
    {"total_output_net_capacitance", LutVar::TOTAL_OUTPUT_NET_CAPACITANCE},
    {"input_net_transition", LutVar::INPUT_NET_TRANSITION},
    {"constrained_pin_transition", LutVar::CONSTRAINED_PIN_TRANSITION},
    {"related_pin_transition", LutVar::RELATED_PIN_TRANSITION},
    {"input_transition_timing", LutVar::INPUT_TRANSITION_TIME},
    {"input_transition_time", LutVar::INPUT_TRANSITION_TIME}};

bool is_time_lut_var(LutVar);
bool is_capacitance_lut_var(LutVar);

struct LutTemplate {
    LutTemplate() = default;

    std::string name;

    std::optional<LutVar> variable1;
    std::optional<LutVar> variable2;

    std::vector<float> indices1;
    std::vector<float> indices2;

    int id = -1;
};

class Lut {
public:
    Lut() = default;

    std::string name;

    std::vector<float> indices1;
    std::vector<float> indices2;
    std::vector<float> table;

    const LutTemplate* lut_template{nullptr};

    bool set_ = false;
};

};  // namespace gt
