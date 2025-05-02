#include "Lut.h"

namespace gt {

bool is_time_lut_var(LutVar v) {
    switch (v) {
        case LutVar::INPUT_NET_TRANSITION:
        case LutVar::CONSTRAINED_PIN_TRANSITION:
        case LutVar::RELATED_PIN_TRANSITION:
        case LutVar::INPUT_TRANSITION_TIME:
            return true;
            break;

        default:
            return false;
            break;
    }
}

bool is_capacitance_lut_var(LutVar v) {
    switch (v) {
        case LutVar::TOTAL_OUTPUT_NET_CAPACITANCE:
            return true;
            break;

        default:
            return false;
            break;
    }
}

};  // namespace gt
