
#pragma once

#include "common/lib/sdc/json.hpp"
#include "common/lib/tokenizer.h"
#include "mask.h"
#include "object.h"

namespace gt::sdc {

using Json = nlohmann::json;

std::unique_ptr<char*, std::function<void(char**)>> c_args(const std::vector<std::string>&);

// Set a fixed transition on input or inout ports
struct SetUnits {
    inline static constexpr auto command = "set_units";

    std::optional<std::string> time;
    std::optional<std::string> capacitance;
    std::optional<std::string> resistance;
    std::optional<std::string> current;
    std::optional<std::string> voltage;

    SetUnits() = default;
    SetUnits(const Json&);
};

struct SetInputDelay {
    inline static constexpr auto command = "set_input_delay";

    std::string clock;
    std::optional<std::byte> clock_fall;
    std::optional<std::byte> level_sensitive;
    std::optional<std::byte> add_delay;
    std::optional<std::byte> network_latency_included;
    std::optional<std::byte> source_latency_included;
    std::optional<std::byte> min;
    std::optional<std::byte> max;
    std::optional<std::byte> rise;
    std::optional<std::byte> fall;
    std::optional<float> delay_value;
    std::optional<Object> port_pin_list;

    SetInputDelay() = default;
    SetInputDelay(const Json&);
};

struct SetDrivingCell {
    inline static constexpr auto command = "set_driving_cell";

    std::string clock;
    std::optional<std::byte> min;
    std::optional<std::byte> max;
    std::optional<std::byte> rise;
    std::optional<std::byte> fall;
    std::optional<std::byte> clock_fall;
    std::array<std::optional<float>, 2> transitions;
    std::optional<Object> port_list;

    SetDrivingCell() = default;
    SetDrivingCell(const Json&);
};

// Set a fixed transition on input or inout ports
struct SetInputTransition {
    inline static constexpr auto command = "set_input_transition";

    std::string clock;
    std::optional<std::byte> min;
    std::optional<std::byte> max;
    std::optional<std::byte> rise;
    std::optional<std::byte> fall;
    std::optional<std::byte> clock_fall;
    std::optional<float> transition;
    std::optional<Object> port_list;

    SetInputTransition() = default;
    SetInputTransition(const Json&);
};

struct SetOutputDelay {
    inline static constexpr auto command = "set_output_delay";

    std::string clock;
    std::optional<std::byte> clock_fall;
    std::optional<std::byte> level_sensitive;
    std::optional<std::byte> rise;
    std::optional<std::byte> fall;
    std::optional<std::byte> max;
    std::optional<std::byte> min;
    std::optional<std::byte> add_delay;
    std::optional<std::byte> network_latency_included;
    std::optional<std::byte> source_latency_included;
    std::optional<float> delay_value;
    std::optional<Object> port_pin_list;

    SetOutputDelay() = default;
    SetOutputDelay(const Json&);
};

struct SetLoad {
    inline static constexpr auto command = "set_load";

    std::optional<std::byte> min;
    std::optional<std::byte> max;
    std::optional<std::byte> subtract_pin_load;
    std::optional<std::byte> pin_load;
    std::optional<std::byte> wire_load;
    std::optional<float> value;
    std::optional<Object> objects;

    SetLoad() = default;
    SetLoad(const Json&);
};

struct CreateClock {
    inline static constexpr auto command = "create_clock";

    std::optional<float> period;
    std::optional<std::byte> add;
    std::string name;
    std::string comment;
    std::optional<std::array<float, MAX_TRAN>> waveform;
    std::optional<Object> port_pin_list;

    CreateClock() = default;
    CreateClock(const Json&);
};

struct SetClockUncertainty {
    inline static constexpr auto command = "set_clock_uncertainty";

    std::optional<float> uncertainty;
    std::optional<Object> object_list;

    SetClockUncertainty() = default;
    SetClockUncertainty(const Json&);
};

using Command = std::variant<SetUnits, SetInputDelay, SetDrivingCell, SetInputTransition, SetOutputDelay, SetLoad, CreateClock>;

struct SDC {
    std::vector<Command> commands;
    void read(const std::filesystem::path&);
};

};  // namespace gt::sdc
