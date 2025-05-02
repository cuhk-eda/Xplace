#include "sdc.h"

namespace gt::sdc {

std::unique_ptr<char*, std::function<void(char**)>> c_args(const std::vector<std::string>& args) {
    std::unique_ptr<char*, std::function<void(char**)>> ptr(new char*[args.size() + 1], [n = args.size()](char** ptr) {
        for (size_t i = 0; i <= n; ++i) {
            delete[] ptr[i];
        }
        delete[] ptr;
    });

    for (size_t i = 0; i < args.size(); ++i) {
        ptr.get()[i] = new char[args[i].length() + 1];
        std::strcpy(ptr.get()[i], args[i].c_str());
    }

    ptr.get()[args.size()] = nullptr;

    return ptr;
}

void SDC::read(const std::filesystem::path& path) {
    logger.infoif(!std::filesystem::exists(path), "sdc file %s doesn't exist", path.c_str());
    logger.info("loading sdc %s", path.c_str());

    auto sdc_path = std::filesystem::absolute(path);
    auto sdc_json = sdc_path;
    sdc_json.replace_extension(".json");

    if (auto cpid = ::fork(); cpid == -1) {
        logger.error("can't fork sdc reader");
        return;
    }
    // child
    else if (cpid == 0) {
#define TCLSH_PATH "/usr/bin/tclsh"
        auto sdc_home = std::filesystem::current_path() / "cpp_to_py" / "common" / "lib" / "sdc";
        if (::chdir(sdc_home.c_str()) == -1) {
            logger.error("can't enter into %s", sdc_home.c_str());
            ::exit(EXIT_FAILURE);
        }

        std::vector<std::string> args{TCLSH_PATH, "sdc.tcl", sdc_path.c_str(), sdc_json.c_str()};

        ::execvp(args[0].c_str(), c_args(args).get());
        logger.error("exec failed: %s", strerror(errno));
        ::exit(EXIT_FAILURE);
        logger.warning("sdc reader not implemented yet");
    }
    // parent
    else {
        int s;

        do {
            int w = ::waitpid(cpid, &s, WUNTRACED | WCONTINUED);

            logger.errorif(w == -1, "failed to reap sdc reader");

            if (WIFEXITED(s)) {
                logger.errorif(WEXITSTATUS(s) != EXIT_SUCCESS, "sdc reader exited with failure");
            } else if (WIFSIGNALED(s)) {
                logger.error("sdc reader killed by signal %s", WTERMSIG(s));
                return;
            } else if (WIFSTOPPED(s)) {
                logger.info("sdc reader stopped");
            } else if (WIFCONTINUED(s)) {
                logger.info("sdc reader continued");
            }
        } while (!WIFEXITED(s) && !WIFSIGNALED(s));

        std::ifstream ifs(sdc_json);

        logger.infoif(!ifs, "failed to open %s", sdc_json);

        Json json;
        ifs >> json;

        for (const auto& j : json) {
            if (const auto& c = j["command"]; c == "set_input_delay") {
                commands.emplace_back(std::in_place_type_t<SetInputDelay>{}, j);
            } else if (c == "set_driving_cell") {
                commands.emplace_back(std::in_place_type_t<SetDrivingCell>{}, j);
            } else if (c == "set_input_transition") {
                commands.emplace_back(std::in_place_type_t<SetInputTransition>{}, j);
            } else if (c == "set_output_delay") {
                commands.emplace_back(std::in_place_type_t<SetOutputDelay>{}, j);
            } else if (c == "set_load") {
                commands.emplace_back(std::in_place_type_t<SetLoad>{}, j);
            } else if (c == "create_clock") {
                commands.emplace_back(std::in_place_type_t<CreateClock>{}, j);
            } else if (c == "set_units") {
                commands.emplace_back(std::in_place_type_t<SetUnits>{}, j);
            } else {
                logger.error("sdc command %s not supported yet", c);
            }
        }

        try {
            std::filesystem::remove(sdc_json);
        } catch (const std::exception& e) {
            logger.warning("can't remove %s: %s", sdc_json, e.what());
        }
    }
}

// Constructor
SetUnits::SetUnits(const Json& json) {
    for (auto itr = json.begin(); itr != json.end(); ++itr) {
        auto& key = itr.key();
        if (key == "-time") {
            time = unquoted(itr.value());
        } else if (key == "-capacitance") {
            capacitance = unquoted(itr.value());
        } else if (key == "-voltage") {
            voltage = unquoted(itr.value());
        } else if (key == "-current") {
            current = unquoted(itr.value());
        } else if (key == "-resistance") {
            resistance = unquoted(itr.value());
        } else if (key == "command") {
            logger.errorif(itr.value() != command, "wrong command field: %s", itr.value());
        } else {
            logger.error("%s: %s not supported", command, std::quoted(key));
        }
    }
}

// Constructor
SetInputDelay::SetInputDelay(const Json& json) {
    for (auto itr = json.begin(); itr != json.end(); ++itr) {
        if (auto& key = itr.key(); key == "-clock") {
            clock = itr.value();
        } else if (key == "-clock_fall") {
            clock_fall.emplace();
        } else if (key == "-level_sensitive") {
            level_sensitive.emplace();
        } else if (key == "-rise") {
            rise.emplace();
        } else if (key == "-fall") {
            fall.emplace();
        } else if (key == "-min") {
            min.emplace();
        } else if (key == "-max") {
            max.emplace();
        } else if (key == "-add_delay") {
            add_delay.emplace();
        } else if (key == "-network_latency_included") {
            network_latency_included.emplace();
        } else if (key == "-source_latency_included") {
            source_latency_included.emplace();
        } else if (key == "delay_value") {
            delay_value = std::stof(unquoted(itr.value()));
        } else if (key == "port_pin_list") {
            port_pin_list = parse_port(unquoted(itr.value()));
        } else if (key == "command") {
            logger.errorif(itr.value() != command, "wrong command field: %s", itr.value());
        } else {
            logger.error("%s: %s not supported", command, std::quoted(key));
        }
    }
}

// Constructor
SetDrivingCell::SetDrivingCell(const Json& json) {
    for (auto itr = json.begin(); itr != json.end(); ++itr) {
        if (const auto& key = itr.key(); key == "-clock") {
            clock = itr.value();
        } else if (key == "-clock_fall") {
            clock_fall.emplace();
        } else if (key == "-lib_cell") {
            // TODO: do nothing
        } else if (key == "-pin") {
            // TODO: do nothing
        } else if (key == "-min") {
            min.emplace();
        } else if (key == "-max") {
            max.emplace();
        } else if (key == "-input_transition_fall") {
            fall.emplace();
            transitions[1] = std::stof(unquoted(itr.value()));
        } else if (key == "-input_transition_rise") {
            rise.emplace();
            transitions[0] = std::stof(unquoted(itr.value()));
        } else if (key == "port_list") {
            port_list = parse_port(unquoted(itr.value()));
        } else if (key == "command") {
            logger.errorif(itr.value() != command, "wrong command field: %s", itr.value());
        } else {
            logger.error("%s: %s not supported", command, std::quoted(key));
        }
    }
}

// Constructor
SetInputTransition::SetInputTransition(const Json& json) {
    for (auto itr = json.begin(); itr != json.end(); ++itr) {
        if (const auto& key = itr.key(); key == "-clock") {
            clock = itr.value();
        } else if (key == "-clock_fall") {
            clock_fall.emplace();
        } else if (key == "-rise") {
            rise.emplace();
        } else if (key == "-fall") {
            fall.emplace();
        } else if (key == "-min") {
            min.emplace();
        } else if (key == "-max") {
            max.emplace();
        } else if (key == "transition") {
            transition = std::stof(unquoted(itr.value()));
        } else if (key == "port_list") {
            port_list = parse_port(unquoted(itr.value()));
        } else if (key == "command") {
            logger.errorif(itr.value() != command, "wrong command field: %s", itr.value());
        } else {
            logger.error("%s: %s not supported", command, std::quoted(key));
        }
    }
}

// Constructor
SetOutputDelay::SetOutputDelay(const Json& json) {
    for (auto itr = json.begin(); itr != json.end(); ++itr) {
        if (auto& key = itr.key(); key == "-clock") {
            clock = itr.value();
        } else if (key == "-clock_fall") {
            clock_fall.emplace();
        } else if (key == "-level_sensitive") {
            level_sensitive.emplace();
        } else if (key == "-rise") {
            rise.emplace();
        } else if (key == "-fall") {
            fall.emplace();
        } else if (key == "-min") {
            min.emplace();
        } else if (key == "-max") {
            max.emplace();
        } else if (key == "-add_delay") {
            add_delay.emplace();
        } else if (key == "-network_latency_included") {
            network_latency_included.emplace();
        } else if (key == "-source_latency_included") {
            source_latency_included.emplace();
        } else if (key == "delay_value") {
            delay_value = std::stof(unquoted(itr.value()));
        } else if (key == "port_pin_list") {
            port_pin_list = parse_port(unquoted(itr.value()));
        } else if (key == "command") {
            logger.errorif(itr.value() != command, "wrong command field: %s", itr.value());
        } else {
            logger.error("%s: %s not supported", command, std::quoted(key));
        }
    }
}

// Constructor
SetLoad::SetLoad(const Json& json) {
    for (auto itr = json.begin(); itr != json.end(); ++itr) {
        if (const auto& key = itr.key(); key == "-min") {
            min.emplace();
        } else if (key == "-max") {
            max.emplace();
        } else if (key == "-subtract_pin_load") {
            subtract_pin_load.emplace();
        } else if (key == "-pin_load") {
            pin_load.emplace();
        } else if (key == "-wire_load") {
            wire_load.emplace();
        } else if (key == "objects") {
            objects = parse_port(unquoted(itr.value()));
        } else if (key == "value") {
            value = std::stof(unquoted(itr.value()));
        } else if (key == "command") {
            logger.errorif(itr.value() != command, "wrong command field: %s", itr.value());
        } else {
            logger.error("%s: %s not supported", command, std::quoted(key));
        }
    }
}

// Constructor
CreateClock::CreateClock(const Json& json) {
    for (auto itr = json.begin(); itr != json.end(); ++itr) {
        if (auto& key = itr.key(); key == "-period") {
            period = std::stof(unquoted(itr.value()));
        } else if (key == "-add") {
            add.emplace();
        } else if (key == "-comment") {
            comment = itr.value();
        } else if (key == "-name") {
            name = itr.value();
        } else if (key == "-waveform") {
            // TODO
        } else if (key == "port_pin_list") {
            port_pin_list = parse_port(unquoted(itr.value()));
        } else if (key == "command") {
            logger.errorif(itr.value() != command, "wrong command field: %s", itr.value());
        } else {
            logger.error("%s: %s not supported", command, std::quoted(key));
        }
    }
}

// Constructor
SetClockUncertainty::SetClockUncertainty(const Json& json) {
    for (auto itr = json.begin(); itr != json.end(); ++itr) {
        if (auto& key = itr.key(); key == "uncertainty") {
        } else if (key == "object_list") {
            object_list = parse_port(unquoted(itr.value()));
        } else if (key == "command") {
            logger.errorif(itr.value() != command, "wrong command field: %s", itr.value());
        } else {
            logger.error("%s: %s not supported", command, std::quoted(key));
        }
    }
}

};  // namespace gt::sdc
