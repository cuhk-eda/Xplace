#include <zlib.h>
#include <iostream>

#include "Liberty.h"
#include "Timing.h"
#include "common/db/Cell.h"
#include "common/db/Database.h"
#include "Lut.h"

using std::ifstream;

namespace gt {

LutTemplate* CellLib::get_lut_template(const std::string& name) {
    if (auto itr = lut_templates_.find(name); itr == lut_templates_.end()) {
        return nullptr;
    } else {
        return itr->second;
    }
}

std::optional<float> CellLib::extract_operating_conditions(token_iterator& itr, const token_iterator end) {
    std::optional<float> voltage;
    std::string operating_condition_name;
    if (itr = on_next_parentheses(itr, end, [&](auto& name) mutable { operating_condition_name = name; }); itr == end) {
        logger.info("can't find lut template name");
    }
    // Extract the lut template group
    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find lut template group brace '{'");
    }

    int stack = 1;
    while (stack && ++itr != end) {
        // variable 1
        if (*itr == "voltage") {  // Read the variable.

            if (++itr == end) {
                logger.info("volate error in operating_conditions template %s", operating_condition_name);
            }

            voltage = std::strtof(std::string(*itr).c_str(), nullptr);
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // undefined token TODO:
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find operating_conditions template group brace '}'");
    }

    return voltage;
}

LutTemplate* CellLib::extract_lut_template(token_iterator& itr, const token_iterator end) {
    LutTemplate* lt = new LutTemplate();

    if (itr = on_next_parentheses(itr, end, [&](auto& name) mutable { lt->name = name; }); itr == end) {
        logger.info("can't find lut template name");
    }

    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find lut template group brace '{'");
    }

    int stack = 1;
    while (stack && ++itr != end) {
        if (*itr == "variable_1") {
            if (++itr == end) {
                logger.info("variable_1 error in lut template %s", lt->name.c_str());
            }

            if (auto vitr = lut_vars.find(*itr); vitr != lut_vars.end()) {
                lt->variable1 = vitr->second;
            } else {
                logger.warning(
                    "unexpected lut template variable %.*s", static_cast<int>((*itr).length()), (*itr).data());
            }
        } else if (*itr == "variable_2") {
            if (++itr == end) {
                logger.info("variable_2 error in lut template %s", lt->name.c_str());
            }
            if (auto vitr = lut_vars.find(*itr); vitr != lut_vars.end()) {
                lt->variable2 = vitr->second;
            } else {
                logger.warning(
                    "unexpected lut template variable %.*s", static_cast<int>((*itr).length()), (*itr).data());
            }
        } else if (*itr == "index_1") {
            itr = on_next_parentheses(
                itr, end, [&](auto& str) { lt->indices1.push_back(std::strtof(str.data(), nullptr)); });
        } else if (*itr == "index_2") {
            itr = on_next_parentheses(
                itr, end, [&](auto& str) { lt->indices2.push_back(std::strtof(str.data(), nullptr)); });
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // undefined token TODO:
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find lut template brace '}'");
    }

    lut_templates_[lt->name] = lt;

    return lt;
}

Lut* CellLib::extract_lut(token_iterator& itr, const token_iterator end) {
    Lut* lut = new Lut();

    if (itr = on_next_parentheses(itr, end, [&](auto& name) mutable { lut->name = name; }); itr == end) {
        logger.info("can't find lut template name");
    }

    lut->lut_template = get_lut_template(lut->name);

    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("group brace '{' error in lut ", lut->name);
    }

    int stack = 1;
    size_t size1 = 1;
    size_t size2 = 1;
    while (stack && ++itr != end) {
        if (*itr == "index_1") {
            itr = on_next_parentheses(
                itr, end, [&](auto& v) mutable { lut->indices1.push_back(std::strtof(v.data(), nullptr)); });

            if (lut->indices1.size() == 0) {
                logger.info("syntax error in %s index_1", lut->name);
            }

            size1 = lut->indices1.size();
        } else if (*itr == "index_2") {
            itr = on_next_parentheses(
                itr, end, [&](auto& v) mutable { lut->indices2.push_back(std::strtof(v.data(), nullptr)); });

            if (lut->indices2.size() == 0) {
                logger.info("syntax error in %s index_2", lut->name);
            }

            size2 = lut->indices2.size();
        } else if (*itr == "values") {
            if (lut->indices1.empty()) {
                if (size1 != 1) {
                    logger.info("empty indices1 in non-scalar lut %s", lut->name);
                }
                lut->indices1.resize(size1);
            }

            if (lut->indices2.empty()) {
                if (size2 != 1) {
                    logger.info("empty indices2 in non-scalar lut %s", lut->name);
                }
                lut->indices2.resize(size2);
            }

            lut->table.resize(size1 * size2);

            int id{0};
            itr = on_next_parentheses(
                itr, end, [&](auto& v) mutable { lut->table[id++] = std::strtof(v.data(), nullptr); });
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // undefined token TODO:
        }
    }

    lut->set_ = true;

    if (stack != 0 || *itr != "}") {
        logger.info("can't find group brace '}' in lut ");
    }

    return lut;
}

TimingArc* CellLib::extractTimingArc(token_iterator& itr, const token_iterator end, LibertyPort* cell_port) {
    TimingArc* timing_arc = new TimingArc();
    timing_arc->liberty_port_ = cell_port;
    cell_port->timing_arcs_.push_back(timing_arc);

    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find group brace '{' in timing");
    }
    int stack = 1;
    while (stack && ++itr != end) {
        if (*itr == "cell_fall") {
            timing_arc->cell_delay_[1] = extract_lut(itr, end);
        } else if (*itr == "cell_rise") {
            timing_arc->cell_delay_[0] = extract_lut(itr, end);
        } else if (*itr == "fall_transition") {
            timing_arc->transition_[1] = extract_lut(itr, end);
        } else if (*itr == "rise_transition") {
            timing_arc->transition_[0] = extract_lut(itr, end);
        } else if (*itr == "fall_constraint") {
            timing_arc->constraint_[1] = extract_lut(itr, end);
        } else if (*itr == "rise_constraint") {
            timing_arc->constraint_[0] = extract_lut(itr, end);
        } else if (*itr == "timing_sense") {
            logger.infoif(++itr == end, "can't get the timing_sense in cellpin ");
            timing_arc->timing_sense_ = findTimingSense(string(*itr));
        } else if (*itr == "timing_type") {
            logger.infoif(++itr == end, "can't get the timing_type in cellpin ");
            timing_arc->timing_type_ = findTimingType(string(*itr));
        } else if (*itr == "sdf_cond") {
            logger.infoif(++itr == end, "can't get the sdf_cond in cellpin ");
            timing_arc->sdf_cond_ = *itr;
            timing_arc->is_cond_ = true;
        } else if (*itr == "related_pin") {
            logger.infoif(++itr == end, "can't get the related port ");
            timing_arc->related_port_name_ = *itr;
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // undefined token TODO:
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find group brace '}' in cell timing ");
    }

    return timing_arc;
}

LibertyPort* CellLib::extractLibertyPort(token_iterator& itr, const token_iterator end, LibertyCell* liberty_cell) {
    LibertyPort* cell_port = new LibertyPort();
    cell_port->cell_ = liberty_cell;

    on_next_parentheses(itr, end, [&](auto& name) mutable { cell_port->name = name; });
    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find group brace '{' in port");
    }

    int stack = 1;
    while (stack && ++itr != end) {
        if (*itr == "direction") {
            logger.infoif(++itr == end, "can't get direction in cell ", cell_port->name);
            cell_port->direction_ = findPortDirection(string(*itr));
        } else if (*itr == "capacitance") {
            logger.infoif(++itr == end, "can't get the capacitance in cellpin");
            cell_port->port_capacitance_[2] = std::strtof(itr->data(), nullptr);
        } else if (*itr == "fall_capacitance") {
            logger.infoif(++itr == end, "can't get fall_capacitance in cellpin");
            cell_port->port_capacitance_[1] = std::strtof(itr->data(), nullptr);
        } else if (*itr == "rise_capacitance") {
            logger.infoif(++itr == end, "can't get rise_capacitance in cellpin");
            cell_port->port_capacitance_[0] = std::strtof(itr->data(), nullptr);
        } else if (*itr == "max_capacitance") {
            logger.infoif(++itr == end, "can't get the max_capacitance in cellpin");
            cell_port->max_capacitance = std::strtof(itr->data(), nullptr);
        } else if (*itr == "min_capacitance") {
            logger.infoif(++itr == end, "can't get the min_capacitance in cellpin");
            cell_port->min_capacitance = std::strtof(itr->data(), nullptr);
        } else if (*itr == "max_transition") {
            logger.infoif(++itr == end, "can't get the max_transition in cellpin");
            cell_port->max_transition = std::strtof(itr->data(), nullptr);
        } else if (*itr == "min_transition") {
            logger.infoif(++itr == end, "can't get the min_transition in cellpin");
            cell_port->min_transition = std::strtof(itr->data(), nullptr);
        } else if (*itr == "fanout_load") {
            logger.infoif(++itr == end, "can't get fanout_load in cellpin");
            cell_port->fanout_load = std::strtof(itr->data(), nullptr);
        } else if (*itr == "max_fanout") {
            logger.infoif(++itr == end, "can't get max_fanout in cellpin");
            cell_port->max_fanout = std::strtof(itr->data(), nullptr);
        } else if (*itr == "min_fanout") {
            logger.infoif(++itr == end, "can't get min_fanout in cellpin");
            cell_port->min_fanout = std::strtof(itr->data(), nullptr);
        } else if (*itr == "clock") {
            logger.infoif(++itr == end, "can't get the clock status in cellpin");
            cell_port->is_clock_ = (*itr == "true") ? true : false;
        } else if (*itr == "timing") {
            TimingArc* timing_arc_ = extractTimingArc(itr, end, cell_port);
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // undefined token TODO:
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find group brace '}' in cell port");
    }

    return cell_port;
}

LibertyCell* CellLib::extractLibertyCell(token_iterator& itr, const token_iterator end) {
    LibertyCell* liberty_cell = new LibertyCell();

    on_next_parentheses(itr, end, [&](auto& name) mutable { liberty_cell->name = name; });
    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find group brace '{' in cell %s", liberty_cell->name);
    }

    int stack = 1;
    int stage = -1;
    while (stack && ++itr != end) {
        if (*itr == "cell_leakage_power") {
            logger.infoif(++itr == end, "can't get the cell_leakage_power ");
            liberty_cell->leakage_power_ = scale_factors["power"] * std::strtof(itr->data(), nullptr);
        }
        if (*itr == "leakage_power") {
            itr = std::find(itr, end, "{");
            int stack_1 = 1;
            while (stack_1 && ++itr != end) {
                if (*itr == "value") {
                    logger.infoif(++itr == end, "can't get value in cell %s", liberty_cell->name);
                    liberty_cell->leakage_powers_.push_back(std::strtof(itr->data(), nullptr));
                } else if (*itr == "}")
                    stack_1--;
                else if (*itr == "{")
                    stack_1++;
            }
        } else if (*itr == "area") {
            logger.infoif(++itr == end, "can't get area in cell %s", liberty_cell->name);
            liberty_cell->area_ = std::strtof(itr->data(), nullptr);
        } else if (*itr == "pin") {
            logger.infoif(++itr == end, "can't get port in cell %s", liberty_cell->name);
            LibertyPort* cell_port_ = extractLibertyPort(itr, end, liberty_cell);
            liberty_cell->ports_.push_back(cell_port_);
        } else if (*itr == "bundle") {
            LibertyPort* cell_port_bundle = new LibertyPort();
            liberty_cell->ports_.push_back(cell_port_bundle);
            cell_port_bundle->cell_ = liberty_cell;
            cell_port_bundle->is_bundle_ = true;

            on_next_parentheses(itr, end, [&](auto& name) mutable { cell_port_bundle->name = name; });
            itr = std::find(itr, end, "{");

            int stack_1 = 1;
            while (stack_1 && ++itr != end) {
                if (*itr == "direction") {
                    logger.infoif(++itr == end, "can't get direction in cell %s", liberty_cell->name);
                    cell_port_bundle->direction_ = findPortDirection(string(*itr));
                } else if (*itr == "pin") {
                    LibertyPort* cell_port_ = extractLibertyPort(itr, end, liberty_cell);
                    cell_port_bundle->member_ports_.push_back(cell_port_);
                } else if (*itr == "}")
                    stack_1--;
                else if (*itr == "{")
                    stack_1++;
            }
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // undefined token TODO:
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find group brace '}' in cellpin ");
    }
    return liberty_cell;
}

void CellLib::read(const std::string& file) {
    // process .gz file with zlib
    std::vector<char> buffer;
    if (file.substr(file.find_last_of(".") + 1) == "gz") {
        logger.info("reading gzip celllib %s ...", file.c_str());
        gzFile fs = gzopen(file.c_str(), "rb");
        if (!fs) {
            logger.error("cannot open verilog file: %s", file.c_str());
        }
        char buf[1024];
        int len = 0;
        while ((len = gzread(fs, buf, 1024)) > 0) {
            buffer.insert(buffer.end(), buf, buf + len);
        }
        gzclose(fs);
        buffer.push_back(0);
    } else {
        ifstream fs(file.c_str(), std::ios::ate);
        if (!fs.good()) {
            logger.error("cannot open liberty file: %s", file.c_str());
        }
        logger.info("reading celllib %s ...", file.c_str());

        size_t fsize = fs.tellg();
        fs.seekg(0, std::ios::beg);
        buffer.resize(fsize + 1);
        fs.read(buffer.data(), fsize);
        buffer[fsize] = 0;
    }

    // get tokens
    std::vector<std::string_view> tokens;
    tokens.reserve(buffer.size() / sizeof(std::string));

    uncomment(buffer);
    tokenize(buffer, tokens);

    // Set up the iterator
    auto itr = tokens.begin();
    auto end = tokens.end();

    // Read the library name.
    if (itr = std::find(itr, end, "library"); itr == end) {
        logger.error("can't find keyword %s", "library");
    }

    if (itr = on_next_parentheses(itr, end, [&](auto& str) mutable { name = str; }); itr == end) {
        logger.info("can't find library name");
    }

    if (itr = std::find(itr, tokens.end(), "{"); itr == tokens.end()) {
        logger.info("can't find library group symbol '{'");
    }

    int stack = 1;

    while (stack && ++itr != end) {
        if (*itr == "lu_table_template") {
            auto lut = extract_lut_template(itr, end);
        } else if (*itr == "power_lut_template") {
            auto lut = extract_lut_template(itr, end);
        } else if (*itr == "delay_model") {
            logger.infoif(++itr == end, "syntax error in delay_model");
            delay_model = findDelayModel(string(*itr));
        } else if (*itr == "default_cell_leakage_power" || *itr == "default_inout_pin_cap" ||
                   *itr == "default_input_pin_cap" || *itr == "default_output_pin_cap" ||
                   *itr == "default_fanout_load" || *itr == "default_max_fanout" || *itr == "default_max_transition") {
            logger.infoif(++itr == end, "syntax error");
            default_values[std::string(*itr)] = std::strtof(itr->data(), nullptr);
        } else if (*itr == "operating_conditions") {
            logger.infoif(++itr == end, "syntax error");
            default_values["voltage"] = extract_operating_conditions(itr, end);
        } else if (*itr == "time_unit") {
            logger.infoif(++itr == end, "syntax error");
            time_unit_ = make_time_unit(*itr);
        } else if (*itr == "voltage_unit") {
            logger.infoif(++itr == end, "syntax error");
            voltage_unit_ = make_voltage_unit(*itr);
        } else if (*itr == "current_unit") {
            logger.infoif(++itr == end, "syntax error");
            current_unit_ = make_current_unit(*itr);
        } else if (*itr == "pulling_resistance_unit") {
            logger.infoif(++itr == end, "syntax error");
            resistance_unit_ = make_resistance_unit(*itr);
        } else if (*itr == "capacitive_load_unit") {
            string unit;
            on_next_parentheses(itr, end, [&](auto& str) mutable { unit += str; });
            capacitance_unit_ = make_capacitance_unit(unit);
        } else if (*itr == "leakage_power_unit") {
            logger.infoif(++itr == end, "syntax error");
            auto current_power_unit_ = make_power_unit(*itr);
            if (!power_unit_) power_unit_ = current_power_unit_;
            scale_factors["power"] = *current_power_unit_ / *power_unit_;
        } else if (*itr == "cell") {
            LibertyCell* libterty_cell = extractLibertyCell(itr, end);
            lib_cells_[libterty_cell->name] = libterty_cell;
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // undefined token TODO:
        }
    }
}

void CellLib::finish_port_read(LibertyPort* liberty_port) {
    for (TimingArc* timing_arc : liberty_port->timing_arcs_) {
        if (timing_arc->related_port_name_.empty()) {
            logger.warning("timing arc %s.%s.%s has no related pin",
                           liberty_port->cell_->name.c_str(),
                           liberty_port->name.c_str(),
                           timing_arc->timing_type_);
            continue;
        }
        if (auto related_port = liberty_port->cell_->get_port(timing_arc->related_port_name_);
            related_port == -1) {
            logger.warning("timing arc %s.%s.%s has no related pin",
                           liberty_port->cell_->name.c_str(),
                           liberty_port->name.c_str(),
                           timing_arc->timing_type_);
        } else {
            timing_arc->from_port_ = liberty_port->cell_->ports_[related_port];;
        }
        timing_arc->to_port_ = timing_arc->liberty_port_;
    }

    for (TimingArc* timing_arc : liberty_port->timing_arcs_) {
        timing_arc->encode_str_ = timing_arc->encode_arc();
        if (liberty_port->timing_arcs_map_.find(timing_arc->encode_str_) != liberty_port->timing_arcs_map_.end()) {
            TimingArc* old_timing_arc = liberty_port->timing_arcs_map_[timing_arc->encode_str_];
            if (!timing_arc->is_cond_) {
                liberty_port->timing_arcs_map_[timing_arc->encode_str_] = timing_arc;
            }
        } else
            liberty_port->timing_arcs_map_[timing_arc->encode_str_] = timing_arc;
    }
}

void CellLib::finish_read() {
    for (auto [name, liberty_cell] : lib_cells_) {
        db::CellType* lef_cell_type = rawdb->getCellType(name);
        if (lef_cell_type == nullptr) {
            logger.warning("cell %s not found in lef", name.c_str());
            continue;
        } else {
            lef_cell_type->liberty_cell = liberty_cell;
            liberty_cell->cell_type_ = lef_cell_type;
        }
        // sort port by name
        std::sort(liberty_cell->ports_.begin(),
                  liberty_cell->ports_.end(),
                  [](const LibertyPort* a, const LibertyPort* b) { return a->name < b->name; });
        
        for (int i = 0; i < liberty_cell->ports_.size(); i++) {
            liberty_cell->ports_map_[liberty_cell->ports_[i]->name] = i;
        }

        for (auto port : liberty_cell->ports_) {
            if (port->is_clock_) liberty_cell->is_seq_ = true;
            if (port->is_bundle_) {
                for (auto member_port : port->member_ports_) {
                    finish_port_read(member_port);
                }
            } else
                finish_port_read(port);
        }

        for (auto port : liberty_cell->ports_) {
            LibertyPort* non_bundle_port;
            if (port->is_bundle_) {
                non_bundle_port = port->member_ports_[0];
            } else
                non_bundle_port = port;
            for (auto kvp : non_bundle_port->timing_arcs_map_) {
                // string encode_str = kvp.first;
                // std::cout << encode_str << std::endl;
                TimingArc* timing_arc = kvp.second;
                port->timing_arcs_non_cond_non_bundle_.push_back(timing_arc);
            }
        }
    }
}

};  // namespace gt
