

#include "GTDatabase.h"

#include "common/common.h"
#include "common/db/Cell.h"
#include "common/db/Database.h"
#include "common/db/Pin.h"
#include "common/lib/Liberty.h"
#include "common/lib/Timing.h"
#include "common/lib/sdc/sdc.h"
#include "io_parser/gp/GPDatabase.h"

namespace gt {

bool GTDatabase::is_redundant_timing(const TimingArc* timing_arc, Split el) {
    if (timing_arc->from_port_->name == timing_arc->to_port_->name) return true;
    if (timing_arc->related_port_name_.empty()) return true;
    if (timing_arc->timing_type_ == TimingType::non_seq_setup_rising || timing_arc->timing_type_ == TimingType::non_seq_setup_falling || timing_arc->timing_type_ == TimingType::non_seq_hold_rising ||
        timing_arc->timing_type_ == TimingType::non_seq_hold_falling)
        return true;
    switch (el) {
        case MIN:
            if (timing_arc->is_max_constraint()) {
                return true;
            }
            break;
        case MAX:
            if (timing_arc->is_min_constraint()) {
                return true;
            }
            break;
    }
    return false;
}

GTDatabase::GTDatabase(shared_ptr<db::Database> rawdb_, shared_ptr<gp::GPDatabase> gpdb_, shared_ptr<TimingTorchRawDB> timing_raw_db_) : rawdb(*rawdb_), gpdb(*gpdb_), timing_raw_db(*timing_raw_db_) {
    cell_libs_[MIN] = rawdb.cell_libs_[MIN];
    cell_libs_[MAX] = rawdb.cell_libs_[MAX];
}


void GTDatabase::ExtractTimingGraph() {
    res_unit = cell_libs_[MIN]->resistance_unit_->value();
    cap_unit = cell_libs_[MIN]->capacitance_unit_->value();
    time_unit = cell_libs_[MIN]->time_unit_->value();
    pin_names = gpdb.getPinNames();
    net_names = gpdb.getNetNames();
    
    //  Flatten Liberty Cell Timing
    for (db::CellType* cell_type : rawdb.celltypes) {
        string cell_type_name = cell_type->name;
        array<LibertyCell*, 2> liberty_cell_view = {cell_libs_[MIN]->get_cell(cell_type_name), cell_libs_[MAX]->get_cell(cell_type_name)};
        if (!liberty_cell_view[MIN] || !liberty_cell_view[MAX]) {
            liberty_cell_type2port_list_end.push_back(liberty_cell_type2port_list_end.back());
            continue;
        }
        liberty_cell_type2port_list_end.push_back(liberty_cell_type2port_list_end.back() + liberty_cell_view[MIN]->ports_.size());
        for (int i = 0; i < liberty_cell_view[MIN]->ports_.size(); i++) {
            array<LibertyPort*, 2> liberty_port_view = {liberty_cell_view[MIN]->ports_[i], liberty_cell_view[MAX]->ports_[i]};
            for_each_el(el) {
                liberty_port_capacitance.push_back(liberty_port_view[el]->port_capacitance_[0].value_or(nanf("")));
                liberty_port_capacitance.push_back(liberty_port_view[el]->port_capacitance_[1].value_or(nanf("")));
                liberty_port_capacitance.push_back(liberty_port_view[el]->port_capacitance_[2].value_or(0.0f));
            }

            for_each_el(el) {
                liberty_port2timing_list_end.push_back(liberty_port2timing_list_end.back() + liberty_port_view[el]->timing_arcs_non_cond_non_bundle_.size());
                for (int j = 0; j < liberty_port_view[el]->timing_arcs_non_cond_non_bundle_.size(); j++) {
                    liberty_timing_arcs.push_back(liberty_port_view[el]->timing_arcs_non_cond_non_bundle_[j]);
                }
            }
        }
    }

    //  Traverse Circuit Pins
    //
    num_pins = gpdb.getPins().size();
    pin_names = gpdb.getPinNames();
    net_names = gpdb.getNetNames();
    pin_id2cell_type_id.resize(num_pins);
    pin_id2port_offset_id.resize(num_pins);
    STA_pins.resize(num_pins, nullptr);
    pin_capacitance.resize(2 * 3 * num_pins, 0.0f);
    for (auto& gppin : gpdb.getPins()) {
        int pin_id = gppin.getId();
        string pin_name = gppin.getName();
        string pin_macro_name = gppin.getMacroName();
        STA_pins[pin_id] = new STAPin();
        auto [ori_node_id, ori_node_pin_id, ori_net_id] = gppin.getOriDBInfo();
        if (ori_node_pin_id == -1) {
            auto dbiopin = rawdb.iopins[ori_node_id];
            pin_id2cell_type_id[pin_id] = -1;
            if (dbiopin->type->direction() == 'i') {
                primary_outputs.push_back(pin_id);
                endpoints_id.push_back(pin_id);
                primary_output2pin_id[pin_name] = pin_id;
            } else if (dbiopin->type->direction() == 'o') {
                primary_inputs.push_back(pin_id);
                primary_input2pin_id[pin_name] = pin_id;
            }
        } else {
            auto& dbcell = rawdb.cells[ori_node_id];
            LibertyCell* liberty_cell = dbcell->ctype()->liberty_cell;
            pin_id2cell_type_id[pin_id] = dbcell->ctype()->libcell();
            pin_id2port_offset_id[pin_id] = liberty_cell->ports_map_[pin_macro_name];

            int liberty_port_id = liberty_cell_type2port_list_end[pin_id2cell_type_id[pin_id]] + pin_id2port_offset_id[pin_id];

            for_each_el(el) {
                pin_capacitance[6 * pin_id + el * 2 + 0] = liberty_port_capacitance[6 * liberty_port_id + el * 3 + 0];
                pin_capacitance[6 * pin_id + el * 2 + 1] = liberty_port_capacitance[6 * liberty_port_id + el * 3 + 1];
                pin_capacitance[6 * pin_id + 4 + el] = liberty_port_capacitance[6 * liberty_port_id + el * 3 + 2];
            }
        }
    }
    num_POs = primary_outputs.size();


    //  Map Pin to Liberty Timing
    //
    auto connect_from_to_pin = [&](int from_pin_id, int to_pin_id) -> pair<STAPin*, STAPin*> {
        STAPin* from_pin = STA_pins[from_pin_id];
        STAPin* to_pin = STA_pins[to_pin_id];
        from_pin->fanout_pin_ids.insert(to_pin_id);
        to_pin->fanin_pin_ids.insert(from_pin_id);
        timing_arc_from_pin_id.push_back(from_pin_id);
        timing_arc_to_pin_id.push_back(to_pin_id);
        from_pin->timing_arc_out.push_back(num_arcs);
        to_pin->timing_arc_in.push_back(num_arcs);
        return {from_pin, to_pin};
    };

    for (auto& gpnet : gpdb.getNets()) {
        int driver_pin_id = gpnet.pins()[0];
        for (index_type i = 1; i < static_cast<index_type>(gpnet.pins().size()); i++) {
            int sink_pin_id = gpnet.pins()[i];
            auto [from_pin, to_pin] = connect_from_to_pin(driver_pin_id, sink_pin_id);
            timing_arc_id_map.push_back(-1);
            timing_arc_id_map.push_back(-1);
            arc_types.push_back(0);
            arc_id2test_id.push_back(-1);
            num_arcs++;
        }
    }

    cell_node_type_map.resize(gpdb.getNodes().size(), -1);
    for (auto& dbcell : rawdb.cells) {
        int gpdb_id = dbcell->gpdb_id;
        int libcell_id = dbcell->ctype()->libcell();
        cell_node_type_map[gpdb_id] = libcell_id;
        for_each_el(el) {
            for (int pin_id : gpdb.getNodes()[gpdb_id].pins()) {
                int pin_id2port_start = liberty_cell_type2port_list_end[libcell_id];
                int pin_id2port_offset = pin_id2port_offset_id[pin_id];
                int port_id = pin_id2port_start + pin_id2port_offset;
                int start = liberty_port2timing_list_end[2 * port_id + el];
                int end = liberty_port2timing_list_end[2 * port_id + el + 1];
                for (int i = start; i < end; i++) {
                    TimingArc* timing_arc = liberty_timing_arcs[i];
                    if (is_redundant_timing(timing_arc, el)) {
                        continue;
                    }
                    array<int, 2> timing_view = {-1, -1};
                    timing_view[el] = i;

                    int from_pin_id = gpdb.getNodes()[gpdb_id].getPinbyPortName(timing_arc->from_port_->name);;
                    int to_pin_id = gpdb.getNodes()[gpdb_id].getPinbyPortName(timing_arc->to_port_->name);
                    auto [from_pin, to_pin] = connect_from_to_pin(from_pin_id, to_pin_id);
                    timing_arc_id_map.push_back(timing_view[MIN]);
                    timing_arc_id_map.push_back(timing_view[MAX]);
                    arc_types.push_back(1);
                    num_arcs++;

                    if (timing_arc->is_constraint()) {
                        arc_id2test_id.push_back(num_tests++);
                        test_id2_arc_id.push_back(num_arcs - 1);
                        endpoints_id.push_back(to_pin_id);
                    } else {
                        arc_id2test_id.push_back(-1);
                    }
                }
            }
        }
    }

    //  Construct Connectivity Graph
    //
    for (int i = 0; i < num_pins; i++) total_num_fanouts += STA_pins[i]->fanout_pin_ids.size();

    pin_fanout_list_end.resize(num_pins + 1);
    pin_fanout_list_end[0] = 0;
    pin_num_fanin.resize(num_pins);
    pin_fanout_list.resize(total_num_fanouts);

    index_type ptr = 0;
    index_type last_idx = 0;
    for (index_type i = 0; i < static_cast<index_type>(num_pins); i++) {
        for (auto fanout_pin_id : STA_pins[i]->fanout_pin_ids) pin_fanout_list[ptr++] = fanout_pin_id;
        last_idx += STA_pins[i]->fanout_pin_ids.size();
        pin_fanout_list_end[i + 1] = last_idx;
        pin_num_fanin[i] = STA_pins[i]->fanin_pin_ids.size();
    }
    for (int i = 0; i < num_pins; i++) {
        if (pin_num_fanin[i] == 0) pin_frontiers.push_back(i);
    }

    pin_forward_arc_list_end.push_back(0);
    pin_backward_arc_list_end.push_back(0);
    for (index_type i = 0; i < static_cast<index_type>(num_pins); i++) {
        for (auto fanout_arc : STA_pins[i]->timing_arc_out) {
            pin_forward_arc_list.push_back(fanout_arc);
        }
        pin_forward_arc_list_end.push_back(pin_forward_arc_list.size());
        for (auto fanin_arc : STA_pins[i]->timing_arc_in) {
            pin_backward_arc_list.push_back(fanin_arc);
        }
        pin_backward_arc_list_end.push_back(pin_backward_arc_list.size());
    }

    // gputimer arrays
    auto device = timing_raw_db.node_size.device();
    auto options = torch::TensorOptions().dtype(torch::kInt32);
    // Timer graph topology variables
    timing_raw_db.pin_forward_arc_list = torch::from_blob(pin_forward_arc_list.data(), {static_cast<index_type>(pin_forward_arc_list.size())}, options).contiguous().to(device);
    timing_raw_db.pin_forward_arc_list_end = torch::from_blob(pin_forward_arc_list_end.data(), {static_cast<index_type>(pin_forward_arc_list_end.size())}, options).contiguous().to(device);
    timing_raw_db.pin_backward_arc_list = torch::from_blob(pin_backward_arc_list.data(), {static_cast<index_type>(pin_backward_arc_list.size())}, options).contiguous().to(device);
    timing_raw_db.pin_backward_arc_list_end = torch::from_blob(pin_backward_arc_list_end.data(), {static_cast<index_type>(pin_backward_arc_list_end.size())}, options).contiguous().to(device);
    timing_raw_db.timing_arc_from_pin_id = torch::from_blob(timing_arc_from_pin_id.data(), {static_cast<index_type>(timing_arc_from_pin_id.size())}, options).contiguous().to(device);
    timing_raw_db.timing_arc_to_pin_id = torch::from_blob(timing_arc_to_pin_id.data(), {static_cast<index_type>(timing_arc_to_pin_id.size())}, options).contiguous().to(device);
    timing_raw_db.pin_num_fanin = torch::from_blob(pin_num_fanin.data(), {static_cast<index_type>(pin_num_fanin.size())}, options).contiguous().to(device);
    timing_raw_db.pin_fanout_list = torch::from_blob(pin_fanout_list.data(), {static_cast<index_type>(pin_fanout_list.size())}, options).contiguous().to(device);
    timing_raw_db.pin_fanout_list_end = torch::from_blob(pin_fanout_list_end.data(), {static_cast<index_type>(pin_fanout_list_end.size())}, options).contiguous().to(device);

    // Timer timing liberty variables
    timing_raw_db.arc_types = torch::from_blob(arc_types.data(), {static_cast<int>(arc_types.size())}, options).contiguous().to(device);
    timing_raw_db.timing_arc_id_map = torch::from_blob(timing_arc_id_map.data(), {static_cast<int>(timing_arc_id_map.size())}, options).contiguous().to(device);
    timing_raw_db.arc_id2test_id = torch::from_blob(arc_id2test_id.data(), {static_cast<int>(arc_id2test_id.size())}, options).contiguous().to(device);
    timing_raw_db.test_id2_arc_id = torch::from_blob(test_id2_arc_id.data(), {static_cast<int>(test_id2_arc_id.size())}, options).contiguous().to(device);
    timing_raw_db.endpoints_id = torch::from_blob(endpoints_id.data(), {static_cast<index_type>(endpoints_id.size())}, options).contiguous().to(device);

    timing_raw_db.pinSlew = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinLoad = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinRAT = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinAT = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinImpulse = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinRootDelay = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    torch::fill_(timing_raw_db.pinSlew, nanf(""));
    torch::fill_(timing_raw_db.pinRAT, nanf(""));
    torch::fill_(timing_raw_db.pinAT, nanf(""));
    torch::fill_(timing_raw_db.pinImpulse, nanf(""));
    torch::fill_(timing_raw_db.pinRootDelay, nanf(""));

    timing_raw_db.arcDelay = torch::zeros({num_arcs, 2 * NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinImpulse_ref = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinLoad_ref = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinLoad_ratio = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinRootDelay_ref = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinRootDelay_ratio = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinRootDelay_compensation = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();

    logger.info("Design info: %d pins, %d arcs, %d tests", num_pins, num_arcs, num_tests);
}

void GTDatabase::readSdc(sdc::SDC& sdc) {
    for (auto& command : sdc.commands) {
        std::visit(Functors{[this](auto&& cmd) { _read_sdc(cmd); }}, command);
    }

    // string clock_name = clocks.begin()->second.source_name();
    string clock_name = gpdb.getPins()[clocks.begin()->second.source_id()].getName();
    float period = clocks.begin()->second.period();
    logger.info("clock: %s, period: %.2f", clock_name.c_str(), period);

    net_is_clock.resize(gpdb.getNets().size(), 0);
    for (auto& gpnet : gpdb.getNets()) {
        if (gpnet.getName() == clock_name) {
            net_is_clock[gpnet.getId()] = 1;
        }
    }

    // set nan slew of PIs to half period
    for (auto& pi : primary_inputs) {
        if (torch::isnan(timing_raw_db.pinSlew[pi][0]).item<bool>()) timing_raw_db.pinSlew[pi][0] = 0.0f;
        if (torch::isnan(timing_raw_db.pinSlew[pi][1]).item<bool>()) timing_raw_db.pinSlew[pi][1] = 0.0f;
        if (torch::isnan(timing_raw_db.pinSlew[pi][2]).item<bool>()) timing_raw_db.pinSlew[pi][2] = 0.0f;
        if (torch::isnan(timing_raw_db.pinSlew[pi][3]).item<bool>()) timing_raw_db.pinSlew[pi][3] = 0.0f;
        // if (torch::isnan(pinAT[pi][0]).item<bool>()) pinAT[pi][0] = 0.0f;
        // if (torch::isnan(pinAT[pi][1]).item<bool>()) pinAT[pi][1] = period / 2.0;
        // if (torch::isnan(pinAT[pi][2]).item<bool>()) pinAT[pi][2] = 0.0f;
        // if (torch::isnan(pinAT[pi][3]).item<bool>()) pinAT[pi][3] = period / 2.0;
    }

    if (clocks.begin()->second.source_id() != -1) {
        int clock_pin_id = clocks.begin()->second.source_id();
        if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][0]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][0] = 0.0f;
        if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][1]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][1] = 0.0f;
        if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][2]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][2] = 0.0f;
        if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][3]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][3] = 0.0f;
        // if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][0]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][0] = 0.0f;
        // if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][1]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][1] = period / 2.0;
        // if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][2]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][2] = 0.0f;
        // if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][3]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][3] = period / 2.0;
    }
}

// Sets input delay on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetUnits& obj) {
    if (obj.time.has_value()) {
        auto s = *obj.time;
        if (s == "ps") sdc_time_unit = 1e-12;
        if (s == "ns") sdc_time_unit = 1e-9;
        if (s == "us") sdc_time_unit = 1e-6;
        if (s == "ms") sdc_time_unit = 1e-3;
        if (s == "s") sdc_time_unit = 1.0;
    }
    if (obj.capacitance.has_value()) {
        auto s = *obj.capacitance;
        if (s == "fF") sdc_cap_unit = 1e-15;
        if (s == "pF") sdc_cap_unit = 1e-12;
        if (s == "nF") sdc_cap_unit = 1e-9;
        if (s == "uF") sdc_cap_unit = 1e-6;
        if (s == "F") sdc_cap_unit = 1.0;
    }
    if (obj.resistance.has_value()) {
        auto s = *obj.resistance;
        if (s == "Ohm") sdc_res_unit = 1.0;
        if (s == "kOhm") sdc_res_unit = 1e3;
        if (s == "MOhm") sdc_res_unit = 1e6;
    }
    if (sdc_time_unit.has_value()) printf("sdc time unit: %.2E\n", *sdc_time_unit);
    if (sdc_cap_unit.has_value()) printf("sdc capacitance unit: %.2E\n", *sdc_cap_unit);
    if (sdc_res_unit.has_value()) printf("sdc resistance unit: %.2E\n", *sdc_res_unit);
}

// Sets input delay on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetInputDelay& obj) {
    assert(obj.delay_value && obj.port_pin_list);

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);

    std::visit(Functors{[&](sdc::AllInputs&) {
                            for (auto& pi : primary_inputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    float delay = *obj.delay_value;
                                    if (sdc_time_unit.has_value()) delay = delay * *sdc_time_unit / time_unit;
                                    timing_raw_db.pinAT[pi][(el << 1) + rf] = delay;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(port); itr != primary_input2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float delay = *obj.delay_value;
                                        if (sdc_time_unit.has_value()) delay = delay * *sdc_time_unit / time_unit;
                                        timing_raw_db.pinAT[itr->second][(el << 1) + rf] = delay;
                                    }
                                } else {
                                    printf(obj.command, ": port ", std::quoted(port), " not found");
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_pin_list);
}

// Sets input transition on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetInputTransition& obj) {
    assert(obj.transition && obj.port_list);

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);

    std::visit(Functors{[&](sdc::AllInputs&) {
                            for (auto& pi : primary_inputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    float transition = *obj.transition;
                                    if (sdc_time_unit.has_value()) transition = transition * *sdc_time_unit / time_unit;
                                    timing_raw_db.pinSlew[pi][(el << 1) + rf] = transition;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(port); itr != primary_input2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float transition = *obj.transition;
                                        if (sdc_time_unit.has_value()) transition = transition * *sdc_time_unit / time_unit;
                                        timing_raw_db.pinSlew[itr->second][(el << 1) + rf] = transition;
                                    }
                                } else {
                                    printf(obj.command, ": port ", std::quoted(port), " not found");
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_list);
}

// Sets input transition on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetDrivingCell& obj) {
    assert((obj.transitions[0] || obj.transitions[1]) && obj.port_list);

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);

    std::visit(Functors{[&](sdc::AllInputs&) {
                            for (auto& pi : primary_inputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    float transition = *obj.transitions[el];
                                    if (sdc_time_unit.has_value()) transition = transition * *sdc_time_unit / time_unit;
                                    timing_raw_db.pinSlew[pi][(el << 1) + rf] = transition;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(port); itr != primary_input2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float transition = *obj.transitions[el];
                                        if (sdc_time_unit.has_value()) transition = transition * *sdc_time_unit / time_unit;
                                        timing_raw_db.pinSlew[itr->second][(el << 1) + rf] = transition;
                                    }
                                } else {
                                    printf(obj.command, ": port ", std::quoted(port), " not found");
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_list);
}

// Sets output delay on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetOutputDelay& obj) {
    assert(obj.delay_value && obj.port_pin_list);

    if (clocks.find(obj.clock) == clocks.end()) {
        printf(obj.command, ": clock ", std::quoted(obj.clock), " not found");
        return;
    }

    auto& clock = clocks.at(obj.clock);

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);

    std::visit(Functors{[&](sdc::AllOutputs&) {
                            for (auto& po : primary_outputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    float delay = *obj.delay_value;
                                    if (sdc_time_unit.has_value()) delay = delay * *sdc_time_unit / time_unit;
                                    timing_raw_db.pinRAT[po][(el << 1) + rf] = el == MIN ? -delay : clock.period() - delay;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_output2pin_id.find(port); itr != primary_output2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float delay = *obj.delay_value;
                                        if (sdc_time_unit.has_value()) delay = delay * *sdc_time_unit / time_unit;
                                        timing_raw_db.pinRAT[itr->second][(el << 1) + rf] = el == MIN ? -delay : clock.period() - delay;
                                    }
                                } else {
                                    printf(obj.command, ": port ", std::quoted(port), " not found");
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_pin_list);
}

// Sets the load attribute to a specified value on specified ports and nets.
void GTDatabase::_read_sdc(sdc::SetLoad& obj) {
    assert(obj.value && obj.objects);

    auto mask = sdc::TimingMask(obj.min, obj.max, std::nullopt, std::nullopt);

    std::visit(Functors{[&](sdc::AllOutputs&) {
                            for (auto& po : primary_outputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    float load = *obj.value;
                                    if (sdc_res_unit.has_value()) load = load * *sdc_res_unit / res_unit;
                                    timing_raw_db.pinLoad[po][(el << 1) + rf] = load;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_output2pin_id.find(port); itr != primary_output2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float load = *obj.value;
                                        if (sdc_res_unit.has_value()) load = load * *sdc_res_unit / res_unit;
                                        timing_raw_db.pinLoad[itr->second][(el << 1) + rf] = load;
                                    }
                                } else {
                                    printf(obj.command, ": port ", std::quoted(port), " not found");
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.objects);
}

void GTDatabase::_read_sdc(sdc::CreateClock& obj) {
    assert(obj.period && !obj.name.empty());

    // create clock from given sources
    if (obj.port_pin_list) {
        std::visit(Functors{[&](sdc::GetPorts& get_ports) {
                                auto& ports = get_ports.ports;
                                assert(ports.size() == 1);
                                if (auto itr = primary_input2pin_id.find(ports.front()); itr != primary_input2pin_id.end()) {
                                    clocks.try_emplace(obj.name, obj.name, itr->second, *obj.period);
                                } else {
                                    printf(obj.command, ": port ", std::quoted(ports.front()), " not found");
                                }
                            },
                            [](auto&&) { assert(false); }},
                   *obj.port_pin_list);
    }
    // create virtual clock
    else {
        clocks.try_emplace(obj.name, obj.name, *obj.period);
    }
}

TimingTorchRawDB::TimingTorchRawDB(torch::Tensor node_lpos_init_,
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
                                   float wire_capacitance_per_micron_) {
    node_lpos_init = node_lpos_init_;
    node_size = node_size_;
    pin_rel_lpos = pin_rel_lpos_;

    node_size_x = node_size.index({"...", 0}).clone().contiguous();
    node_size_y = node_size.index({"...", 1}).clone().contiguous();
    init_x = node_lpos_init.index({"...", 0}).clone().contiguous();
    init_y = node_lpos_init.index({"...", 1}).clone().contiguous();
    pin_offset_x = pin_rel_lpos.index({"...", 0}).clone().contiguous();
    pin_offset_y = pin_rel_lpos.index({"...", 1}).clone().contiguous();
    x = init_x.clone().contiguous();
    y = init_y.clone().contiguous();

    num_nodes = node_size.size(0);
    num_pins = pin_id2node_id_.size(0);
    num_nets = hyperedge_list_end_.size(0);
    num_movable_nodes = num_movable_nodes_;
    net_mask = net_mask_;

    pinAT = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(node_size.device()))).contiguous();
    pinRAT = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(node_size.device()))).contiguous();
    at_prefix_pin = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kInt32).device(torch::Device(node_size.device()))).contiguous();
    at_prefix_arc = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kInt32).device(torch::Device(node_size.device()))).contiguous();
    at_prefix_attr = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kInt32).device(torch::Device(node_size.device()))).contiguous();

    flat_node2pin_start_map = torch::cat({torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::Device(node_size.device()))), node2pin_list_end_}, 0).to(torch::kInt32).contiguous();
    flat_node2pin_map = node2pin_list_.to(torch::kInt32);
    pin2node_map = pin_id2node_id_.to(torch::kInt32);

    flat_net2pin_start_map = torch::cat({torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::Device(node_size.device()))), hyperedge_list_end_}, 0).to(torch::kInt32).contiguous();
    flat_net2pin_map = hyperedge_list_.to(torch::kInt32);
    pin2net_map = pin_id2net_id_.to(torch::kInt32);

    num_threads = std::max(6, 1);
    scale_factor = scale_factor_;
    microns = microns_;
    wire_resistance_per_micron = wire_resistance_per_micron_;
    wire_capacitance_per_micron = wire_capacitance_per_micron_;
}

void TimingTorchRawDB::commit_from(torch::Tensor x_, torch::Tensor y_) {
    // commit external pos to original pos
    init_x.index({torch::indexing::Slice(0, num_movable_nodes)}).data().copy_(x_.index({torch::indexing::Slice(0, num_movable_nodes)}));
    init_y.index({torch::indexing::Slice(0, num_movable_nodes)}).data().copy_(y_.index({torch::indexing::Slice(0, num_movable_nodes)}));
    x.index({torch::indexing::Slice(0, num_movable_nodes)}).data().copy_(x_.index({torch::indexing::Slice(0, num_movable_nodes)}));
    y.index({torch::indexing::Slice(0, num_movable_nodes)}).data().copy_(y_.index({torch::indexing::Slice(0, num_movable_nodes)}));
}

torch::Tensor TimingTorchRawDB::get_curr_cposx() { return x + node_size_x / 2; }
torch::Tensor TimingTorchRawDB::get_curr_cposy() { return y + node_size_y / 2; }
torch::Tensor TimingTorchRawDB::get_curr_lposx() { return x; }
torch::Tensor TimingTorchRawDB::get_curr_lposy() { return y; }

}  // namespace gt