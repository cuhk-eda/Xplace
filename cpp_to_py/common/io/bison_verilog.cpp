#include <unordered_set>
#include <fstream>

#include "common/db/Cell.h"
#include "common/db/Database.h"
#include "common/db/Net.h"
#include "common/db/Pin.h"
#include "verilog/verilog_driver.hpp"
namespace db {

// helper type for the visitor #4
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

class VerilogParser : public verilog::ParserVerilogInterface {
public:
    Database &db;
    bool lef_read = false;
    bool def_read = false;
    VerilogParser() = default;
    VerilogParser(Database *db, bool lef_read, bool def_read) : db(*db), lef_read(lef_read), def_read(def_read) {}
    virtual ~VerilogParser() {}
    vector<verilog::Port> ports;

    void add_module(std::string &&name) { 
        std::cout << "Module name = " << name << '\n'; 
        db.module_name = name;
    }

    void add_port(verilog::Port &&port) {
        ports.push_back(port);
        auto addIOPin = [&](const std::string &iopinname, const char direction) {
            IOPin *iopin;
            if (def_read) {
                iopin = db.getIOPin(iopinname);
                if (!iopin) {
                    logger.warning("IO pin is not defined: %s", iopinname.c_str());
                    return;
                }
            } else {
                iopin = db.addIOPin(iopinname, iopinname, direction);
            }

            Pin *pin = iopin->pin;
            iopin->is_connected = true;
            Net *net = db.addNet(iopinname);
            net->is_port = true;

            pin->net = net;
            pin->is_connected = true;
            net->addPin(iopin->pin);
        };
        for (auto &name : port.names) {
            if (port.dir == verilog::PortDirection::INPUT || port.dir == verilog::PortDirection::OUTPUT) {
                char direction = 'x';
                direction = (port.dir == verilog::PortDirection::INPUT) ? 'o' : 'i';
                if ((port.beg != -1) && (port.end != -1)) {
                    for (int i = port.beg; i >= port.end; i--) {
                        // INPUT to the chip, output from external
                        // OUTPUT to the chip, input to external
                        std::string iopinname(name + "[" + std::to_string(i) + "]");
                        addIOPin(iopinname, direction);
                    }
                } else {
                    std::string iopinname(name);
                    addIOPin(iopinname, direction);
                }
            }
        }
    }

    void add_net(verilog::Net &&net) {
        auto addNet = [&](const std::string &netName) { db.addNet(netName); };
        for (auto &name : net.names) {
            if ((net.beg != -1) && (net.end != -1)) {
                for (int i = net.beg; i >= net.end; i--) {
                    std::string netName(name + "[" + std::to_string(i) + "]");
                    netName = validate_token(netName);
                    addNet(netName);
                }
            } else {
                std::string netName(name);
                netName = validate_token(netName);
                addNet(netName);
            }
        }
    }

    void add_assignment(verilog::Assignment &&ast) {
        // std::cout << "Assignment: " << ast << '\n';
        // TODO:
    }

    void add_instance(verilog::Instance &&inst) {
        // remove '\' in the head
        string cellName(inst.inst_name);
        cellName = validate_token(cellName);
        Cell *cell;
        if (def_read) {
            cell = db.getCell(cellName);
            if (!cell) {
                logger.error("Cell is not defined: %s", cellName.c_str());
                return;
            }
        } else {
            // // TODO:
            // string macroName(inst.module_name);
            // CellType *celltype = db.getCellType(macroName);
            // if (!celltype) {
            //     celltype = db.addCellType(macroName, db.celltypes.size());
            //     for (auto [cellpin_name, cellpin] : db.cell_libs_[0]->cell(macroName)->cellpins) {
            //         char direction = 'x';
            //         char type = 's';
            //         switch (*cellpin.direction) {
            //             case gt::CellPortDirection::input:
            //                 direction = 'i';
            //                 break;
            //             case gt::CellPortDirection::output:
            //                 direction = 'o';
            //                 break;
            //             case gt::CellPortDirection::inout:
            //                 if (cellpin_name != "VDD" && cellpin_name != "vdd" && cellpin_name != "VSS" && cellpin_name != "vss") {
            //                     logger.warning("unknown pin %s.%s direction: %s", macroName.c_str(), cellpin_name.c_str(), "INOUT");
            //                 }
            //                 break;
            //             default:
            //                 logger.error("unknown pin %s.%s direction: %s", macroName.c_str(), cellpin_name.c_str(), "UNKNOWN");
            //                 break;
            //         }
            //         PinType* pintype = celltype->addPin(cellpin_name, direction, type);
            //     }
            // }

            // cell = db.addCell(cellName, celltype);
        }
        for (size_t i = 0; i < inst.pin_names.size(); i++) {
            if (inst.net_names[i].size() > 1) logger.error("Bus net name is not supported\n");
            // define std::string and NetBit visit methods
            std::string pin_name =
                std::visit(overloaded{
                               [](std::string &v) { return v; },
                               [](verilog::NetBit &v) { return v.name + '[' + std::to_string(v.bit) + ']'; },
                               [](verilog::NetRange &v) { return v.name + '[' + std::to_string(v.beg) + ':' + std::to_string(v.end) + ']'; },
                               [](verilog::Constant &v) { return v.value; },
                           },
                           inst.pin_names[i]);
            // printf("gate:pin %s:%s\n", cellName.c_str(), pin_name.c_str());

            std::string net_name =
                std::visit(overloaded{
                               [](std::string &v) { return v; },
                               [](verilog::NetBit &v) { return v.name + '[' + std::to_string(v.bit) + ']'; },
                               [](verilog::NetRange &v) { return v.name + '[' + std::to_string(v.beg) + ':' + std::to_string(v.end) + ']'; },
                               [](verilog::Constant &v) { return v.value; },
                           },
                           inst.net_names[i][0]);

            std::string pinName(pin_name);
            std::string netName(net_name);
            netName = validate_token(netName);
            Net *net = db.getNet(netName);
            if (!net) logger.error("Net is not defined: %s", netName.c_str());
            Pin *pin;
            if (def_read) {
                pin = cell->pin(pinName);
                if (!pin) logger.error("Pin is not defined: %s", pinName.c_str());
            } else {
                pin = cell->pin(pinName);
                if (!pin) logger.error("Pin is not defined: %s", pinName.c_str());
            }
            pin->net = net;
            net->addPin(pin);
            pin->is_connected = true;
        }
        cell->is_connected = true;
    }
};

template <typename T>
void removeDuplicates(std::vector<T> &vec) {
    std::unordered_set<T> uniqueElements;
    vec.erase(std::remove_if(vec.begin(), vec.end(), [&uniqueElements](const T &element) { return !uniqueElements.insert(element).second; }),
              vec.end());
}

bool Database::readVerilog_yy(const std::string &file) {
    verilog_parser = new VerilogParser(this, lef_read, def_read);
    verilog_parser->read(file);

    // remove empty nets
    for (int i = 0; i < (int)nets.size(); i++) {
        if (nets[i]->pins.size() == 0) {
            nets.erase(nets.begin() + i);
            logger.warning("Empty net %s", nets[i]->name.c_str());
            i--;
        }
    }

    // remove duplicates in net pins
    for (auto &net : nets) {
        auto &pins = net->pins;
        removeDuplicates(net->pins);
    }

    return true;
}

string tokenize_name(const string& name) {
    // if '[' or ']' in the name, add '\' in the front
    string new_name;
    for (auto &c : name) {
        if (c == '[' || c == ']' || c == '$' || c == '.') {
            new_name += '\\';
            break;
        }
    }
    new_name += name;
    return new_name;
}

}