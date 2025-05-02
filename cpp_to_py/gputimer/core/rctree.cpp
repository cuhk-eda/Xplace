
#include "GPUTimer.h"
#include "common/utils/utils.h"
#include "common/db/Database.h"
#include "gputimer/db/GTDatabase.h"
#include <flute.hpp>
using namespace Flute;

namespace gt {

void update_rc_timing_cuda(float* x,
                           float* y,
                           const float* pin_offset_x,
                           const float* pin_offset_y,
                           const int* pin2node_map,
                           const int* flat_net2pin_start_map,
                           const int* flat_net2pin_map,
                           float* pinLoad,
                           float* pinImpulse,
                           float* pinCap,
                           float* pinWireCap,
                           float* pinRootDelay,
                           float* pinRootRes,
                           int num_nets,
                           int num_pins,
                           float unit_to_micron,
                           int* net_is_clock,
                           float cf,
                           float rf);

void GPUTimer::update_rc_timing(torch::Tensor node_lpos, bool record, bool load, bool conpensation) {
    timing_raw_db.commit_from(node_lpos.index({"...", 0}).contiguous(), node_lpos.index({"...", 1}).contiguous());
    float unit_to_micron = scale_factor * microns;
    float rf = wire_resistance_per_micron / res_unit;
    float cf = wire_capacitance_per_micron / cap_unit;
    update_rc_timing_cuda(x,
                          y,
                          pin_offset_x,
                          pin_offset_y,
                          pin2node_map,
                          flat_net2pin_start_map,
                          flat_net2pin_map,
                          pinLoad,
                          pinImpulse,
                          pinCap,
                          pinWireCap,
                          pinRootDelay,
                          pinRootRes,
                          num_nets,
                          num_pins,
                          unit_to_micron,
                          net_is_clock,
                          cf,
                          rf);
    if (record) {
        auto ratio_load = torch::nan_to_num(timing_raw_db.pinLoad_ref / timing_raw_db.pinLoad, 1.0);
        timing_raw_db.pinLoad_ratio.data().copy_(ratio_load.contiguous().data());
        auto ratio_delay = torch::sqrt(torch::nan_to_num(timing_raw_db.pinRootDelay_ref / timing_raw_db.pinRootDelay, 1.0));
        timing_raw_db.pinRootDelay_ratio.data().copy_(ratio_delay.contiguous().data());

        auto delay_comp = (torch::nan_to_num(timing_raw_db.pinRootDelay_ref - timing_raw_db.pinRootDelay, 0)).clamp(0.0);
        timing_raw_db.pinRootDelay_compensation.data().copy_(delay_comp.contiguous().data());
    }
    if (load) {
        timing_raw_db.pinImpulse.data().copy_(timing_raw_db.pinImpulse_ref.data());
        timing_raw_db.pinLoad *= timing_raw_db.pinLoad_ratio;
        if (conpensation)
            timing_raw_db.pinRootDelay += timing_raw_db.pinRootDelay_compensation;
        else
            timing_raw_db.pinRootDelay *= timing_raw_db.pinRootDelay_ratio;
    }
}

// ------------------------------------------------------------------------------------------------------------------------
//

auto& retrieve_pins_from_pos(std::map<utils::PointT<int>, std::set<int>>& pos2pins_map, const utils::PointT<int>& point, int& index) {
    if (pos2pins_map.find(point) != pos2pins_map.end()) return pos2pins_map[point];
    pos2pins_map.emplace(point, std::set<int>{index++});
    return pos2pins_map[point];
}

tuple<vector<int>, vector<int>, vector<float>, vector<int>, vector<int>, vector<int>, int, int> FluteRCTree(TimingTorchRawDB& timing_raw_db,
                                                                                                            float rf,
                                                                                                            float cf) {
    torch::Tensor flat_node2pin_start_map_at = timing_raw_db.flat_node2pin_start_map.clone().cpu().contiguous();
    torch::Tensor flat_node2pin_map_at = timing_raw_db.flat_node2pin_map.clone().cpu().contiguous();
    torch::Tensor pin2node_map_at = timing_raw_db.pin2node_map.clone().cpu().contiguous();
    torch::Tensor flat_net2pin_start_map_at = timing_raw_db.flat_net2pin_start_map.clone().cpu().contiguous();
    torch::Tensor flat_net2pin_map_at = timing_raw_db.flat_net2pin_map.clone().cpu().contiguous();
    torch::Tensor pin2net_map_at = timing_raw_db.pin2net_map.clone().cpu().contiguous();
    torch::Tensor x_at = timing_raw_db.x.clone().cpu().contiguous();
    torch::Tensor y_at = timing_raw_db.y.clone().cpu().contiguous();
    torch::Tensor pin_offset_x_at = timing_raw_db.pin_offset_x.clone().cpu().contiguous();
    torch::Tensor pin_offset_y_at = timing_raw_db.pin_offset_y.clone().cpu().contiguous();

    const int* flat_node2pin_start_map = flat_node2pin_start_map_at.data_ptr<int>();
    const int* flat_node2pin_map = flat_node2pin_map_at.data_ptr<int>();
    const int* pin2node_map = pin2node_map_at.data_ptr<int>();
    const int* flat_net2pin_start_map = flat_net2pin_start_map_at.data_ptr<int>();
    const int* flat_net2pin_map = flat_net2pin_map_at.data_ptr<int>();
    const int* pin2net_map = pin2net_map_at.data_ptr<int>();
    const float* x = x_at.data_ptr<float>();
    const float* y = y_at.data_ptr<float>();
    const float* pin_offset_x = pin_offset_x_at.data_ptr<float>();
    const float* pin_offset_y = pin_offset_y_at.data_ptr<float>();
    int& num_nets = timing_raw_db.num_nets;

    constexpr const int scale = 1000;  // flute only supports integers.
    using Point2i = utils::PointT<int>;

    vector<int> edge_from;
    vector<int> edge_to;
    vector<float> edge_wl;
    vector<int> flat_net2node_start_map;
    vector<int> flat_net2edge_start_map;
    vector<int> node2pin_map;
    int node_count = 0;
    int edge_count = 0;
    flat_net2node_start_map.push_back(0);
    flat_net2edge_start_map.push_back(0);

    vector<vector<int>> net_id2edge_from(num_nets);
    vector<vector<int>> net_id2edge_to(num_nets);
    vector<vector<float>> net_id2edge_wl(num_nets);
    vector<vector<int>> net_id2node2pin_map(num_nets);

    omp_lock_t lock;
    omp_init_lock(&lock);
#pragma omp parallel for
    for (int i = 0; i < num_nets; ++i) {
        const int degree = flat_net2pin_start_map[i + 1] - flat_net2pin_start_map[i];
        const int root = flat_net2pin_map[flat_net2pin_start_map[i]];
        std::map<Point2i, std::set<int>> pos2pins_map;
        std::vector<int> vx, vy;
        vx.reserve(degree);
        vy.reserve(degree);

        std::map<int, int> global2inner_map;

        for (int j = 0; j < degree; ++j) {
            int pin = flat_net2pin_map[j + flat_net2pin_start_map[i]];
            int node = pin2node_map[pin];
            float offset_x = pin_offset_x[pin], offset_y = pin_offset_y[pin];
            // Find the correct pin locations given cell locations.
            auto x_ = static_cast<int>((x[node] + offset_x) * scale);
            auto y_ = static_cast<int>((y[node] + offset_y) * scale);
            global2inner_map[pin] = j;

            if (pos2pins_map.find(Point2i(x_, y_)) != pos2pins_map.end())
                pos2pins_map[Point2i(x_, y_)].insert(j);
            else {
                pos2pins_map.emplace(Point2i(x_, y_), std::set<int>{j});
                vx.emplace_back(x_);
                vy.emplace_back(y_);
            }
        }
        const int valid_size = static_cast<int>(vx.size());
        int num_pins = degree;
        std::set<Point2i> multipin_pos;
        std::map<Point2i, Point2i> pos2neighbor_map;

        if (valid_size > 1) {
            Tree flutetree = flute(valid_size, vx.data(), vy.data(), 8);

            for (int bid = 0; bid < 2 * valid_size - 2; ++bid) {
                Branch& branch1 = flutetree.branch[bid];
                Branch& branch2 = flutetree.branch[branch1.n];

                Point2i p1(branch1.x, branch1.y), p2(branch2.x, branch2.y);

                if (p1 == p2) continue;

                pos2neighbor_map.emplace(p2, p1);
                auto& id1 = retrieve_pins_from_pos(pos2pins_map, p1, num_pins);
                auto& id2 = retrieve_pins_from_pos(pos2pins_map, p2, num_pins);

                auto distance = Dist(p1, p2);
                float wl = static_cast<float>(distance * 1.0) / scale;

                if (!id1.empty() && !id2.empty()) {
                    auto base1 = id1.begin(), base2 = id2.begin();
                    if (*base1 != *base2) {
                        net_id2edge_from[i].emplace_back(*base1);
                        net_id2edge_to[i].emplace_back(*base2);
                        net_id2edge_wl[i].emplace_back(wl);
                    }
                    if (id1.size() > 1) multipin_pos.insert(p1);
                    if (id2.size() > 1) multipin_pos.insert(p2);
                }
            }
            free(flutetree.branch);
        } else if (valid_size == 1 && degree > 1) {
            multipin_pos.emplace(vx[0], vy[0]);
        }
        for (const auto& pos : multipin_pos) {
            const auto& pins = pos2pins_map[pos];
            int adj_pin = global2inner_map[root];
            const auto& _ppos = pos2neighbor_map[pos];
            if (auto itr = pos2pins_map.find(_ppos); itr != pos2pins_map.end()) {
                adj_pin = *itr->second.cbegin();
            }
            auto distance = Dist(pos, _ppos);
            float wl = static_cast<float>(distance * 1.0) / scale;
            for (auto it = std::next(pins.cbegin()); it != pins.cend(); ++it) {
                net_id2edge_from[i].emplace_back(adj_pin);
                net_id2edge_to[i].emplace_back(*it);
                net_id2edge_wl[i].emplace_back(0);
            }
        }

        for (int j = 0; j < num_pins; ++j) {
            if (j < degree)
                net_id2node2pin_map[i].push_back(flat_net2pin_map[j + flat_net2pin_start_map[i]]);
            else
                net_id2node2pin_map[i].push_back(-1);
        }
    }
    omp_destroy_lock(&lock);

    for (int i = 0; i < num_nets; ++i) {
        for (int j = 0; j < net_id2edge_from[i].size(); ++j) {
            edge_from.push_back(node_count + net_id2edge_from[i][j]);
            edge_to.push_back(node_count + net_id2edge_to[i][j]);
            edge_wl.push_back(net_id2edge_wl[i][j]);
            edge_count++;
        }
        node_count += net_id2node2pin_map[i].size();
        for (int j = 0; j < net_id2node2pin_map[i].size(); ++j) {
            node2pin_map.push_back(net_id2node2pin_map[i][j]);
        }
        flat_net2node_start_map.push_back(node_count);
        flat_net2edge_start_map.push_back(edge_count);
    }

    return {edge_from, edge_to, edge_wl, flat_net2node_start_map, flat_net2edge_start_map, node2pin_map, node_count, edge_count};
}

void flatten_rc_tree(std::vector<int> host_edge_from,
                     std::vector<int> host_edge_to,
                     float* edge_res,
                     float* node_cap,
                     std::vector<int> host_flat_net2node_start_map,
                     std::vector<int> host_flat_net2edge_start_map,
                     std::vector<int> host_node2pin_map,
                     int* node_order,
                     int* edge_order,
                     int* parent_node,
                     float* res_parent,
                     float* pinLoad,
                     float* pinImpulse,
                     float* pinCap,
                     float* pinWireCap,
                     float* pinRootDelay,
                     float* pinRootRes,
                     int num_nets,
                     int num_pins,
                     int num_nodes,
                     int num_edges);

void propagate_rc_tree(std::vector<int> host_edge_from,
                       std::vector<int> host_edge_to,
                       float* edge_res,
                       float* node_cap,
                       std::vector<int> host_flat_net2node_start_map,
                       std::vector<int> host_flat_net2edge_start_map,
                       std::vector<int> host_node2pin_map,
                       int* node_order,
                       int* parent_node,
                       float* res_parent,
                       float* pinLoad,
                       float* pinImpulse,
                       float* pinCap,
                       float* pinWireCap,
                       float* pinRootDelay,
                       float* pinRootRes,
                       int num_nets,
                       int num_pins,
                       int num_nodes,
                       int num_edges);


void calc_res_cap(std::vector<int> host_edge_from,
                  std::vector<int> host_edge_to,
                  int* edge_order,
                  float* edge_res,
                  float* node_cap,
                  std::vector<int> host_flat_net2node_start_map,
                  std::vector<int> host_flat_net2edge_start_map,
                  std::vector<int> host_node2pin_map,
                  std::vector<float> host_edge_wl,
                  int num_nets,
                  int num_edges,
                  int num_nodes,
                  int* net_is_clock,
                  float unit_to_micron,
                  float rf,
                  float cf);

void GPUTimer::update_rc_timing_flute(torch::Tensor node_lpos, bool record) {
    timing_raw_db.commit_from(node_lpos.index({"...", 0}).contiguous(), node_lpos.index({"...", 1}).contiguous());

    float unit_to_micron = scale_factor * microns;
    float rf = wire_resistance_per_micron / res_unit;
    float cf = wire_capacitance_per_micron / cap_unit;

    auto [edge_from, edge_to, edge_wl, flat_net2node_start_map, flat_net2edge_start_map, node2pin_map, num_nodes, num_edges] =
        FluteRCTree(timing_raw_db, rf, cf);
    auto device = timing_raw_db.node_size.device();
    torch::Tensor node_order = torch::zeros({num_nodes}, torch::dtype(torch::kInt32).device(device)).contiguous();
    torch::Tensor edge_order = torch::zeros({num_edges}, torch::dtype(torch::kInt32).device(device)).contiguous();
    torch::Tensor parent_node = -torch::ones({num_nodes}, torch::dtype(torch::kInt32).device(device)).contiguous();
    torch::Tensor res_parent = torch::zeros({num_nodes * NUM_ATTR}, torch::dtype(torch::kFloat32).device(device)).contiguous();
    torch::Tensor node_cap = torch::zeros({num_nodes * NUM_ATTR}, torch::dtype(torch::kFloat32).device(device)).contiguous();
    torch::Tensor edge_res = torch::zeros({num_edges}, torch::dtype(torch::kFloat32).device(device)).contiguous();

    calc_res_cap(edge_from,
                 edge_to,
                 edge_order.data_ptr<int>(),
                 edge_res.data_ptr<float>(),
                 node_cap.data_ptr<float>(),
                 flat_net2node_start_map,
                 flat_net2edge_start_map,
                 node2pin_map,
                 edge_wl,
                 num_nets,
                 num_edges,
                 num_nodes,
                 net_is_clock,
                 unit_to_micron,
                 rf,
                 cf);

    flatten_rc_tree(edge_from,
                    edge_to,
                    edge_res.data_ptr<float>(),
                    node_cap.data_ptr<float>(),
                    flat_net2node_start_map,
                    flat_net2edge_start_map,
                    node2pin_map,
                    node_order.data_ptr<int>(),
                    edge_order.data_ptr<int>(),
                    parent_node.data_ptr<int>(),
                    res_parent.data_ptr<float>(),
                    pinLoad,
                    pinImpulse,
                    pinCap,
                    pinWireCap,
                    pinRootDelay,
                    pinRootRes,
                    num_nets,
                    num_pins,
                    num_nodes,
                    num_edges);

    propagate_rc_tree(edge_from,
                      edge_to,
                      edge_res.data_ptr<float>(),
                      node_cap.data_ptr<float>(),
                      flat_net2node_start_map,
                      flat_net2edge_start_map,
                      node2pin_map,
                      node_order.data_ptr<int>(),
                      parent_node.data_ptr<int>(),
                      res_parent.data_ptr<float>(),
                      pinLoad,
                      pinImpulse,
                      pinCap,
                      pinWireCap,
                      pinRootDelay,
                      pinRootRes,
                      num_nets,
                      num_pins,
                      num_nodes,
                      num_edges);

    if (record) {
        timing_raw_db.pinImpulse_ref.data().copy_(timing_raw_db.pinImpulse.data());
        timing_raw_db.pinLoad_ref.data().copy_(timing_raw_db.pinLoad.data());
        timing_raw_db.pinRootDelay_ref.data().copy_(timing_raw_db.pinRootDelay.data());
    }
}

void GPUTimer::update_rc_timing_spef() {

    torch::Tensor flat_net2pin_start_map_at = timing_raw_db.flat_net2pin_start_map.clone().cpu().contiguous();
    torch::Tensor flat_net2pin_map_at = timing_raw_db.flat_net2pin_map.clone().cpu().contiguous();
    const int* flat_net2pin_start_map = flat_net2pin_start_map_at.data_ptr<int>();
    const int* flat_net2pin_map = flat_net2pin_map_at.data_ptr<int>();

    vector<vector<int>> net_id2edge_from(num_nets);
    vector<vector<int>> net_id2edge_to(num_nets);
    vector<vector<float>> net_id2node_cap(num_nets);
    vector<vector<float>> net_id2edge_res(num_nets);
    vector<vector<int>> net_id2node2pin_map(num_nets);
    vector<std::unordered_map<std::string, int>> node_name2node_id_map(num_nets);

    printf("num_nets: %d\n", num_nets);
    printf("num_nets in spef file: %d\n", spef.nets.size());
    float spef_res_ratio = *gtdb.spef_res_unit / gtdb.res_unit;
    float spef_cap_ratio = *gtdb.spef_cap_unit / gtdb.cap_unit;
    float spef_time_ratio = *gtdb.spef_time_unit / gtdb.time_unit;
    logger.info("spef lib ratios: res %.5E cap %.5E time %.5E", spef_res_ratio, spef_cap_ratio, spef_time_ratio);

    auto add_node_cap = [&](const std::string& node_name, float cap, int net_idx) {
        if (auto itr = std::find(gtdb.pin_names.begin(), gtdb.pin_names.end(), node_name); itr != gtdb.pin_names.end()) {
            int in_net_idx = flat_net2pin_map[std::distance(gtdb.pin_names.begin(), itr)] - flat_net2pin_start_map[net_idx];
            net_id2node_cap[net_idx][in_net_idx] = cap * spef_cap_ratio;
            node_name2node_id_map[net_idx][node_name] = in_net_idx;
        } else {
            net_id2node_cap[net_idx].push_back(cap * spef_cap_ratio);
            node_name2node_id_map[net_idx][node_name] = net_id2node_cap[net_idx].size() - 1;
            net_id2node2pin_map[net_idx].push_back(-1);
        }
    };
    for (const auto& n : spef.nets) {
        string net_name = n.name;
        net_name = validate_token(net_name);
        if (auto itr = std::find(gtdb.net_names.begin(), gtdb.net_names.end(), net_name); itr == gtdb.net_names.end()) {
            continue;
        } else {
            int net_idx = std::distance(gtdb.net_names.begin(), itr);

            // Put pin nodes in the front
            for (int j = 0; j < flat_net2pin_start_map[net_idx + 1] - flat_net2pin_start_map[net_idx]; ++j) {
                net_id2node2pin_map[net_idx].push_back(flat_net2pin_map[j + flat_net2pin_start_map[net_idx]]); // Pins in the front
            }
            net_id2node_cap[net_idx].resize(net_id2node2pin_map[net_idx].size(), 0);

            // Add ground-node capacitance
            for (const auto& [node1, node2, cap] : n.caps) {
                if (node2.empty()) {
                    add_node_cap(node1, cap, net_idx);
                }
            }

            // Add node-node resistance
            for (const auto& [node1, node2, value] : n.ress) {
                if (node_name2node_id_map[net_idx].find(node1) == node_name2node_id_map[net_idx].end()) {
                    add_node_cap(node1, 0, net_idx);
                }
                if (node_name2node_id_map[net_idx].find(node2) == node_name2node_id_map[net_idx].end()) {
                    add_node_cap(node2, 0, net_idx);
                }
                int from = node_name2node_id_map[net_idx][node1];
                int to = node_name2node_id_map[net_idx][node2];
                net_id2edge_from[net_idx].push_back(from);
                net_id2edge_to[net_idx].push_back(to);
                net_id2edge_res[net_idx].push_back(value * spef_res_ratio);
            }
        }
    }
    for (int i = 0; i < num_nets; ++i) {
        if (net_id2node2pin_map[i].empty()) {
            logger.warning("net %s has no spef rc, assign 0", gtdb.net_names[i].c_str());
            for (int j = 0; j < flat_net2pin_start_map[i + 1] - flat_net2pin_start_map[i]; ++j) {
                net_id2node2pin_map[i].push_back(flat_net2pin_map[j + flat_net2pin_start_map[i]]); // Pins in the front
                net_id2node_cap[i].push_back(0);
                if (j != 0) {
                    net_id2edge_from[i].push_back(0);
                    net_id2edge_to[i].push_back(j);
                    net_id2edge_res[i].push_back(0);
                }
            }
        }
        
    }
    

    // data vectors
    vector<int> edge_from;
    vector<int> edge_to;
    vector<float> node_cap_vec;
    vector<float> edge_res_vec;
    vector<int> flat_net2node_start_map;
    vector<int> flat_net2edge_start_map;
    vector<int> node2pin_map;
    int num_nodes = 0;
    int num_edges = 0;
    flat_net2node_start_map.push_back(0);
    flat_net2edge_start_map.push_back(0);
    for (int i = 0; i < num_nets; ++i) {
        for (int j = 0; j < net_id2edge_from[i].size(); ++j) {
            edge_from.push_back(num_nodes + net_id2edge_from[i][j]);
            edge_to.push_back(num_nodes + net_id2edge_to[i][j]);
            float res = gtdb.net_is_clock[i] == 1 ? 0 : net_id2edge_res[i][j];
            edge_res_vec.push_back(res);
            num_edges++;
        }
        num_nodes += net_id2node2pin_map[i].size();
        for (int j = 0; j < net_id2node_cap[i].size(); ++j) {
            node2pin_map.push_back(net_id2node2pin_map[i][j]);
            float cap = gtdb.net_is_clock[i] == 1 ? 0 : net_id2node_cap[i][j];
            for (int k = 0; k < NUM_ATTR; k++) {
                node_cap_vec.push_back(cap);
            }
        }
        flat_net2node_start_map.push_back(num_nodes);
        flat_net2edge_start_map.push_back(num_edges);
    }

    auto device = timing_raw_db.node_size.device();
    torch::Tensor edge_res = torch::from_blob(edge_res_vec.data(), {num_edges}, torch::dtype(torch::kFloat32)).contiguous().to(device);
    torch::Tensor node_cap = torch::from_blob(node_cap_vec.data(), {num_nodes * NUM_ATTR}, torch::dtype(torch::kFloat32)).contiguous().to(device);
    torch::Tensor node_order = torch::zeros({num_nodes}, torch::kInt32).contiguous().to(device);
    torch::Tensor edge_order = torch::zeros({num_edges}, torch::kInt32).contiguous().to(device);
    torch::Tensor parent_node = -torch::ones({num_nodes}, torch::dtype(torch::kInt32).device(device));
    torch::Tensor res_parent = torch::zeros({num_nodes * NUM_ATTR}, torch::dtype(torch::kFloat32).device(device));

    flatten_rc_tree(edge_from,
                    edge_to,
                    edge_res.data_ptr<float>(),
                    node_cap.data_ptr<float>(),
                    flat_net2node_start_map,
                    flat_net2edge_start_map,
                    node2pin_map,
                    node_order.data_ptr<int>(),
                    edge_order.data_ptr<int>(),
                    parent_node.data_ptr<int>(),
                    res_parent.data_ptr<float>(),
                    pinLoad,
                    pinImpulse,
                    pinCap,
                    pinWireCap,
                    pinRootDelay,
                    pinRootRes,
                    num_nets,
                    num_pins,
                    num_nodes,
                    num_edges);

    propagate_rc_tree(edge_from,
                      edge_to,
                      edge_res.data_ptr<float>(),
                      node_cap.data_ptr<float>(),
                      flat_net2node_start_map,
                      flat_net2edge_start_map,
                      node2pin_map,
                      node_order.data_ptr<int>(),
                      parent_node.data_ptr<int>(),
                      res_parent.data_ptr<float>(),
                      pinLoad,
                      pinImpulse,
                      pinCap,
                      pinWireCap,
                      pinRootDelay,
                      pinRootRes,
                      num_nets,
                      num_pins,
                      num_nodes,
                      num_edges);
}

}  // namespace gt