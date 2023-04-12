#pragma once
#include "common/common.h"
#include "common/db/Database.h"
#include "gpugr/db/GRDatabase.h"
#include "gpugr/gr/GPURouter.h"

namespace gr {

class RouteForce {
public:
    RouteForce(std::shared_ptr<gr::GRDatabase> grdb_);
    void run_ggr();

public:
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> getDemandMap();
    torch::Tensor getCapacityMap();
    torch::Tensor calcRouteGrad(torch::Tensor mask_map,
                                torch::Tensor wire_dmd_map_2d,
                                torch::Tensor via_dmd_map_2d,
                                torch::Tensor cap_map_2d,
                                torch::Tensor dist_weights,
                                torch::Tensor wirelength_weights,
                                torch::Tensor route_gradmat,
                                torch::Tensor node2pin_list,
                                torch::Tensor node2pin_list_end,
                                float grad_weight,
                                float unit_wire_cost,
                                float unit_via_cost,
                                int num_nodes);
    torch::Tensor calcFillerRouteGrad(torch::Tensor filler_pos,
                                      torch::Tensor filler_size,
                                      torch::Tensor filler_weight,
                                      torch::Tensor expand_ratio,
                                      torch::Tensor grad_mat,
                                      float grad_weight,
                                      float unit_len_x,
                                      float unit_len_y,
                                      int num_bin_x,
                                      int num_bin_y,
                                      int num_fillers);
    torch::Tensor calcPseudoPinGrad(torch::Tensor node_pos, torch::Tensor pseudo_pin_pos, float gamma);
    torch::Tensor calcNodeInflateRatio(torch::Tensor node_pos,
                                       torch::Tensor node_size,
                                       torch::Tensor node_weight,
                                       torch::Tensor expand_ratio,
                                       torch::Tensor inflate_mat,
                                       float grad_weight,
                                       float unit_len_x,
                                       float unit_len_y,
                                       int num_bin_x,
                                       int num_bin_y,
                                       bool use_weighted_inflation);
    torch::Tensor calcInflatedPinRelCpos(torch::Tensor node_inflate_ratio,
                                         torch::Tensor old_pin_rel_cpos,
                                         torch::Tensor pin_id2node_id,
                                         int num_movable_nodes);
    int getNumOvflNets();
    int getMicrons();
    std::tuple<int, int> getGcellStep();
    std::vector<int> getLayerPitch();
    std::vector<int> getLayerWidth();

private:
    gr::GRDatabase& grdb;
    gr::GPURouter router;
};

}  // namespace gr