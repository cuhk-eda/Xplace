#include "RouteForce.h"

namespace gr {

RouteForce::RouteForce(std::shared_ptr<gr::GRDatabase> grdb_) : grdb(*grdb_) {}

void RouteForce::run_ggr() {
    logger.enable_logger();
    utils::timer T_total;
    T_total.start();
    // we only need the PR segment, our current data structure unsupport MR route force
    // if rrrIters > 0, this router can only be used for congestion map computation or solution evaluation
    int runMazeRouteTimes = grSetting.rrrIters;
    // Parameters
    int rrrIterLimit = 1 + runMazeRouteTimes;
    double _unitWireCostRaw = 0.5 * grdb.microns / grdb.m2pitch;
    double _unitViaCostRaw = 4;
    double _unitViaCost = _unitViaCostRaw / _unitWireCostRaw * grdb.microns;
    double _unitShortVioCostRaw = 500;
    double rrrInitVioCostDiscount = 0.1;

    router.initialize(grSetting.deviceId,
                      grdb.nLayers,
                      grdb.xSize,
                      grdb.ySize,
                      grdb.nMaxGrid,
                      grdb.cgxsize,
                      grdb.cgysize,
                      grdb.m1direction,
                      grdb.csrnScale);
    router.setMap(grdb.capacity, grdb.wireDist, grdb.fixedLength, grdb.fixedUsage);

    std::vector<float> _unitShortVioCost(grdb.nLayers), _unitShortVioCostDiscounted(grdb.nLayers);
    router.setFromNets(grdb.grNets, grdb.gpdb.getPins().size());
    router.setUnitViaCost(_unitViaCost);
    for (int i = 0; i < grdb.nLayers; ++i) {
        _unitShortVioCost[i] =
            _unitShortVioCostRaw * grdb.layerWidth[i] * grdb.microns / grdb.m2pitch / grdb.m2pitch / _unitWireCostRaw;
    }
    double tot_time = 0;
    for (int iter = 0; iter < rrrIterLimit; iter++) {
        router.setLogisticSlope(1 << iter);
        router.setUnitVioCost(_unitShortVioCost, 0.1);
        if (iter == 0) {
            router.setUnitViaMultiplier(1);
        } else {
            router.setUnitViaMultiplier(max(100 / pow(5, iter - 1), 4.0));
            router.setUnitVioCost(_unitShortVioCost,
                                  rrrInitVioCostDiscount + (1.0 - rrrInitVioCostDiscount) / (rrrIterLimit - 1) * iter);
        }
        utils::timer T;
        T.start();
        router.route(grdb.grNets, iter);
        tot_time += T.elapsed();
        logger.info("##### GPU Routing Iter: %d Time: %.4f #####", iter, T.elapsed());
        // break;
    }
    router.setToNets(grdb.grNets);
    logger.info("Total GPU Routing time: %.4f", tot_time);

    if (grSetting.routeGuideFile != "") {
        grdb.writeGuides(grSetting.routeGuideFile);
    }

    logger.info("Total GPU GR Time: %.4f", T_total.elapsed());
    logger.reset_logger();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RouteForce::getDemandMap() { return router.getDemandMap(); }

torch::Tensor RouteForce::getCapacityMap() { return router.getCapacityMap(); }

torch::Tensor RouteForce::calcRouteGrad(torch::Tensor mask_map,
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
                                        int num_nodes) {
    return router.calcRouteGrad(mask_map,
                                wire_dmd_map_2d,
                                via_dmd_map_2d,
                                cap_map_2d,
                                dist_weights,
                                wirelength_weights,
                                route_gradmat,
                                node2pin_list,
                                node2pin_list_end,
                                grad_weight,
                                unit_wire_cost,
                                unit_via_cost,
                                num_nodes);
};

torch::Tensor RouteForce::calcFillerRouteGrad(torch::Tensor filler_pos,
                                              torch::Tensor filler_size,
                                              torch::Tensor filler_weight,
                                              torch::Tensor expand_ratio,
                                              torch::Tensor grad_mat,
                                              float grad_weight,
                                              float unit_len_x,
                                              float unit_len_y,
                                              int num_bin_x,
                                              int num_bin_y,
                                              int num_fillers) {
    return router.calcFillerRouteGrad(filler_pos,
                                      filler_size,
                                      filler_weight,
                                      expand_ratio,
                                      grad_mat,
                                      grad_weight,
                                      unit_len_x,
                                      unit_len_y,
                                      num_bin_x,
                                      num_bin_y,
                                      num_fillers);
}

torch::Tensor RouteForce::calcPseudoPinGrad(torch::Tensor node_pos, torch::Tensor pseudo_pin_pos, float gamma) {
    return router.calcPseudoPinGrad(node_pos, pseudo_pin_pos, gamma);
}

torch::Tensor RouteForce::calcNodeInflateRatio(torch::Tensor node_pos,
                                               torch::Tensor node_size,
                                               torch::Tensor node_weight,
                                               torch::Tensor expand_ratio,
                                               torch::Tensor inflate_mat,
                                               float grad_weight,
                                               float unit_len_x,
                                               float unit_len_y,
                                               int num_bin_x,
                                               int num_bin_y,
                                               bool use_weighted_inflation) {
    return router.calcNodeInflateRatio(node_pos,
                                       node_size,
                                       node_weight,
                                       expand_ratio,
                                       inflate_mat,
                                       grad_weight,
                                       unit_len_x,
                                       unit_len_y,
                                       num_bin_x,
                                       num_bin_y,
                                       use_weighted_inflation);
}

torch::Tensor RouteForce::calcInflatedPinRelCpos(torch::Tensor node_inflate_ratio,
                                                 torch::Tensor old_pin_rel_cpos,
                                                 torch::Tensor pin_id2node_id,
                                                 int num_movable_conn_nodes) {
    return router.calcInflatedPinRelCpos(node_inflate_ratio, old_pin_rel_cpos, pin_id2node_id, num_movable_conn_nodes);
}

int RouteForce::getNumOvflNets() { return router.getNumOvflNets(); }

int RouteForce::getMicrons() { return grdb.microns; }

std::tuple<int, int> RouteForce::getGcellStep() { return {grdb.mainGcellStepX, grdb.mainGcellStepY}; }

std::vector<int> RouteForce::getLayerPitch() { return grdb.layerPitch; }

std::vector<int> RouteForce::getLayerWidth() { return grdb.layerWidth; }

}  // namespace gr