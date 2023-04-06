#pragma once
#include "MazeRoute.h"
#include "PatternRoute.h"
#include "common/common.h"
#include "gpugr/db/GrNet.h"

namespace gr {

typedef int dtype;

class GPURouter {
public:
    GPURouter(){};
    GPURouter(
        int device_id, int layer, int x, int y, int N_, int cgxsize_, int cgysize_, int direction, int csrn_scale) {
        initialize(device_id, layer, x, y, N_, cgxsize_, cgysize_, direction, csrn_scale);
    }
    ~GPURouter();

    void initialize(
        int device_id, int layer, int x, int y, int N_, int cgxsize_, int cgysize_, int direction, int csrn_scale);

    void setMap(const vector<float> &cap,
                const vector<float> &wir,
                const vector<float> &fixedL,
                const vector<float> &fix);
    void setFromNets(vector<GrNet> &nets, int numPlPin_);
    void setToNets(vector<GrNet> &nets);
    void route(vector<GrNet> &nets, int iterleft);
    void setUnitViaMultiplier(float w);
    void setUnitVioCost(vector<float>& cost, float discount);
    void setLogisticSlope(float value);
    void setUnitViaCost(float value);
    void query();

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
    int getNumOvflNets() { return numOvflNets; }

private:
    GPUMazeRouter gpuMR;
    // routes:
    //    (x, y): starting point x, length |y|; negative y implies vias

    int DEVICE_ID;
    int LAYER, N, X, Y, NET_NUM, DIRECTION;
    int COARSENING_SCALE;
    int cgxsize, cgysize;

    int *pinNum = nullptr, *pinNumOffset = nullptr, *pins = nullptr;
    int *routes = nullptr, *routesOffset = nullptr, *routesOffsetCPU = nullptr, *pinNumCPU = nullptr;
    int *allpins;
    int *points = nullptr, *gbpoints = nullptr;
    int *gbpinRoutes = nullptr, *gbpin2netId = nullptr, *plPinId2gbPinId = nullptr;
    float *capacity, *wireDist, *fixedLength, *fixed;
    int *wires, *vias, *prev;
    int *isOverflowWire, *isOverflowVia, *isOverflowNet = nullptr;
    int *boundaries, *isLocked;
    int *wiresCPU, *viasCPU;
    int *cudaIndex, *cudaCostIndex;
    int *modifiedVia, *modifiedWire, *viasToBeUpdated, *wiresToBeUpdated;
    dtype *dist, *cost, *viaCost;
    int64_t *costSum;
    float *unitShortCostDiscounted, unitViaCost, unitViaMultiplier = 1, logisticSlope = 1, *cell_resource;

    int numGbPin, numPlPin;

    int numOvflNets = 0;

    const int MAX_BATCH_SIZE = 100, MAX_PIN_SIZE_PER_NET = 500000;

    std::vector<std::vector<short>> vis, visLL, visRR;
};

}  // namespace gr
