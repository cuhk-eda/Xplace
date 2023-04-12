#pragma once

#include "common/common.h"
#include "common/db/Database.h"

namespace dp {

class DPTorchRawDB {
public:
    DPTorchRawDB(torch::Tensor node_lpos_init_,
                 torch::Tensor node_size_,
                 torch::Tensor node_weight_,
                 torch::Tensor pin_rel_lpos_,
                 torch::Tensor pin_id2node_id_,
                 torch::Tensor pin_id2net_id_,
                 torch::Tensor node2pin_list_,
                 torch::Tensor node2pin_list_end_,
                 torch::Tensor hyperedge_list_,
                 torch::Tensor hyperedge_list_end_,
                 torch::Tensor net_mask_,
                 torch::Tensor node_id2region_id_,
                 torch::Tensor region_boxes_,
                 torch::Tensor region_boxes_end_,
                 float xl_,
                 float xh_,
                 float yl_,
                 float yh_,
                 int num_movable_nodes_,
                 int num_nodes_,
                 float site_width_,
                 float row_height_);
    bool check(float scale_factor);
    void scale(float scale_factor, bool use_round);
    void commit();
    void rollback();
    void commit_from(torch::Tensor x_, torch::Tensor y_);
    torch::Tensor get_curr_cposx();
    torch::Tensor get_curr_cposy();
    torch::Tensor get_curr_lposx();
    torch::Tensor get_curr_lposy();

public:
    /* node info */
    // for backup
    torch::Tensor node_lpos_init;
    torch::Tensor node_size;
    torch::Tensor pin_rel_lpos;

    torch::Tensor node_weight;

    torch::Tensor init_x;  // original pos (keep it const except committing)
    torch::Tensor init_y;  // original pos (keep it const except committing)
    torch::Tensor x;       // mutable/cached pos (current)
    torch::Tensor y;       // mutable/cached pos (current)
    torch::Tensor node_size_x;
    torch::Tensor node_size_y;

    /* pin info */
    torch::Tensor pin_offset_x;
    torch::Tensor pin_offset_y;

    torch::Tensor flat_node2pin_start_map;
    torch::Tensor flat_node2pin_map;
    torch::Tensor pin2node_map;

    /* net info */
    torch::Tensor flat_net2pin_start_map;
    torch::Tensor flat_net2pin_map;
    torch::Tensor pin2net_map;
    torch::Tensor net_mask;

    /* fence info */
    torch::Tensor flat_region_boxes_start;
    torch::Tensor flat_region_boxes;
    torch::Tensor node2fence_region_map;

    /* chip info */
    float xl;
    float yl;
    float xh;
    float yh;

    /* row info */
    int num_sites_x;
    int num_sites_y;

    int num_pins;
    int num_nets;
    int num_nodes;
    int num_movable_nodes;
    int num_regions;

    float site_width;
    float row_height;

    int num_threads;
};

/* API for python */
// Legalization
bool macroLegalization(DPTorchRawDB& at_db, int num_bins_x, int num_bins_y);
void abacusLegalization(DPTorchRawDB& at_db, int num_bins_x, int num_bins_y);
void greedyLegalization(DPTorchRawDB& at_db, int num_bins_x, int num_bins_y);

// Detailed Placement
void kReorder(DPTorchRawDB& at_db, int num_bins_x, int num_bins_y, int K, int max_iters);
void globalSwap(DPTorchRawDB& at_db, int num_bins_x, int num_bins_y, int batch_size, int max_iters);
void independentSetMatching(
    DPTorchRawDB& at_db, int num_bins_x, int num_bins_y, int batch_size, int set_size, int max_iters);

// Legality Check
bool legalityCheck(DPTorchRawDB& at_db, float scale_factor);

}  // namespace dp