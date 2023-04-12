#include "gpudp/db/dp_torch.h"

#include "common/common.h"
#include "common/db/Database.h"

namespace dp {

DPTorchRawDB::DPTorchRawDB(torch::Tensor node_lpos_init_,
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
                           float row_height_) {
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

    num_nodes = num_nodes_;
    num_pins = pin_id2node_id_.size(0);
    num_nets = hyperedge_list_end_.size(0);
    num_regions = region_boxes_end_.size(0);
    num_movable_nodes = num_movable_nodes_;

    flat_node2pin_start_map =
        torch::cat({torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::Device(node_size.device()))),
                    node2pin_list_end_},
                   0)
            .contiguous();
    flat_node2pin_map = node2pin_list_;
    pin2node_map = pin_id2node_id_;

    flat_net2pin_start_map =
        torch::cat({torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::Device(node_size.device()))),
                    hyperedge_list_end_},
                   0)
            .contiguous();
    flat_net2pin_map = hyperedge_list_;
    pin2net_map = pin_id2net_id_;

    flat_region_boxes_start =
        torch::cat({torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::Device(node_size.device()))),
                    region_boxes_end_},
                   0)
            .contiguous();
    flat_region_boxes = region_boxes_.flatten().contiguous().clone();
    node2fence_region_map = node_id2region_id_;

    net_mask = net_mask_;
    node_weight = node_weight_;

    site_width = site_width_;
    row_height = row_height_;
    xl = xl_;
    xh = xh_;
    yl = yl_;
    yh = yh_;

    num_sites_x = std::round((xh - xl) / site_width);
    num_sites_y = std::round((yh - yl) / row_height);

    num_threads = std::max(db::setting.numThreads, 1);
}

bool DPTorchRawDB::check(float scale_factor) {
    // NOTE: if tensors are on GPU, legalityCheck would copy large data from GPU to CPU
    return legalityCheck(*this, scale_factor);
}

void DPTorchRawDB::scale(float scale_factor, bool use_round) {
    pin_rel_lpos.mul_(scale_factor);
    if (use_round) {
        node_size.mul_(scale_factor).round_();
        node_lpos_init.mul_(scale_factor).round_();
        x.mul_(scale_factor).round_();
        y.mul_(scale_factor).round_();
        flat_region_boxes.mul_(scale_factor).round_();
        site_width = round(site_width * scale_factor);
        row_height = round(row_height * scale_factor);
        xl = round(xl * scale_factor);
        xh = round(xh * scale_factor);
        yl = round(yl * scale_factor);
        yh = round(yh * scale_factor);
    } else {
        node_size.mul_(scale_factor);
        node_lpos_init.mul_(scale_factor);
        x.mul_(scale_factor);
        y.mul_(scale_factor);
        flat_region_boxes.mul_(scale_factor);
        site_width = site_width * scale_factor;
        row_height = row_height * scale_factor;
        xl = xl * scale_factor;
        xh = xh * scale_factor;
        yl = yl * scale_factor;
        yh = yh * scale_factor;
    }
}

void DPTorchRawDB::commit() {
    // commit cached pos to original pos
    init_x.index({torch::indexing::Slice(0, num_movable_nodes)})
        .data()
        .copy_(x.index({torch::indexing::Slice(0, num_movable_nodes)}));
    init_y.index({torch::indexing::Slice(0, num_movable_nodes)})
        .data()
        .copy_(y.index({torch::indexing::Slice(0, num_movable_nodes)}));
}

void DPTorchRawDB::rollback() {
    // rollback cached pos to original pos
    x.index({torch::indexing::Slice(0, num_movable_nodes)})
        .data()
        .copy_(init_x.index({torch::indexing::Slice(0, num_movable_nodes)}));
    y.index({torch::indexing::Slice(0, num_movable_nodes)})
        .data()
        .copy_(init_y.index({torch::indexing::Slice(0, num_movable_nodes)}));
}

void DPTorchRawDB::commit_from(torch::Tensor x_, torch::Tensor y_) {
    // commit external pos to original pos
    init_x.index({torch::indexing::Slice(0, num_movable_nodes)})
        .data()
        .copy_(x_.index({torch::indexing::Slice(0, num_movable_nodes)}));
    init_y.index({torch::indexing::Slice(0, num_movable_nodes)})
        .data()
        .copy_(y_.index({torch::indexing::Slice(0, num_movable_nodes)}));
    x.index({torch::indexing::Slice(0, num_movable_nodes)})
        .data()
        .copy_(x_.index({torch::indexing::Slice(0, num_movable_nodes)}));
    y.index({torch::indexing::Slice(0, num_movable_nodes)})
        .data()
        .copy_(y_.index({torch::indexing::Slice(0, num_movable_nodes)}));
}

torch::Tensor DPTorchRawDB::get_curr_cposx() { return x + node_size_x / 2; }
torch::Tensor DPTorchRawDB::get_curr_cposy() { return y + node_size_y / 2; }
torch::Tensor DPTorchRawDB::get_curr_lposx() { return x; }
torch::Tensor DPTorchRawDB::get_curr_lposy() { return y; }

}  // namespace dp