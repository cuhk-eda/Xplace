#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "common/common.h"
#include "common/db/Database.h"
#include "gpudp/db/dp_torch.h"
#include "pitch_nested_vector.cuh"
#include "utils.cuh"

namespace dp {

inline __host__ __device__ int floorDiv(float a, float b, float rtol = 1e-4) { return floor((a + rtol * b) / b); }

inline __host__ __device__ int ceilDiv(float a, float b, float rtol = 1e-4) { return ceil((a - rtol * b) / b); }

inline __host__ __device__ int roundDiv(float a, float b) { return round(a / b); }

template <typename T>
struct Space {
    T xl;
    T xh;
};

template <typename T>
struct Box {
    T xl;
    T yl;
    T xh;
    T yh;
    __host__ __device__ Box() {
        xl = cuda::numeric_limits<T>::max();
        yl = cuda::numeric_limits<T>::max();
        xh = cuda::numeric_limits<T>::lowest();
        yh = cuda::numeric_limits<T>::lowest();
    }
    __host__ __device__ Box(T xxl, T yyl, T xxh, T yyh) : xl(xxl), yl(yyl), xh(xxh), yh(yyh) {}

    __host__ __device__ T center_x() const { return (xl + xh) / 2; }
    __host__ __device__ T center_y() const { return (yl + yh) / 2; }
};

struct RowMapIndex {
    int row_id;
    int sub_id;
};

struct BinMapIndex {
    int bin_id;
    int sub_id;
};

class DetailedPlaceData {
public:
    DetailedPlaceData() {}
    DetailedPlaceData(DPTorchRawDB& at_db)
        : x(at_db.x.data_ptr<float>()),
          y(at_db.y.data_ptr<float>()),
          init_x(at_db.init_x.data_ptr<float>()),
          init_y(at_db.init_y.data_ptr<float>()),
          node_size_x(at_db.node_size_x.data_ptr<float>()),
          node_size_y(at_db.node_size_y.data_ptr<float>()),
          pin_offset_x(at_db.pin_offset_x.data_ptr<float>()),
          pin_offset_y(at_db.pin_offset_y.data_ptr<float>()),
          flat_node2pin_start_map(at_db.flat_node2pin_start_map.data_ptr<int>()),
          flat_node2pin_map(at_db.flat_node2pin_map.data_ptr<int>()),
          pin2node_map(at_db.pin2node_map.data_ptr<int>()),
          flat_net2pin_start_map(at_db.flat_net2pin_start_map.data_ptr<int>()),
          flat_net2pin_map(at_db.flat_net2pin_map.data_ptr<int>()),
          pin2net_map(at_db.pin2net_map.data_ptr<int>()),
          flat_region_boxes_start(at_db.flat_region_boxes_start.data_ptr<int>()),
          flat_region_boxes(at_db.flat_region_boxes.data_ptr<float>()),
          node2fence_region_map(at_db.node2fence_region_map.data_ptr<int>()),
          net_mask(at_db.net_mask.data_ptr<bool>()),
          node_weight(at_db.node_weight.data_ptr<float>()),
          xl(at_db.xl),
          xh(at_db.xh),
          yl(at_db.yl),
          yh(at_db.yh),
          row_height(at_db.row_height),
          site_width(at_db.site_width),
          num_sites_x(at_db.num_sites_x),
          num_sites_y(at_db.num_sites_y),
          num_threads(at_db.num_threads),
          num_nodes(at_db.num_nodes),
          num_movable_nodes(at_db.num_movable_nodes),
          num_nets(at_db.num_nets),
          num_pins(at_db.num_pins),
          num_regions(at_db.num_regions) {}

public:
    typedef float type;

    float* x;
    float* y;
    const float* init_x;
    const float* init_y;
    const float* node_size_x;
    const float* node_size_y;

    const float* pin_offset_x;
    const float* pin_offset_y;

    const int* flat_node2pin_start_map;
    const int* flat_node2pin_map;
    const int* pin2node_map;

    const int* flat_net2pin_start_map;
    const int* flat_net2pin_map;
    const int* pin2net_map;

    const int* flat_region_boxes_start;
    const float* flat_region_boxes;
    const int* node2fence_region_map;

    const bool* net_mask;
    const float* node_weight;

    /* chip info */
    float xl;
    float yl;
    float xh;
    float yh;

    /* row info */
    int num_sites_x;
    int num_sites_y;
    float row_height;
    float site_width;

    int num_nets;
    int num_movable_nodes;
    int num_nodes;
    int num_pins;
    int num_regions;

    int num_threads;

    int num_bins_x;
    int num_bins_y;
    float bin_size_x;
    float bin_size_y;

public:
    void set_num_bins(int num_bins_x_, int num_bins_y_) {
        num_bins_x = num_bins_x_;
        num_bins_y = num_bins_y_;
        bin_size_x = (xh - xl) / num_bins_x_;
        bin_size_y = (yh - yl) / num_bins_y_;
    }
    inline __device__ int pos2site_x(float xx) const {
        return min(max((int)floorDiv((xx - xl), site_width), 0), num_sites_x - 1);
    }
    inline __device__ int pos2site_y(float yy) const {
        return min(max((int)floorDiv((yy - yl), row_height), 0), num_sites_y - 1);
    }
    inline __device__ int pos2site_ub_x(float xx) const {
        return min(max(ceilDiv((xx - xl), site_width), 1), num_sites_x);
    }
    inline __device__ int pos2site_ub_y(float yy) const {
        return min(max(ceilDiv((yy - yl), row_height), 1), num_sites_y);
    }
    inline __device__ int pos2bin_x(float xx) const {
        int bx = floorDiv((xx - xl), bin_size_x);
        bx = max(bx, 0);
        bx = min(bx, num_bins_x - 1);
        return bx;
    }
    inline __device__ int pos2bin_y(float yy) const {
        int by = floorDiv((yy - yl), bin_size_y);
        by = max(by, 0);
        by = min(by, num_bins_y - 1);
        return by;
    }
    inline __device__ void shift_box_to_layout(Box<float>& box) const {
        box.xl = max(box.xl, xl);
        box.xl = min(box.xl, xh);
        box.xh = max(box.xh, xl);
        box.xh = min(box.xh, xh);
        box.yl = max(box.yl, yl);
        box.yl = min(box.yl, yh);
        box.yh = max(box.yh, yl);
        box.yh = min(box.yh, yh);
    }
    inline __device__ float align2site(float xx) const {
        return (int)floorDiv((xx - xl), site_width) * site_width + xl;
    }
    inline __device__ Space<float> align2site(Space<float> space) const {
        space.xl = ceilDiv((space.xl - xl), site_width) * site_width + xl;
        space.xh = floorDiv((space.xh - xl), site_width) * site_width + xl;
        return space;
    }
    __device__ Box<float> compute_optimal_region(int node_id, const float* xx, const float* yy) const {
        Box<float> box(xh, yh, xl, yl);
        for (int node2pin_id = flat_node2pin_start_map[node_id]; node2pin_id < flat_node2pin_start_map[node_id + 1];
             ++node2pin_id) {
            int node_pin_id = flat_node2pin_map[node2pin_id];
            int net_id = pin2net_map[node_pin_id];
            if (net_mask[net_id]) {
                for (int net2pin_id = flat_net2pin_start_map[net_id]; net2pin_id < flat_net2pin_start_map[net_id + 1];
                     ++net2pin_id) {
                    int net_pin_id = flat_net2pin_map[net2pin_id];
                    int other_node_id = pin2node_map[net_pin_id];
                    if (node_id != other_node_id) {
                        box.xl = min(box.xl, xx[other_node_id] + pin_offset_x[net_pin_id]);
                        box.xh = max(box.xh, xx[other_node_id] + pin_offset_x[net_pin_id]);
                        box.yl = min(box.yl, yy[other_node_id] + pin_offset_y[net_pin_id]);
                        box.yh = max(box.yh, yy[other_node_id] + pin_offset_y[net_pin_id]);
                    }
                }
            }
        }
        shift_box_to_layout(box);

        return box;
    }
    __device__ float compute_net_hpwl(int net_id, const float* xx, const float* yy) const {
        Box<float> box(xh, yh, xl, yl);
        for (int net2pin_id = flat_net2pin_start_map[net_id]; net2pin_id < flat_net2pin_start_map[net_id + 1];
             ++net2pin_id) {
            int net_pin_id = flat_net2pin_map[net2pin_id];
            int other_node_id = pin2node_map[net_pin_id];
            box.xl = min(box.xl, xx[other_node_id] + pin_offset_x[net_pin_id]);
            box.xh = max(box.xh, xx[other_node_id] + pin_offset_x[net_pin_id]);
            box.yl = min(box.yl, yy[other_node_id] + pin_offset_y[net_pin_id]);
            box.yh = max(box.yh, yy[other_node_id] + pin_offset_y[net_pin_id]);
        }
        if (box.xl == xh || box.yl == yh) {
            return (float)0;
        }
        return (box.xh - box.xl) + (box.yh - box.yl);
    }
    // __device__ float compute_total_hpwl() const {
    //     float total_hpwl = 0;
    //     for (int net_id = 0; net_id < num_nets; ++net_id) {
    //         total_hpwl += compute_net_hpwl(net_id, x, y);
    //     }
    //     return total_hpwl;
    // }
    __device__ bool inside_fence(int node_id, float xx, float yy) const {
        float node_xl = xx;
        float node_yl = yy;
        float node_xh = node_xl + node_size_x[node_id];
        float node_yh = node_yl + node_size_y[node_id];

        bool legal_flag = true;
        int region_id = node2fence_region_map[node_id];
        if (region_id < num_regions) {
            int box_bgn = flat_region_boxes_start[region_id];
            int box_end = flat_region_boxes_start[region_id + 1];
            float node_area = (node_xh - node_xl) * (node_yh - node_yl);
            // assume there is no overlap between boxes of a region
            // otherwise, preprocessing is required
            for (int box_id = box_bgn; box_id < box_end; ++box_id) {
                int box_offset = box_id * 4;
                float box_xl = flat_region_boxes[box_offset];
                float box_xh = flat_region_boxes[box_offset + 1];
                float box_yl = flat_region_boxes[box_offset + 2];
                float box_yh = flat_region_boxes[box_offset + 3];

                float dx = max(min(node_xh, box_xh) - max(node_xl, box_xl), (float)0);
                float dy = max(min(node_yh, box_yh) - max(node_yl, box_yl), (float)0);
                float overlap = dx * dy;
                if (overlap > 0) {
                    node_area -= overlap;
                }
            }
            if (node_area > 0) {
                // not consumed by boxes within a region
                legal_flag = false;
            }
        }
        return legal_flag;
    }

    void make_row2node_map(const float* host_x,
                           const float* host_y,
                           const float* host_node_size_x,
                           const float* host_node_size_y,
                           int host_num_nodes,
                           std::vector<std::vector<int>>& row2node_map) {
        // distribute cells to rows
        for (int i = 0; i < host_num_nodes; ++i) {
            float node_yl = host_y[i];
            float node_yh = node_yl + host_node_size_y[i];

            int row_idxl = floorDiv(node_yl - yl, row_height);
            int row_idxh = ceilDiv(node_yh - yl, row_height);
            row_idxl = max(row_idxl, 0);
            row_idxh = min(row_idxh, num_sites_y);

            for (int row_id = row_idxl; row_id < row_idxh; ++row_id) {
                float row_yl = yl + row_id * row_height;
                float row_yh = row_yl + row_height;

                if (node_yl < row_yh && node_yh > row_yl)  // overlap with row
                {
                    row2node_map[row_id].push_back(i);
                }
            }
        }

        // sort cells within rows
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
        for (int i = 0; i < num_sites_y; ++i) {
            auto& row2nodes = row2node_map[i];
            // sort cells within rows according to left edges
            std::sort(row2nodes.begin(), row2nodes.end(), [&](int node_id1, int node_id2) {
                float x1 = host_x[node_id1];
                float x2 = host_x[node_id2];
                return x1 < x2 || (x1 == x2 && node_id1 < node_id2);
            });
            if (!row2nodes.empty()) {
                std::vector<int> tmp_nodes;
                tmp_nodes.reserve(row2nodes.size());
                tmp_nodes.push_back(row2nodes.front());
                for (int j = 1, je = row2nodes.size(); j < je; ++j) {
                    int node_id1 = row2nodes.at(j - 1);
                    int node_id2 = row2nodes.at(j);
                    // two fixed cells
                    if (node_id1 >= num_movable_nodes && node_id2 >= num_movable_nodes) {
                        float xl1 = host_x[node_id1];
                        float xl2 = host_x[node_id2];
                        float width1 = host_node_size_x[node_id1];
                        float width2 = host_node_size_x[node_id2];
                        float xh1 = xl1 + width1;
                        float xh2 = xl2 + width2;
                        // only collect node_id2 if its right edge is righter than node_id1
                        if (xh1 < xh2) {
                            tmp_nodes.push_back(node_id2);
                        }
                    } else {
                        tmp_nodes.push_back(node_id2);
                    }
                }
                row2nodes.swap(tmp_nodes);

                // sort according to center
                std::sort(row2nodes.begin(), row2nodes.end(), [&](int node_id1, int node_id2) {
                    float x1 = host_x[node_id1] + host_node_size_x[node_id1] / 2;
                    float x2 = host_x[node_id2] + host_node_size_x[node_id2] / 2;
                    return x1 < x2 || (x1 == x2 && node_id1 < node_id2);
                });
            }
        }
    }

    void make_row2node_map_with_spaces(const float* host_x,
                                       const float* host_y,
                                       const float* host_node_size_x,
                                       const float* host_node_size_y,
                                       std::vector<std::vector<int>>& row2node_map,
                                       std::vector<RowMapIndex>& node2row_map,
                                       std::vector<Space<float>>& spaces) {
        make_row2node_map(host_x, host_y, host_node_size_x, host_node_size_y, num_nodes + 2, row2node_map);

        // construct node2row_map
        for (int i = 0; i < num_sites_y; ++i) {
            for (unsigned int j = 0; j < row2node_map[i].size(); ++j) {
                int node_id = row2node_map[i][j];
                if (node_id < num_movable_nodes) {
                    RowMapIndex& row_id = node2row_map[node_id];
                    row_id.row_id = i;
                    row_id.sub_id = j;
                }
            }
        }

        // construct spaces
        for (int i = 0; i < num_sites_y; ++i) {
            for (unsigned int j = 0; j < row2node_map[i].size(); ++j) {
                int node_id = row2node_map[i][j];
                if (node_id < num_movable_nodes) {
                    assert(j);
                    int left_node_id = row2node_map[i][j - 1];
                    spaces[node_id].xl = host_x[left_node_id] + host_node_size_x[left_node_id];
                    assert(j + 1 < row2node_map[i].size());
                    int right_node_id = row2node_map[i][j + 1];
                    spaces[node_id].xh = host_x[right_node_id];
                }
            }
        }
    }

    void make_bin2node_map(const float* host_x,
                           const float* host_y,
                           const float* host_node_size_x,
                           const float* host_node_size_y,
                           std::vector<std::vector<int>>& bin2node_map,
                           std::vector<BinMapIndex>& node2bin_map) {
        // construct bin2node_map
        for (int i = 0; i < num_movable_nodes; ++i) {
            int node_id = i;
            float node_x = host_x[node_id] + host_node_size_x[node_id] / 2;
            float node_y = host_y[node_id] + host_node_size_y[node_id] / 2;

            int bx = min(max((int)floorDiv(node_x - xl, bin_size_x), 0), num_bins_x - 1);
            int by = min(max((int)floorDiv(node_y - yl, bin_size_y), 0), num_bins_y - 1);
            int bin_id = bx * num_bins_y + by;
            int sub_id = bin2node_map.at(bin_id).size();
            bin2node_map.at(bin_id).push_back(node_id);
        }
        for (int bin_id = 0; bin_id < bin2node_map.size(); ++bin_id) {
            for (int sub_id = 0; sub_id < bin2node_map[bin_id].size(); ++sub_id) {
                int node_id = bin2node_map[bin_id][sub_id];
                BinMapIndex& bm_idx = node2bin_map.at(node_id);
                bm_idx.bin_id = bin_id;
                bm_idx.sub_id = sub_id;
            }
        }
    }
};

float compute_total_hpwl(const DetailedPlaceData& db, const float* xx, const float* yy, double* net_hpwls);

}  // namespace dp