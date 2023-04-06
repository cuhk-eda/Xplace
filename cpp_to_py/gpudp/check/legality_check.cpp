#include "common/common.h"
#include "gpudp/db/dp_torch.h"
#include "gpudp/lg/legalization_db.h"

namespace dp {

bool boundaryCheck(const float* x,
                   const float* y,
                   const float* node_size_x,
                   const float* node_size_y,
                   const float scale_factor,
                   float xl,
                   float yl,
                   float xh,
                   float yh,
                   int num_movable_nodes) {
    // use scale factor to control the precision
    float precision = (scale_factor == 1.0) ? 1e-6 : scale_factor * 0.1;
    bool legal_flag = true;
    // check node within boundary
    for (int i = 0; i < num_movable_nodes; ++i) {
        float node_xl = x[i];
        float node_yl = y[i];
        float node_xh = node_xl + node_size_x[i];
        float node_yh = node_yl + node_size_y[i];
        if (node_xl + precision < xl || node_xh > xh + precision || node_yl + precision < yl ||
            node_yh > yh + precision) {
            logger.error("node %d (%g, %g, %g, %g) out of boundary\n", i, node_xl, node_yl, node_xh, node_yh);
            legal_flag = false;
        }
    }
    return legal_flag;
}

bool siteAlignmentCheck(const float* x,
                        const float* y,
                        const float site_width,
                        const float row_height,
                        const float scale_factor,
                        const float xl,
                        const float yl,
                        int num_movable_nodes) {
    // use scale factor to control the precision
    float precision = (scale_factor == 1.0) ? 1e-6 : scale_factor * 0.1;
    bool legal_flag = true;
    // check row and site alignment
    for (int i = 0; i < num_movable_nodes; ++i) {
        float node_xl = x[i];
        float node_yl = y[i];

        float row_id_f = (node_yl - yl) / row_height;
        int row_id = floorDiv(node_yl - yl, row_height);
        float row_yl = yl + row_height * row_id;
        float row_yh = row_yl + row_height;

        if (std::abs(row_id_f - row_id) > precision) {
            logger.error("node %d (%g, %g) failed to align to row %d (%g, %g), gap %g, yl %g, row_height %g",
                         i,
                         node_xl,
                         node_yl,
                         row_id,
                         row_yl,
                         row_yh,
                         std::abs(node_yl - row_yl),
                         yl,
                         row_height);
            legal_flag = false;
        }

        float site_id_f = (node_xl - xl) / site_width;
        int site_id = floorDiv(node_xl - xl, site_width);
        if (std::abs(site_id_f - site_id) > precision) {
            logger.error("node %d (%g, %g) failed to align to row %d (%g, %g) and site; xl %g, site_width %g",
                         i,
                         node_xl,
                         node_yl,
                         row_id,
                         row_yl,
                         row_yh,
                         xl,
                         site_width);
            legal_flag = false;
        }
    }

    return legal_flag;
}

bool fenceRegionCheck(const float* x,
                      const float* y,
                      const float* node_size_x,
                      const float* node_size_y,
                      const float* flat_region_boxes,
                      const int* flat_region_boxes_start,
                      const int* node2fence_region_map,
                      int num_movable_nodes,
                      int num_regions) {
    bool legal_flag = true;
    // check fence regions
    for (int i = 0; i < num_movable_nodes; ++i) {
        float node_xl = x[i];
        float node_yl = y[i];
        float node_xh = node_xl + node_size_x[i];
        float node_yh = node_yl + node_size_y[i];

        int region_id = node2fence_region_map[i];
        if (region_id < num_regions) {
            int box_bgn = flat_region_boxes_start[region_id];
            int box_end = flat_region_boxes_start[region_id + 1];
            float node_area = (node_xh - node_xl) * (node_yh - node_yl);
            // I assume there is no overlap between boxes of a region
            // otherwise, preprocessing is required
            for (int box_id = box_bgn; box_id < box_end; ++box_id) {
                int box_offset = box_id * 4;
                float box_xl = flat_region_boxes[box_offset];
                float box_xh = flat_region_boxes[box_offset + 1];
                float box_yl = flat_region_boxes[box_offset + 2];
                float box_yh = flat_region_boxes[box_offset + 3];

                float dx = std::max(std::min(node_xh, box_xh) - std::max(node_xl, box_xl), (float)0);
                float dy = std::max(std::min(node_yh, box_yh) - std::max(node_yl, box_yl), (float)0);
                float overlap = dx * dy;
                if (overlap > 0) {
                    node_area -= overlap;
                }
            }
            if (node_area > 0) {  // not consumed by boxes within a region
                std::string fence_str = "";
                for (int box_id = box_bgn; box_id < box_end; ++box_id) {
                    int box_offset = box_id * 4;
                    float box_xl = flat_region_boxes[box_offset];
                    float box_xh = flat_region_boxes[box_offset + 1];
                    float box_yl = flat_region_boxes[box_offset + 2];
                    float box_yh = flat_region_boxes[box_offset + 3];
                    fence_str += (" (" + std::to_string(box_xl) + ", " + std::to_string(box_yl) + ", " +
                                  std::to_string(box_xh) + ", " + std::to_string(box_yh) + ")");
                }
                logger.error("node %d (%g, %g, %g, %g), out of fence region %d: %s",
                             i,
                             node_xl,
                             node_yl,
                             node_xh,
                             node_yh,
                             region_id,
                             fence_str.c_str());
                legal_flag = false;
            }
        }
    }
    return legal_flag;
}

bool overlapCheck(const float* x,
                  const float* y,
                  const float* node_size_x,
                  const float* node_size_y,
                  float site_width,
                  float row_height,
                  float scale_factor,
                  float xl,
                  float yl,
                  float xh,
                  float yh,
                  int num_nodes,
                  int num_movable_nodes) {
    bool legal_flag = true;
    int num_rows = ceilDiv(yh - yl, row_height);
    assert(num_rows > 0);
    std::vector<std::vector<int> > row_nodes(num_rows);

    // general to node and fixed boxes
    auto getXL = [&](int id) { return x[id]; };
    auto getYL = [&](int id) { return y[id]; };
    auto getXH = [&](int id) { return x[id] + node_size_x[id]; };
    auto getYH = [&](int id) { return y[id] + node_size_y[id]; };

    auto getSiteXL = [&](float xx) { return int(floorDiv(xx - xl, site_width)); };
    auto getSiteYL = [&](float yy) { return int(floorDiv(yy - yl, row_height)); };
    auto getSiteXH = [&](float xx) { return int(ceilDiv(xx - xl, site_width)); };
    auto getSiteYH = [&](float yy) { return int(ceilDiv(yy - yl, row_height)); };

    // add a box to row
    auto addBox2Row = [&](int id, float bxl, float byl, float bxh, float byh) {
        int row_idxl = floorDiv(byl - yl, row_height);
        int row_idxh = ceilDiv(byh - yl, row_height);
        row_idxl = std::max(row_idxl, 0);
        row_idxh = std::min(row_idxh, num_rows);

        for (int row_id = row_idxl; row_id < row_idxh; ++row_id) {
            float row_yl = yl + row_id * row_height;
            float row_yh = row_yl + row_height;

            if (byl < row_yh && byh > row_yl)  // overlap with row
            {
                row_nodes[row_id].push_back(id);
            }
        }
    };
    // distribute movable cells to rows
    for (int i = 0; i < num_nodes; ++i) {
        float node_xl = x[i];
        float node_yl = y[i];
        float node_xh = node_xl + node_size_x[i];
        float node_yh = node_yl + node_size_y[i];

        addBox2Row(i, node_xl, node_yl, node_xh, node_yh);
    }

    // sort cells within rows
    for (int i = 0; i < num_rows; ++i) {
        auto& nodes_in_row = row_nodes.at(i);
        // using left edge
        std::sort(nodes_in_row.begin(), nodes_in_row.end(), [&](int node_id1, int node_id2) {
            float x1 = getXL(node_id1);
            float x2 = getXL(node_id2);
            return x1 < x2 || (x1 == x2 && (node_id1 < node_id2));
        });
        // After sorting by left edge,
        // there is a special case for fixed cells where
        // one fixed cell is completely within another in a row.
        // This will cause failure to detect some overlaps.
        // We need to remove the "small" fixed cell that is inside another.
        if (!nodes_in_row.empty()) {
            std::vector<int> tmp_nodes;
            tmp_nodes.reserve(nodes_in_row.size());
            tmp_nodes.push_back(nodes_in_row.front());
            for (int j = 1, je = nodes_in_row.size(); j < je; ++j) {
                int node_id1 = nodes_in_row.at(j - 1);
                int node_id2 = nodes_in_row.at(j);
                // two fixed cells
                if (node_id1 >= num_movable_nodes && node_id2 >= num_movable_nodes) {
                    float xh1 = getXH(node_id1);
                    float xh2 = getXH(node_id2);
                    if (xh1 < xh2) {
                        tmp_nodes.push_back(node_id2);
                    }
                } else {
                    tmp_nodes.push_back(node_id2);
                }
            }
            nodes_in_row.swap(tmp_nodes);
        }
    }

    // check overlap
    // use scale factor to control the precision
    // auto scaleBack2Integer = [&](float value) {
    //     return (scale_factor == 1.0) ? value : std::round(value / scale_factor);
    // };
    for (int i = 0; i < num_rows; ++i) {
        for (unsigned int j = 0; j < row_nodes.at(i).size(); ++j) {
            if (j > 0) {
                int node_id = row_nodes[i][j];
                int prev_node_id = row_nodes[i][j - 1];

                if (node_id < num_movable_nodes || prev_node_id < num_movable_nodes)  // ignore two fixed nodes
                {
                    float prev_xl = getXL(prev_node_id);
                    float prev_yl = getYL(prev_node_id);
                    float prev_xh = getXH(prev_node_id);
                    float prev_yh = getYH(prev_node_id);
                    float cur_xl = getXL(node_id);
                    float cur_yl = getYL(node_id);
                    float cur_xh = getXH(node_id);
                    float cur_yh = getYH(node_id);
                    int prev_site_xl = getSiteXL(prev_xl);
                    int prev_site_xh = getSiteXH(prev_xh);
                    int cur_site_xl = getSiteXL(cur_xl);
                    int cur_site_xh = getSiteXH(cur_xh);
                    // detect overlap
                    if (prev_site_xh > cur_site_xl) {
                        logger.error(
                            "row %d (%g, %g), overlap node %d (%g, %g, %g, %g) with "
                            "node %d (%g, %g, %g, %g) site (%d, %d), gap %g",
                            i,
                            yl + i * row_height,
                            yl + (i + 1) * row_height,
                            prev_node_id,
                            prev_xl,
                            prev_yl,
                            prev_xh,
                            prev_yh,
                            node_id,
                            cur_xl,
                            cur_yl,
                            cur_xh,
                            cur_yh,
                            cur_site_xl,
                            cur_site_xh,
                            prev_xh - cur_xl);
                        legal_flag = false;
                    }
                }
            }
        }
    }

    return legal_flag;
}

bool legalityCheckKernelCPU(const float* x,
                            const float* y,
                            const float* node_size_x,
                            const float* node_size_y,
                            const float* flat_region_boxes,
                            const int* flat_region_boxes_start,
                            const int* node2fence_region_map,
                            float xl,
                            float yl,
                            float xh,
                            float yh,
                            float site_width,
                            float row_height,
                            int num_nodes,
                            int num_movable_nodes,
                            int num_regions,
                            float scale_factor) {
    bool legal_flag = true;
    int num_rows = ceil((yh - yl) / row_height);
    assert(num_rows > 0);
    fflush(stdout);
    std::vector<std::vector<int> > row_nodes(num_rows);

    // check node within boundary
    if (!boundaryCheck(x, y, node_size_x, node_size_y, scale_factor, xl, yl, xh, yh, num_movable_nodes)) {
        legal_flag = false;
        std::cerr << "boundary check error!" << std::endl;
    }

    // check row and site alignment
    if (!siteAlignmentCheck(x, y, site_width, row_height, scale_factor, xl, yl, num_movable_nodes)) {
        legal_flag = false;
        std::cerr << "site alignment check error!" << std::endl;
    }

    if (!overlapCheck(
            x, y, node_size_x, node_size_y, site_width, row_height, scale_factor, xl, yl, xh, yh, num_nodes, num_movable_nodes)) {
        legal_flag = false;
        std::cerr << "overlap check error!" << std::endl;
    }

    // check fence regions
    if (!fenceRegionCheck(x,
                          y,
                          node_size_x,
                          node_size_y,
                          flat_region_boxes,
                          flat_region_boxes_start,
                          node2fence_region_map,
                          num_movable_nodes,
                          num_regions)) {
        legal_flag = false;
        std::cerr << "fence region check error!" << std::endl;
    }

    if (!legal_flag) {
        logger.error("placement legality check error!");
    }

    return legal_flag;
}

bool legalityCheck(DPTorchRawDB& at_db, float scale_factor) {
    return legalityCheckKernelCPU(at_db.x.cpu().data_ptr<float>(),
                                  at_db.y.cpu().data_ptr<float>(),
                                  at_db.node_size_x.cpu().data_ptr<float>(),
                                  at_db.node_size_y.cpu().data_ptr<float>(),
                                  at_db.flat_region_boxes.cpu().data_ptr<float>(),
                                  at_db.flat_region_boxes_start.cpu().data_ptr<int>(),
                                  at_db.node2fence_region_map.cpu().data_ptr<int>(),
                                  at_db.xl,
                                  at_db.yl,
                                  at_db.xh,
                                  at_db.yh,
                                  at_db.site_width,
                                  at_db.row_height,
                                  at_db.num_nodes,
                                  at_db.num_movable_nodes,
                                  at_db.num_regions,
                                  scale_factor);
}

}  // namespace dp