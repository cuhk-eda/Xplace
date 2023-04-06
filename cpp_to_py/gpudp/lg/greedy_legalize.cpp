#include "gpudp/lg/legalization_db.h"

namespace dp {

template <typename T>
struct Interval {
    T xl;
    T xh;

    Interval(T l, T h) : xl(l), xh(h) {}

    void intersect(T rhs_xl, T rhs_xh) {
        xl = std::max(xl, rhs_xl);
        xh = std::min(xh, rhs_xh);
    }
};

template <typename T>
struct Blank {
    T xl;
    T yl;
    T xh;
    T yh;

    void intersect(const Blank& rhs) {
        xl = std::max(xl, rhs.xl);
        xh = std::min(xh, rhs.xh);
        yl = std::max(yl, rhs.yl);
        yh = std::min(yh, rhs.yh);
    }
};

void distributeCells2Bins(const LegalizationData& db,
                          const float* x,
                          const float* y,
                          const float* node_size_x,
                          const float* node_size_y,
                          float bin_size_x,
                          float bin_size_y,
                          float xl,
                          float yl,
                          float xh,
                          float yh,
                          int num_bins_x,
                          int num_bins_y,
                          int num_nodes,
                          int num_movable_nodes,
                          std::vector<std::vector<int>>& bin_cells) {
    // do not handle large macros
    // one cell cannot be distributed to one bin
    for (int i = 0; i < num_movable_nodes; i += 1) {
        if (!db.is_dummy_fixed(i)) {
            int bin_id_x = (x[i] + node_size_x[i] / 2 - xl) / bin_size_x;
            int bin_id_y = (y[i] + node_size_y[i] / 2 - yl) / bin_size_y;

            bin_id_x = std::min(std::max(bin_id_x, 0), num_bins_x - 1);
            bin_id_y = std::min(std::max(bin_id_y, 0), num_bins_y - 1);

            int bin_id = bin_id_x * num_bins_y + bin_id_y;

            bin_cells[bin_id].push_back(i);
        }
    }
}

void distributeFixedCells2Bins(const LegalizationData& db,
                               const float* x,
                               const float* y,
                               const float* node_size_x,
                               const float* node_size_y,
                               float bin_size_x,
                               float bin_size_y,
                               float xl,
                               float yl,
                               float xh,
                               float yh,
                               int num_bins_x,
                               int num_bins_y,
                               int num_nodes,
                               int num_movable_nodes,
                               std::vector<std::vector<int>>& bin_cells) {
    // one cell can be assigned to multiple bins
    for (int i = 0; i < num_nodes; i += 1) {
        if (db.is_dummy_fixed(i) || i >= num_movable_nodes) {
            int node_id = i;
            int bin_id_xl = std::max((int)floorDiv(x[node_id] - xl, bin_size_x), 0);
            int bin_id_xh = std::min((int)ceilDiv((x[node_id] + node_size_x[node_id] - xl), bin_size_x), num_bins_x);
            int bin_id_yl = std::max((int)floorDiv(y[node_id] - yl, bin_size_y), 0);
            int bin_id_yh = std::min((int)ceilDiv((y[node_id] + node_size_y[node_id] - yl), bin_size_y), num_bins_y);

            for (int bin_id_x = bin_id_xl; bin_id_x < bin_id_xh; ++bin_id_x) {
                for (int bin_id_y = bin_id_yl; bin_id_y < bin_id_yh; ++bin_id_y) {
                    int bin_id = bin_id_x * num_bins_y + bin_id_y;

                    bin_cells[bin_id].push_back(node_id);
                }
            }
        }
    }
}

void distributeBlanks2Bins(const float* x,
                           const float* y,
                           const float* node_size_x,
                           const float* node_size_y,
                           const std::vector<std::vector<int>>& bin_fixed_cells,
                           float bin_size_x,
                           float bin_size_y,
                           float blank_bin_size_y,
                           float xl,
                           float yl,
                           float xh,
                           float yh,
                           float site_width,
                           float row_height,
                           int num_bins_x,
                           int num_bins_y,
                           int blank_num_bins_y,
                           std::vector<std::vector<Blank<float>>>& bin_blanks) {
    for (int i = 0; i < num_bins_x * num_bins_y; i += 1) {
        int bin_id_x = i / num_bins_y;
        int bin_id_y = i - bin_id_x * num_bins_y;
        int blank_num_bins_per_bin = roundDiv(bin_size_y, blank_bin_size_y);
        int blank_bin_id_yl = bin_id_y * blank_num_bins_per_bin;
        int blank_bin_id_yh = std::min(blank_bin_id_yl + blank_num_bins_per_bin, blank_num_bins_y);
        for (int blank_bin_id_y = blank_bin_id_yl; blank_bin_id_y < blank_bin_id_yh; ++blank_bin_id_y) {
            float bin_xl = xl + bin_id_x * bin_size_x;
            float bin_xh = std::min(bin_xl + bin_size_x, xh);
            float bin_yl = yl + blank_bin_id_y * blank_bin_size_y;
            float bin_yh = std::min(bin_yl + blank_bin_size_y, yh);
            int blank_bin_id = bin_id_x * blank_num_bins_y + blank_bin_id_y;

            for (float by = bin_yl; by < bin_yh; by += row_height) {
                Blank<float> blank;
                blank.xl = floorDiv((bin_xl - xl), site_width) * site_width + xl;  // align blanks to sites
                blank.xh = floorDiv((bin_xh - xl), site_width) * site_width + xl;  // align blanks to sites
                blank.yl = by;
                blank.yh = by + row_height;

                bin_blanks.at(blank_bin_id).push_back(blank);
            }

            const std::vector<int>& cells = bin_fixed_cells.at(i);
            std::vector<Blank<float>>& blanks = bin_blanks.at(blank_bin_id);

            for (unsigned int bi = 0; bi < blanks.size(); ++bi) {
                Blank<float>& blank = blanks.at(bi);
                for (unsigned int ci = 0; ci < cells.size(); ++ci) {
                    int node_id = cells.at(ci);
                    float node_xl = x[node_id];
                    float node_yl = y[node_id];
                    float node_xh = node_xl + node_size_x[node_id];
                    float node_yh = node_yl + node_size_y[node_id];

                    if (node_yh > blank.yl && node_yl < blank.yh && node_xh > blank.xl &&
                        node_xl < blank.xh)  // overlap
                    {
                        if (node_xl <= blank.xl && node_xh >= blank.xh)  // erase
                        {
                            bin_blanks.at(blank_bin_id).erase(bin_blanks.at(blank_bin_id).begin() + bi);
                            --bi;
                            break;
                        } else if (node_xl <= blank.xl) {                                       // one blank
                            blank.xl = ceilDiv((node_xh - xl), site_width) * site_width + xl;   // align blanks to sites
                        } else if (node_xh >= blank.xh) {                                       // one blank
                            blank.xh = floorDiv((node_xl - xl), site_width) * site_width + xl;  // align blanks to sites
                        } else {                                                                // two blanks
                            Blank<float> new_blank = blank;
                            blank.xh = floorDiv((node_xl - xl), site_width) * site_width + xl;  // align blanks to sites
                            new_blank.xl =
                                floorDiv((node_xh - xl), site_width) * site_width + xl;  // align blanks to sites
                            bin_blanks.at(blank_bin_id).insert(bin_blanks.at(blank_bin_id).begin() + bi + 1, new_blank);
                            --bi;
                            break;
                        }
                    }
                }
            }
        }
    }
}

void legalizeBin(
    const float* init_x,
    const float* init_y,
    const float* node_size_x,
    const float* node_size_y,
    std::vector<std::vector<Blank<float>>>& bin_blanks,  // blanks in each bin, sorted from low to high, left to right
    std::vector<std::vector<int>>& bin_cells,            // unplaced cells in each bin
    float* x,
    float* y,
    int num_bins_x,
    int num_bins_y,
    int blank_num_bins_y,
    float bin_size_x,
    float bin_size_y,
    float blank_bin_size_y,
    float site_width,
    float row_height,
    float xl,
    float yl,
    float xh,
    float yh,
    float alpha,   // a parameter to tune anchor initial locations and current locations
    float beta,    // a parameter to tune space reserving
    bool lr_flag,  // from left to right
    int* num_unplaced_cells) {
    for (int i = 0; i < num_bins_x * num_bins_y; i += 1) {
        int bin_id_x = i / num_bins_y;
        int bin_id_y = i - bin_id_x * num_bins_y;
        int blank_num_bins_per_bin = roundDiv(bin_size_y, blank_bin_size_y);
        int blank_bin_id_yl = bin_id_y * blank_num_bins_per_bin;
        int blank_bin_id_yh = std::min(blank_bin_id_yl + blank_num_bins_per_bin, blank_num_bins_y);

        // cells in this bin
        std::vector<int>& cells = bin_cells.at(i);

        // sort cells according to width
        if (lr_flag) {
            std::sort(cells.begin(), cells.end(), [&](int i, int j) -> bool {
                float wi = -1000 * (init_x[i] + node_size_x[i] / 2) + node_size_x[i] + node_size_y[i];
                float wj = -1000 * (init_x[j] + node_size_x[j] / 2) + node_size_x[j] + node_size_y[j];
                return wi < wj || (wi == wj && (init_y[i] > init_y[j] || (init_y[i] == init_y[j] && i < j)));
            });
        } else {
            std::sort(cells.begin(), cells.end(), [&](int i, int j) -> bool {
                float wi = 1000 * (init_x[i] + node_size_x[i] / 2) + node_size_x[i] + node_size_y[i];
                float wj = 1000 * (init_x[j] + node_size_x[j] / 2) + node_size_x[j] + node_size_y[j];
                return wi < wj || (wi == wj && (init_y[i] < init_y[j] || (init_y[i] == init_y[j] && i < j)));
            });
        }

        for (int ci = bin_cells.at(i).size() - 1; ci >= 0; --ci) {
            int node_id = cells.at(ci);
            // align to site
            float init_xl =
                floorDiv(((alpha * init_x[node_id] + (1 - alpha) * x[node_id]) - xl), site_width) * site_width + xl;
            float init_yl = (alpha * init_y[node_id] + (1 - alpha) * y[node_id]);
            float width = ceilDiv(node_size_x[node_id], site_width) * site_width;
            float height = node_size_y[node_id];

            int num_node_rows = ceilDiv(height, row_height);  // may take multiple rows
            int blank_index_offset[num_node_rows];
            std::fill(blank_index_offset, blank_index_offset + num_node_rows, 0);

            int blank_initial_bin_id_y = floorDiv((init_yl - yl), blank_bin_size_y);
            blank_initial_bin_id_y = std::min(blank_bin_id_yh - 1, std::max(blank_bin_id_yl, blank_initial_bin_id_y));
            int blank_bin_id_dist_y = std::max(blank_initial_bin_id_y + 1, blank_bin_id_yh - blank_initial_bin_id_y);

            int best_blank_bin_id_y = -1;
            int best_blank_bi[num_node_rows];
            std::fill(best_blank_bi, best_blank_bi + num_node_rows, -1);
            float best_cost = xh - xl + yh - yl;
            float best_xl = -1;
            float best_yl = -1;
            for (int bin_id_offset_y = 0; abs(bin_id_offset_y) < blank_bin_id_dist_y;
                 bin_id_offset_y = (bin_id_offset_y > 0) ? -bin_id_offset_y : -(bin_id_offset_y - 1)) {
                int blank_bin_id_y = blank_initial_bin_id_y + bin_id_offset_y;
                if (blank_bin_id_y < blank_bin_id_yl || blank_bin_id_y + num_node_rows > blank_bin_id_yh) {
                    continue;
                }
                int blank_bin_id = bin_id_x * blank_num_bins_y + blank_bin_id_y;
                // blanks in this bin
                const std::vector<Blank<float>>& blanks = bin_blanks.at(blank_bin_id);

                int row_best_blank_bi[num_node_rows];
                std::fill(row_best_blank_bi, row_best_blank_bi + num_node_rows, -1);
                float row_best_cost = xh - xl + yh - yl;
                float row_best_xl = -1;
                float row_best_yl = -1;
                bool search_flag = true;
                for (unsigned int bi = 0; search_flag && bi < bin_blanks.at(blank_bin_id).size(); ++bi) {
                    const Blank<float>& blank = blanks[bi];
                    // for multi-row height cells, check blanks in upper rows
                    // find blanks with maximum intersection
                    blank_index_offset[0] = bi;
                    std::fill(blank_index_offset + 1, blank_index_offset + num_node_rows, -1);

                    while (true) {
                        Interval<float> intersect_blank(blank.xl, blank.xh);
                        for (int row_offset = 1; row_offset < num_node_rows; ++row_offset) {
                            int next_blank_bin_id_y = blank_bin_id_y + row_offset;
                            int next_blank_bin_id = bin_id_x * blank_num_bins_y + next_blank_bin_id_y;
                            unsigned int next_bi = blank_index_offset[row_offset] + 1;
                            for (; next_bi < bin_blanks.at(next_blank_bin_id).size(); ++next_bi) {
                                const Blank<float>& next_blank = bin_blanks.at(next_blank_bin_id)[next_bi];
                                Interval<float> intersect_blank_tmp = intersect_blank;
                                intersect_blank_tmp.intersect(next_blank.xl, next_blank.xh);
                                if (intersect_blank_tmp.xh - intersect_blank_tmp.xl >= width) {
                                    intersect_blank = intersect_blank_tmp;
                                    blank_index_offset[row_offset] = next_bi;
                                    break;
                                }
                            }
                            if (next_bi == bin_blanks.at(next_blank_bin_id).size())  // not found
                            {
                                intersect_blank.xl = intersect_blank.xh = 0;
                                break;
                            }
                        }
                        float intersect_blank_width = intersect_blank.xh - intersect_blank.xl;
                        if (intersect_blank_width >= width) {
                            // compute displacement
                            float target_xl = init_xl;
                            float target_yl = blank.yl;
                            // alow tolerance to avoid more dead space
                            float beta = 4;
                            float tolerance = std::min(beta * width, intersect_blank_width / beta);
                            if (target_xl <= intersect_blank.xl + tolerance) {
                                target_xl = intersect_blank.xl;
                            } else if (target_xl + width >= intersect_blank.xh - tolerance) {
                                target_xl = (intersect_blank.xh - width);
                            }
                            float cost = fabs(target_xl - init_xl) + fabs(target_yl - init_yl);
                            // update best cost
                            if (cost < row_best_cost) {
                                std::copy(blank_index_offset, blank_index_offset + num_node_rows, row_best_blank_bi);
                                row_best_cost = cost;
                                row_best_xl = target_xl;
                                row_best_yl = target_yl;
                            } else {  // early exit since we iterate within rows from left to right
                                search_flag = false;
                            }
                        } else {  // not found
                            break;
                        }
                        if (num_node_rows < 2) {  // for single-row height cells
                            break;
                        }
                    }
                }
                if (row_best_cost < best_cost) {
                    best_blank_bin_id_y = blank_bin_id_y;
                    std::copy(row_best_blank_bi, row_best_blank_bi + num_node_rows, best_blank_bi);
                    best_cost = row_best_cost;
                    best_xl = row_best_xl;
                    best_yl = row_best_yl;
                } else if (best_cost + row_height < bin_id_offset_y * row_height) {
                    break;  // early exit since we iterate from close row to far-away row
                }
            }

            // found blank
            if (best_blank_bin_id_y >= 0) {
                x[node_id] = best_xl;
                y[node_id] = best_yl;
                // update cell position and blank
                for (int row_offset = 0; row_offset < num_node_rows; ++row_offset) {
                    assert(best_blank_bi[row_offset] >= 0);
                    // blanks in this bin
                    int best_blank_bin_id = bin_id_x * blank_num_bins_y + best_blank_bin_id_y + row_offset;
                    std::vector<Blank<float>>& blanks = bin_blanks.at(best_blank_bin_id);
                    Blank<float>& blank = blanks.at(best_blank_bi[row_offset]);
                    assert(best_xl >= blank.xl && best_xl + width <= blank.xh);
                    assert(best_yl + row_height * row_offset == blank.yl);
                    if (best_xl == blank.xl) {
                        // update blank
                        blank.xl += width;
                        if (floorDiv((blank.xl - xl), site_width) * site_width != blank.xl - xl) {
                            logger.debug("1. move node %d from %g to %g, blank (%g, %g)",
                                         node_id,
                                         x[node_id],
                                         blank.xl,
                                         blank.xl,
                                         blank.xh);
                        }
                        if (blank.xl >= blank.xh) {
                            bin_blanks.at(best_blank_bin_id)
                                .erase(bin_blanks.at(best_blank_bin_id).begin() + best_blank_bi[row_offset]);
                        }
                    } else if (best_xl + width == blank.xh) {
                        // update blank
                        blank.xh -= width;
                        if (floorDiv((blank.xh - xl), site_width) * site_width != blank.xh - xl) {
                            logger.debug("2. move node %d from %g to %g, blank (%g, %g)",
                                         node_id,
                                         x[node_id],
                                         blank.xh - width,
                                         blank.xl,
                                         blank.xh);
                        }
                        if (blank.xl >= blank.xh) {
                            bin_blanks.at(best_blank_bin_id)
                                .erase(bin_blanks.at(best_blank_bin_id).begin() + best_blank_bi[row_offset]);
                        }
                    } else {
                        // need to update current blank and insert one more blank
                        Blank<float> new_blank;
                        new_blank.xl = best_xl + width;
                        new_blank.xh = blank.xh;
                        new_blank.yl = blank.yl;
                        new_blank.yh = blank.yh;
                        blank.xh = best_xl;
                        if (floorDiv((blank.xl - xl), site_width) * site_width != blank.xl - xl ||
                            floorDiv((blank.xh - xl), site_width) * site_width != blank.xh - xl ||
                            floorDiv((new_blank.xl - xl), site_width) * site_width != new_blank.xl - xl ||
                            floorDiv((new_blank.xh - xl), site_width) * site_width != new_blank.xh - xl) {
                            logger.debug("3. move node %d from %g to %g, blank (%g, %g), new_blank (%g, %g)",
                                         node_id,
                                         x[node_id],
                                         init_xl,
                                         blank.xl,
                                         blank.xh,
                                         new_blank.xl,
                                         new_blank.xh);
                        }
                        bin_blanks.at(best_blank_bin_id)
                            .insert(bin_blanks.at(best_blank_bin_id).begin() + best_blank_bi[row_offset] + 1,
                                    new_blank);
                    }
                }

                // remove from cells
                bin_cells.at(i).erase(bin_cells.at(i).begin() + ci);
            }
        }
        *num_unplaced_cells += bin_cells.at(i).size();
    }
}

template <typename T>
void resizeBinObjects(std::vector<std::vector<T>>& bin_objs, int num_bins_x, int num_bins_y) {
    bin_objs.resize(num_bins_x * num_bins_y);
}

template <typename T>
void countBinObjects(const std::vector<std::vector<T>>& bin_objs) {
    int count = 0;
    for (unsigned int i = 0; i < bin_objs.size(); ++i) {
        count += bin_objs.at(i).size();
    }
}

void mergeBinBlanks(const std::vector<std::vector<Blank<float>>>& src_bin_blanks,
                    int src_num_bins_x,
                    int src_num_bins_y,  // dimensions for the src
                    std::vector<std::vector<Blank<float>>>& dst_bin_blanks,
                    int dst_num_bins_x,
                    int dst_num_bins_y,    // dimensions for the dst
                    int scale_ratio_x,     // roughly src_num_bins_x/dst_num_bins_x
                    float min_blank_width  // minimum blank width to consider
) {
    for (int i = 0; i < dst_num_bins_x * dst_num_bins_y; i += 1) {
        // assume src_num_bins_y == dst_num_bins_y
        int dst_bin_id_x = i / dst_num_bins_y;
        int dst_bin_id_y = i - dst_bin_id_x * dst_num_bins_y;

        int src_bin_id_x_bgn = dst_bin_id_x * scale_ratio_x;
        int src_bin_id_x_end = std::min(src_bin_id_x_bgn + scale_ratio_x, src_num_bins_x);

        std::vector<Blank<float>>& dst_bin_blank = dst_bin_blanks.at(i);

        for (int ix = src_bin_id_x_bgn; ix < src_bin_id_x_end; ++ix) {
            int iy = dst_bin_id_y;  // same as src_bin_id_y
            int src_bin_id = ix * src_num_bins_y + iy;
            const std::vector<Blank<float>>& src_bin_blank = src_bin_blanks.at(src_bin_id);

            int offset = 0;
            if (!dst_bin_blank.empty() && !src_bin_blank.empty()) {
                const Blank<float>& first_blank = src_bin_blank.at(0);
                Blank<float>& last_blank = dst_bin_blank.at(dst_bin_blank.size() - 1);
                if (last_blank.yl == first_blank.yl && last_blank.xh == first_blank.xl) {
                    last_blank.xh = first_blank.xh;
                    offset = 1;
                }
            }
            for (unsigned int k = offset; k < src_bin_blank.size(); ++k) {
                const Blank<float>& blank = src_bin_blank.at(k);
                // prune small blanks
                if (blank.xh - blank.xl >= min_blank_width) {
                    dst_bin_blanks.at(i).push_back(blank);
                }
            }
        }
    }
}

void mergeBinCells(
    const std::vector<std::vector<int>>& src_bin_cells,
    int src_num_bins_x,
    int src_num_bins_y,  // dimensions for the src
    std::vector<std::vector<int>>& dst_bin_cells,
    int dst_num_bins_x,
    int dst_num_bins_y,  // dimensions for the dst
    int scale_ratio_x,
    int scale_ratio_y  // roughly src_num_bins_x/dst_num_bins_x, but may not be exactly the same due to even/odd numbers
) {
    for (int i = 0; i < dst_num_bins_x * dst_num_bins_y; i += 1) {
        int dst_bin_id_x = i / dst_num_bins_y;
        int dst_bin_id_y = i - dst_bin_id_x * dst_num_bins_y;

        int src_bin_id_x_bgn = dst_bin_id_x * scale_ratio_x;
        int src_bin_id_y_bgn = dst_bin_id_y * scale_ratio_y;
        int src_bin_id_x_end = std::min(src_bin_id_x_bgn + scale_ratio_x, src_num_bins_x);
        int src_bin_id_y_end = std::min(src_bin_id_y_bgn + scale_ratio_y, src_num_bins_y);

        for (int ix = src_bin_id_x_bgn; ix < src_bin_id_x_end; ++ix) {
            for (int iy = src_bin_id_y_bgn; iy < src_bin_id_y_end; ++iy) {
                int src_bin_id = ix * src_num_bins_y + iy;

                const std::vector<int>& src_bin_cell = src_bin_cells.at(src_bin_id);

                dst_bin_cells.at(i).insert(dst_bin_cells.at(i).end(), src_bin_cell.begin(), src_bin_cell.end());
            }
        }
    }
}

void minNodeSize(const std::vector<std::vector<int>>& bin_cells,
                 const float* node_size_x,
                 const float* node_size_y,
                 float site_width,
                 float row_height,
                 int num_bins_x,
                 int num_bins_y,
                 int* min_node_size_x) {
    for (int i = 0; i < num_bins_x * num_bins_y; i += 1) {
        const std::vector<int>& cells = bin_cells.at(i);
        float min_size_x = std::numeric_limits<int>::max();
        for (unsigned int k = 0; k < cells.size(); ++k) {
            int node_id = cells.at(k);
            min_size_x = std::min(min_size_x, node_size_x[node_id]);
        }
        if (min_size_x != std::numeric_limits<int>::max()) {
            *min_node_size_x = std::min(*min_node_size_x, (int)ceilDiv(min_size_x, site_width));
        }
    }
}

void greedyLegalization(DPTorchRawDB& at_db, int num_bins_x, int num_bins_y) {
    LegalizationData db(at_db);
    db.set_num_bins(num_bins_x, num_bins_y);
    // first from right to left
    // then from left to right
    for (int i = 0; i < 2; ++i) {
        num_bins_x = 1;
        num_bins_y = 1;
        // adjust bin sizes
        float bin_size_x = (db.xh - db.xl) / static_cast<float>(num_bins_x);
        float bin_size_y = (db.yh - db.yl) / static_cast<float>(num_bins_y);
        bin_size_y = std::max((float)(ceilDiv(bin_size_y, db.row_height) * db.row_height), db.row_height);
        num_bins_y = ceilDiv((db.yh - db.yl), bin_size_y);

        // bin dimension in y direction for blanks is different from that for cells
        float blank_bin_size_y = db.row_height;
        int blank_num_bins_y = floorDiv((db.yh - db.yl), blank_bin_size_y);
        logger.debug("%s blank_num_bins_y = %d", "Standard cell legalization", blank_num_bins_y);

        // allocate bin cells
        std::vector<std::vector<int>> bin_cells(num_bins_x * num_bins_y);
        std::vector<std::vector<int>> bin_cells_copy(num_bins_x * num_bins_y);

        // distribute cells to bins
        distributeCells2Bins(db,
                             db.x,
                             db.y,
                             db.node_size_x,
                             db.node_size_y,
                             bin_size_x,
                             bin_size_y,
                             db.xl,
                             db.yl,
                             db.xh,
                             db.yh,
                             num_bins_x,
                             num_bins_y,
                             db.num_nodes,
                             db.num_movable_nodes,
                             bin_cells);

        // allocate bin fixed cells
        std::vector<std::vector<int>> bin_fixed_cells(num_bins_x * num_bins_y);

        // distribute fixed cells to bins
        distributeFixedCells2Bins(db,
                                  db.init_x,
                                  db.init_y,
                                  db.node_size_x,
                                  db.node_size_y,
                                  bin_size_x,
                                  bin_size_y,
                                  db.xl,
                                  db.yl,
                                  db.xh,
                                  db.yh,
                                  num_bins_x,
                                  num_bins_y,
                                  db.num_nodes,
                                  db.num_movable_nodes,
                                  bin_fixed_cells);

        // allocate bin blanks
        std::vector<std::vector<Blank<float>>> bin_blanks(num_bins_x * blank_num_bins_y);
        std::vector<std::vector<Blank<float>>> bin_blanks_copy(num_bins_x * blank_num_bins_y);

        // distribute blanks to bins
        distributeBlanks2Bins(db.init_x,
                              db.init_y,
                              db.node_size_x,
                              db.node_size_y,
                              bin_fixed_cells,
                              bin_size_x,
                              bin_size_y,
                              blank_bin_size_y,
                              db.xl,
                              db.yl,
                              db.xh,
                              db.yh,
                              db.site_width,
                              db.row_height,
                              num_bins_x,
                              num_bins_y,
                              blank_num_bins_y,
                              bin_blanks);

        int num_unplaced_cells_host;
        // minimum width in sites
        int min_unplaced_node_size_x_host;
        int num_iters = floor(log((float)std::min(num_bins_x, num_bins_y)) / log(2.0)) + 1;
        for (int iter = 0; iter < num_iters; ++iter) {
            logger.debug(
                "%s iteration %d with %dx%d bins", "Standard cell legalization", iter, num_bins_x, num_bins_y);
            num_unplaced_cells_host = 0;
            logger.debug("%s #bin_blanks", "Standard cell legalization");
            countBinObjects(bin_blanks);

            legalizeBin(db.init_x,
                        db.init_y,
                        db.node_size_x,
                        db.node_size_y,
                        bin_blanks,  // blanks in each bin, sorted from low to high, left to right
                        bin_cells,   // unplaced cells in each bin
                        db.x,
                        db.y,
                        num_bins_x,
                        num_bins_y,
                        blank_num_bins_y,
                        bin_size_x,
                        bin_size_y,
                        blank_bin_size_y,
                        db.site_width,
                        db.row_height,
                        db.xl,
                        db.yl,
                        db.xh,
                        db.yh,
                        0.5,
                        4.0,
                        i % 2,
                        &num_unplaced_cells_host);
            logger.debug("%s num_unplaced_cells = %d", "Standard cell legalization", num_unplaced_cells_host);

            if (num_unplaced_cells_host == 0 || iter + 1 == num_iters) {
                break;
            }

            // compute minimum size of unplaced cells
            min_unplaced_node_size_x_host = floorDiv((db.xh - db.xl), db.site_width);
            minNodeSize(bin_cells,
                        db.node_size_x,
                        db.node_size_y,
                        db.site_width,
                        db.row_height,
                        num_bins_x,
                        num_bins_y,
                        &min_unplaced_node_size_x_host);
            logger.debug("%s minimum unplaced node_size_x %d sites",
                         "Standard cell legalization",
                         min_unplaced_node_size_x_host);

            // ceil(num_bins_x/2), ceil(num_bins_y/2)
            int dst_num_bins_x = (num_bins_x >> 1) + (num_bins_x & 1);
            int dst_num_bins_y = (num_bins_y >> 1) + (num_bins_y & 1);
            int scale_ratio_x = (num_bins_x == dst_num_bins_x) ? 1 : num_bins_x / dst_num_bins_x;
            int scale_ratio_y = (num_bins_y == dst_num_bins_y) ? 1 : num_bins_y / dst_num_bins_y;

            resizeBinObjects(bin_cells_copy, dst_num_bins_x, dst_num_bins_y);
            mergeBinCells(bin_cells,
                          num_bins_x,
                          num_bins_y,      // dimensions for the src
                          bin_cells_copy,  // ceil(src_num_bins_x/2) * ceil(src_num_bins_y/2)
                          dst_num_bins_x,
                          dst_num_bins_y,
                          scale_ratio_x,
                          scale_ratio_y);
            resizeBinObjects(bin_blanks_copy, dst_num_bins_x, blank_num_bins_y);
            mergeBinBlanks(bin_blanks,
                           num_bins_x,
                           blank_num_bins_y,  // dimensions for the src
                           bin_blanks_copy,   // ceil(src_num_bins_x/2) * ceil(src_num_bins_y/2)
                           dst_num_bins_x,
                           blank_num_bins_y,
                           scale_ratio_x,
                           min_unplaced_node_size_x_host * db.site_width);

            // update bin dimensions
            num_bins_x = dst_num_bins_x;
            num_bins_y = dst_num_bins_y;

            bin_size_x = bin_size_x * 2;
            bin_size_y = bin_size_y * 2;

            std::swap(bin_cells, bin_cells_copy);
            std::swap(bin_blanks, bin_blanks_copy);
        }
    }
}

}  // namespace dp