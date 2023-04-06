#include "gpudp/lg/legalization_db.h"

namespace dp {

struct AbacusCluster {
    int prev_cluster_id;  ///< previous cluster, set to INT_MIN if the cluster is
                          ///< invalid
    int next_cluster_id;  ///< next cluster, set to INT_MIN if the cluster is
                          ///< invalid
    int bgn_row_node_id;  ///< id of first node in the row
    int end_row_node_id;  ///< id of last node in the row
    float e;              ///< weight of displacement in the objective
    float q;              ///< x = q/e
    float w;              ///< width
    float x;              ///< optimal location

    /// @return whether this is a valid cluster
    bool valid() const { return prev_cluster_id != INT_MIN && next_cluster_id != INT_MIN; }
};

/// @brief helper function for distributing cells to rows
/// sort cells within a row and clean overlapping fixed cells
void sortNodesInRow(const float* host_x,
                    const float* host_y,
                    const float* host_node_size_x,
                    const float* host_node_size_y,
                    int num_movable_nodes,
                    std::vector<int>& nodes_in_row) {
    // sort cells within rows according to left edges
    std::sort(nodes_in_row.begin(), nodes_in_row.end(), [&](int node_id1, int node_id2) {
        float x1 = host_x[node_id1];
        float x2 = host_x[node_id2];
        // put larger width front will help remove
        // overlapping fixed cells, especially when
        // x1 == x2, then we need the wider one comes first
        float w1 = host_node_size_x[node_id1];
        float w2 = host_node_size_x[node_id2];
        return x1 < x2 || (x1 == x2 && (w1 > w2 || (w1 == w2 && node_id1 < node_id2)));
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
        int j_1 = 0;
        for (int j = 1, je = nodes_in_row.size(); j < je; ++j) {
            int node_id1 = nodes_in_row.at(j_1);
            int node_id2 = nodes_in_row.at(j);
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
                    j_1 = j;
                }
            } else {
                tmp_nodes.push_back(node_id2);
                j_1 = j;
            }
        }
        nodes_in_row.swap(tmp_nodes);

        // sort according to center
        std::sort(nodes_in_row.begin(), nodes_in_row.end(), [&](int node_id1, int node_id2) {
            float x1 = host_x[node_id1] + host_node_size_x[node_id1] / 2;
            float x2 = host_x[node_id2] + host_node_size_x[node_id2] / 2;
            return x1 < x2 || (x1 == x2 && node_id1 < node_id2);
        });

        for (int j = 1, je = nodes_in_row.size(); j < je; ++j) {
            int node_id1 = nodes_in_row.at(j - 1);
            int node_id2 = nodes_in_row.at(j);
            float xl1 = host_x[node_id1];
            float xl2 = host_x[node_id2];
            float width1 = host_node_size_x[node_id1];
            float width2 = host_node_size_x[node_id2];
            float xh1 = xl1 + width1;
            float xh2 = xl2 + width2;
            float yl1 = host_y[node_id1];
            float yl2 = host_y[node_id2];
            float yh1 = yl1 + host_node_size_y[node_id1];
            float yh2 = yl2 + host_node_size_y[node_id2];
            assert_msg(xl1 < xl2 && xh1 < xh2,
                       "node %d (%g, %g, %g, %g) overlaps with node %d (%g, %g, %g, %g)",
                       node_id1,
                       xl1,
                       yl1,
                       xh1,
                       yh1,
                       node_id2,
                       xl2,
                       yl2,
                       xh2,
                       yh2);
        }
    }
}

void distributeMovableAndFixedCells2Bins(const float* x,
                                         const float* y,
                                         const float* node_size_x,
                                         const float* node_size_y,
                                         float bin_size_x,
                                         float bin_size_y,
                                         float xl,
                                         float yl,
                                         float xh,
                                         float yh,
                                         float site_width,
                                         int num_bins_x,
                                         int num_bins_y,
                                         int num_nodes,
                                         int num_movable_nodes,
                                         std::vector<std::vector<int>>& bin_cells) {
    for (int i = 0; i < num_nodes; i += 1) {
        if (i < num_movable_nodes && roundDiv(node_size_y[i], bin_size_y) <= 1) {
            // single-row movable nodes only distribute to one bin
            int bin_id_x = (x[i] + node_size_x[i] / 2 - xl) / bin_size_x;
            int bin_id_y = (y[i] + node_size_y[i] / 2 - yl) / bin_size_y;

            bin_id_x = std::min(std::max(bin_id_x, 0), num_bins_x - 1);
            bin_id_y = std::min(std::max(bin_id_y, 0), num_bins_y - 1);

            int bin_id = bin_id_x * num_bins_y + bin_id_y;

            bin_cells[bin_id].push_back(i);
        } else {
            // fixed nodes may distribute to multiple bins
            int node_id = i;
            int bin_id_xl = std::max((x[node_id] - xl) / bin_size_x, (float)0);
            int bin_id_xh = std::min((int)ceil((x[node_id] + node_size_x[node_id] - xl) / bin_size_x), num_bins_x);
            int bin_id_yl = std::max((y[node_id] - yl) / bin_size_y, (float)0);
            int bin_id_yh = std::min((int)ceil((y[node_id] + node_size_y[node_id] - yl) / bin_size_y), num_bins_y);

            for (int bin_id_x = bin_id_xl; bin_id_x < bin_id_xh; ++bin_id_x) {
                for (int bin_id_y = bin_id_yl; bin_id_y < bin_id_yh; ++bin_id_y) {
                    int bin_id = bin_id_x * num_bins_y + bin_id_y;

                    bin_cells[bin_id].push_back(node_id);
                }
            }
        }
    }
}

/// @param row_nodes node indices in this row
/// @param clusters pre-allocated clusters in this row with the same length as
/// that of row_nodes
/// @param num_row_nodes length of row_nodes
/// @return true if succeed, otherwise false
bool abacusPlaceRowCPU(const float* init_x,
                       const float* node_size_x,
                       const float* node_size_y,
                       float* x,
                       float row_height,
                       float xl,
                       float xh,
                       int num_nodes,
                       int num_movable_nodes,
                       int* row_nodes,
                       AbacusCluster* clusters,
                       int num_row_nodes) {
    // a very large number
    float M = std::pow(10, ceilDiv(std::log((xh - xl) * num_row_nodes), log(10)));
    bool ret_flag = true;

    // merge two clusters
    // the second cluster will be invalid
    auto merge_cluster = [&](int dst_cluster_id, int src_cluster_id) {
        assert(dst_cluster_id < num_row_nodes);
        AbacusCluster& dst_cluster = clusters[dst_cluster_id];
        assert(src_cluster_id < num_row_nodes);
        AbacusCluster& src_cluster = clusters[src_cluster_id];

        assert(dst_cluster.valid() && src_cluster.valid());
        for (int i = dst_cluster_id + 1; i < src_cluster_id; ++i) {
            assert(!clusters[i].valid());
        }
        dst_cluster.end_row_node_id = src_cluster.end_row_node_id;
        assert(dst_cluster.e < M && src_cluster.e < M);
        dst_cluster.e += src_cluster.e;
        dst_cluster.q += src_cluster.q - src_cluster.e * dst_cluster.w;
        dst_cluster.w += src_cluster.w;
        // update linked list
        if (src_cluster.next_cluster_id < num_row_nodes) {
            clusters[src_cluster.next_cluster_id].prev_cluster_id = dst_cluster_id;
        }
        dst_cluster.next_cluster_id = src_cluster.next_cluster_id;
        src_cluster.prev_cluster_id = std::numeric_limits<int>::min();
        src_cluster.next_cluster_id = std::numeric_limits<int>::min();
    };

    // collapse clusters between [0, cluster_id]
    // compute the locations and merge clusters
    auto collapse = [&](int cluster_id, float range_xl, float range_xh) {
        int cur_cluster_id = cluster_id;
        assert(cur_cluster_id < num_row_nodes);
        int prev_cluster_id = clusters[cur_cluster_id].prev_cluster_id;
        AbacusCluster* cluster = nullptr;
        AbacusCluster* prev_cluster = nullptr;

        while (true) {
            assert(cur_cluster_id < num_row_nodes);
            cluster = &clusters[cur_cluster_id];
            cluster->x = cluster->q / cluster->e;
            // make sure cluster >= range_xl, so fixed nodes will not be moved
            // in illegal case, cluster+w > range_xh may occur, but it is OK.
            // We can collect failed clusters later
            cluster->x = std::max(std::min(cluster->x, range_xh - cluster->w), range_xl);
            assert(cluster->x >= range_xl && cluster->x + cluster->w <= range_xh);

            prev_cluster_id = cluster->prev_cluster_id;
            if (prev_cluster_id >= 0) {
                prev_cluster = &clusters[prev_cluster_id];
                if (prev_cluster->x + prev_cluster->w > cluster->x) {
                    merge_cluster(prev_cluster_id, cur_cluster_id);
                    cur_cluster_id = prev_cluster_id;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    };

    // initial cluster has only one cell
    for (int i = 0; i < num_row_nodes; ++i) {
        int node_id = row_nodes[i];
        AbacusCluster& cluster = clusters[i];
        cluster.prev_cluster_id = i - 1;
        cluster.next_cluster_id = i + 1;
        cluster.bgn_row_node_id = i;
        cluster.end_row_node_id = i;
        cluster.e = (node_id < num_movable_nodes && node_size_y[node_id] <= row_height) ? 1.0 : M;
        cluster.q = cluster.e * init_x[node_id];
        cluster.w = node_size_x[node_id];
        // this is required since we also include fixed nodes
        cluster.x = (node_id < num_movable_nodes && node_size_y[node_id] > row_height) ? x[node_id] : init_x[node_id];
    }

    // kernel algorithm for placeRow
    float range_xl = xl;
    float range_xh = xh;
    for (int j = 0; j < num_row_nodes; ++j) {
        const AbacusCluster& next_cluster = clusters[j];
        if (next_cluster.e >= M)  // fixed node
        {
            range_xh = std::min(next_cluster.x, range_xh);
            break;
        } else {
            assert(std::abs(node_size_y[row_nodes[j]] - row_height) < 1e-6);
        }
    }
    for (int i = 0; i < num_row_nodes; ++i) {
        const AbacusCluster& cluster = clusters[i];
        if (cluster.e < M) {
            assert(std::abs(node_size_y[row_nodes[i]] - row_height) < 1e-6);
            collapse(i, range_xl, range_xh);
        } else  // set range xl/xh according to fixed nodes
        {
            range_xl = cluster.x + cluster.w;
            range_xh = xh;
            for (int j = i + 1; j < num_row_nodes; ++j) {
                const AbacusCluster& next_cluster = clusters[j];
                if (next_cluster.e >= M)  // fixed node
                {
                    range_xh = std::min(next_cluster.x, range_xh);
                    break;
                }
            }
        }
    }

    // apply solution
    for (int i = 0; i < num_row_nodes; ++i) {
        if (clusters[i].valid()) {
            const AbacusCluster& cluster = clusters[i];
            float xc = cluster.x;
            for (int j = cluster.bgn_row_node_id; j <= cluster.end_row_node_id; ++j) {
                int node_id = row_nodes[j];
                if (node_id < num_movable_nodes && std::abs(node_size_y[node_id] - row_height) < 1e-6) {
                    x[node_id] = xc;
                } else if (xc != x[node_id]) {
                    if (node_id < num_movable_nodes)
                        logger.warning(
                            "multi-row node %d tends to move from %.12f to "
                            "%.12f, ignored",
                            node_id,
                            x[node_id],
                            xc);
                    else
                        logger.warning(
                            "fixed node %d tends to move from %.12f to %.12f, ignored", node_id, x[node_id], xc);
                    ret_flag = false;
                }
                xc += node_size_x[node_id];
            }
        }
    }

    return ret_flag;
}

void abacusLegalizeRow(const float* init_x,
                       const float* node_size_x,
                       const float* node_size_y,
                       float* x,
                       float* y,
                       float xl,
                       float xh,
                       float bin_size_x,
                       float bin_size_y,
                       int num_bins_x,
                       int num_bins_y,
                       int num_nodes,
                       int num_movable_nodes,
                       std::vector<std::vector<int>>& bin_cells,
                       std::vector<std::vector<AbacusCluster>>& bin_clusters) {
    for (unsigned int i = 0; i < bin_cells.size(); i += 1) {
        auto& row2nodes = bin_cells.at(i);

        // sort bin cells from left to right
        sortNodesInRow(x, y, node_size_x, node_size_y, num_movable_nodes, row2nodes);

        auto& clusters = bin_clusters.at(i);
        int num_row_nodes = row2nodes.size();

        int bin_id_x = i / num_bins_y;
        // int bin_id_y = i-bin_id_x*num_bins_y;

        float bin_xl = xl + bin_size_x * bin_id_x;
        float bin_xh = std::min(bin_xl + bin_size_x, xh);

        abacusPlaceRowCPU(init_x,
                          node_size_x,
                          node_size_y,
                          x,
                          bin_size_y,  // must be equal to row_height
                          bin_xl,
                          bin_xh,
                          num_nodes,
                          num_movable_nodes,
                          row2nodes.data(),
                          clusters.data(),
                          num_row_nodes);
    }
    float displace = 0;
    for (int i = 0; i < num_movable_nodes; ++i) {
        displace += fabs(x[i] - init_x[i]);
    }
    logger.debug("average displace = %g", displace / num_movable_nodes);
}

void abacusLegalization(DPTorchRawDB& at_db, int num_bins_x, int num_bins_y) {
    LegalizationData db(at_db);
    db.set_num_bins(num_bins_x, num_bins_y);
    // adjust bin sizes
    float bin_size_x = (db.xh - db.xl) / num_bins_x;
    float bin_size_y = db.row_height;
    num_bins_y = ceilDiv(db.yh - db.yl, bin_size_y);

    // include both movable and fixed nodes
    std::vector<std::vector<int>> bin_cells(num_bins_x * num_bins_y);
    // distribute cells to bins
    distributeMovableAndFixedCells2Bins(db.x,
                                        db.y,
                                        db.node_size_x,
                                        db.node_size_y,
                                        bin_size_x,
                                        bin_size_y,
                                        db.xl,
                                        db.yl,
                                        db.xh,
                                        db.yh,
                                        db.site_width,
                                        num_bins_x,
                                        num_bins_y,
                                        db.num_nodes,
                                        db.num_movable_nodes,
                                        bin_cells);

    std::vector<std::vector<AbacusCluster>> bin_clusters(num_bins_x * num_bins_y);
    for (unsigned int i = 0; i < bin_cells.size(); ++i) {
        bin_clusters[i].resize(bin_cells[i].size());
    }

    abacusLegalizeRow(db.init_x,
                      db.node_size_x,
                      db.node_size_y,
                      db.x,
                      db.y,
                      db.xl,
                      db.xh,
                      bin_size_x,
                      bin_size_y,
                      num_bins_x,
                      num_bins_y,
                      db.num_nodes,
                      db.num_movable_nodes,
                      bin_cells,
                      bin_clusters);
    // need to align nodes to sites
    // this also considers cell width which is not integral times of site_width
    for (auto const& cells : bin_cells) {
        float xxl = db.xl;
        for (auto node_id : cells) {
            if (node_id < db.num_movable_nodes) {
                db.x[node_id] = std::max(std::min(db.x[node_id], db.xh - db.node_size_x[node_id]), xxl);
                db.x[node_id] = floorDiv(db.x[node_id] - db.xl, db.site_width) * db.site_width + db.xl;
                xxl = db.x[node_id] + db.node_size_x[node_id]; 
            } else if (node_id < db.num_nodes) {
                xxl = ceilDiv(db.x[node_id] + db.node_size_x[node_id] - db.xl, db.site_width) * db.site_width + db.xl;
            }
        }
    }
}

}  // namespace dp