#include "gpudp/lg/hannan_legalize.h"
#include "gpudp/lg/legalization_db.h"

namespace dp {
/// @brief The macro legalization follows the way of floorplanning,
/// because macros have quite different sizes.

bool check_macro_legality(LegalizationData& db, const std::vector<int>& macros, bool fast_check) {
    // check legality between movable and fixed macros
    // for debug only, so it is slow
    auto checkOverlap2Nodes = [&](int i,
                                  int node_id1,
                                  float xl1,
                                  float yl1,
                                  float width1,
                                  float height1,
                                  int j,
                                  int node_id2,
                                  float xl2,
                                  float yl2,
                                  float width2,
                                  float height2) {
        float xh1 = xl1 + width1;
        float yh1 = yl1 + height1;
        float xh2 = xl2 + width2;
        float yh2 = yl2 + height2;
        if (std::min(xh1, xh2) > std::max(xl1, xl2) && std::min(yh1, yh2) > std::max(yl1, yl2)) {
            logger.error(
                "macro %d (%g, %g, %g, %g) var %d overlaps with macro %d "
                "(%g, %g, %g, %g) var %d, fixed: %d",
                node_id1,
                xl1,
                yl1,
                xh1,
                yh1,
                i,
                node_id2,
                xl2,
                yl2,
                xh2,
                yh2,
                j,
                (int)(node_id2 >= db.num_movable_nodes));
            return true;
        }
        return false;
    };

    bool legal = true;
    for (unsigned int i = 0, ie = macros.size(); i < ie; ++i) {
        int node_id1 = macros[i];
        float xl1 = db.x[node_id1];
        float yl1 = db.y[node_id1];
        float width1 = db.node_size_x[node_id1];
        float height1 = db.node_size_y[node_id1];
        // constraints with other macros
        for (unsigned int j = i + 1; j < ie; ++j) {
            int node_id2 = macros[j];
            float xl2 = db.x[node_id2];
            float yl2 = db.y[node_id2];
            float width2 = db.node_size_x[node_id2];
            float height2 = db.node_size_y[node_id2];

            bool overlap =
                checkOverlap2Nodes(i, node_id1, xl1, yl1, width1, height1, j, node_id2, xl2, yl2, width2, height2);
            if (overlap) {
                legal = false;
                if (fast_check) {
                    return legal;
                }
            }
        }
        // constraints with fixed macros
        // when considering fixed macros, there is no guarantee to find legal
        // solution with current ad-hoc constraint graphs
        for (int j = db.num_movable_nodes; j < db.num_nodes; ++j) {
            int node_id2 = j;
            float xl2 = db.init_x[node_id2];
            float yl2 = db.init_y[node_id2];
            float width2 = db.node_size_x[node_id2];
            float height2 = db.node_size_y[node_id2];

            bool overlap =
                checkOverlap2Nodes(i, node_id1, xl1, yl1, width1, height1, j, node_id2, xl2, yl2, width2, height2);
            if (overlap) {
                legal = false;
                if (fast_check) {
                    return legal;
                }
            }
        }
    }
    if (legal) {
        logger.debug("Macro legality check PASSED");
    } else {
        logger.error("Macro legality check FAILED");
    }

    return legal;
}

struct MacroLegalizeStats {
    float total_displace;
    float max_displace;
    float total_weighted_displace;  ///< displacement weighted by macro area ratio to
                                    ///< average macro area
    float max_weighted_displace;
    // float average_macro_area;
};

MacroLegalizeStats compute_displace(const LegalizationData& db, const std::vector<int>& macros) {
    MacroLegalizeStats stats;
    stats.total_displace = 0;
    stats.max_displace = 0;
    stats.total_weighted_displace = 0;
    stats.max_weighted_displace = 0;
    // stats.average_macro_area = 0;

    // for (auto node_id : macros)
    //{
    //    stats.average_macro_area += db.node_size_x[node_id] *
    //    db.node_size_y[node_id];
    //}
    // stats.average_macro_area /= macros.size();

    for (auto node_id : macros) {
        float displace = std::abs(db.init_x[node_id] - db.x[node_id]) + std::abs(db.init_y[node_id] - db.y[node_id]);
        stats.total_displace += displace;
        stats.max_displace = std::max(stats.max_displace, displace);

        displace *= db.node_weight[node_id];
        stats.total_weighted_displace += displace;
        stats.max_weighted_displace = std::max(stats.max_weighted_displace, displace);
    }
    return stats;
}

/// @brief Rough legalize some special macros
/// 1. macros that form small clusters overlapping with each other
/// 2. macros blocked by big ones
/// All the other macros are regarded as fixed.
/// @param small_clusters_flag controls whether to perform the legalization for
/// 1
/// @param blocked_macros_flag controls whether to perform the legalization for
/// 2
bool roughLegalize(LegalizationData& db,
                   const std::vector<int>& macros,
                   const std::vector<int>& fixed_macros,
                   bool small_clusters_flag,
                   bool blocked_macros_flag) {
    std::vector<unsigned char> markers(db.num_nodes, false);
    std::vector<int> macros_for_rough_legalize;
    std::vector<int> fixed_macros_for_rough_legalize;

    // collect small clusters
    if (small_clusters_flag) {
        std::vector<std::vector<int> > clusters(macros.size());
        float cluster_area_ratio = 2;
        float cluster_overlap_ratio = 0.5;
        unsigned int cluster_macro_numbers_threshold = 2;
        for (unsigned int i = 0, ie = macros.size(); i < ie; ++i) {
            int node_id1 = macros[i];
            Box box1(db.x[node_id1],
                     db.y[node_id1],
                     db.x[node_id1] + db.node_size_x[node_id1],
                     db.y[node_id1] + db.node_size_y[node_id1]);
            float a1 = box1.area();
            clusters.at(i).push_back(node_id1);
            for (unsigned int j = i + 1; j < ie; ++j) {
                int node_id2 = macros[j];
                Box box2(db.x[node_id2],
                         db.y[node_id2],
                         db.x[node_id2] + db.node_size_x[node_id2],
                         db.y[node_id2] + db.node_size_y[node_id2]);
                float a2 = box2.area();

                if (a1 >= a2 / cluster_area_ratio && a1 <= a2 * cluster_area_ratio) {
                    float overlap = std::max((float)0, std::min(box1.xh, box2.xh) - std::max(box1.xl, box2.xl)) *
                                    std::max((float)0, std::min(box1.yh, box2.yh) - std::max(box1.yl, box2.yl));
                    if (overlap >= std::min(a1, a2) * cluster_overlap_ratio) {
                        clusters.at(i).push_back(node_id2);
                    }
                }
            }
        }
        for (unsigned int i = 0, ie = macros.size(); i < ie; ++i) {
            if (clusters.at(i).size() >= cluster_macro_numbers_threshold) {
                markers.at(macros.at(i)) = true;
            }
        }
    }
    // collect small macros blocked by large ones
    // If a small macro is blocked by two big macros, it is easier to move the
    // small one around. We detect such blocks by checking whether the macro is
    // blocked from left, right, bottom, top 4 directions. Any macro with (left,
    // right) or (bottom, top) blocked will be collected.
    if (blocked_macros_flag) {
        float blocked_macros_area_ratio = 10;         // the area ratio of macros to be regarded as large
        float blocked_macros_direct_threshold = 0.9;  // determine the direction blocked
        for (unsigned int i = 0, ie = macros.size(); i < ie; ++i) {
            int node_id1 = macros[i];
            if (!markers[node_id1]) {
                Box box1(db.x[node_id1],
                         db.y[node_id1],
                         db.x[node_id1] + db.node_size_x[node_id1],
                         db.y[node_id1] + db.node_size_y[node_id1]);
                float a1 = box1.area();
                std::array<unsigned char, 4> intersect_directs;  // from L, R, B, float
                                                                 // direction, the box
                                                                 // is overlapped
                intersect_directs.fill(0);
                for (unsigned int j = 0; j < ie; ++j) {
                    int node_id2 = macros[j];
                    if (i != j && !markers[node_id2]) {
                        Box box2(db.x[node_id2],
                                 db.y[node_id2],
                                 db.x[node_id2] + db.node_size_x[node_id2],
                                 db.y[node_id2] + db.node_size_y[node_id2]);
                        float a2 = box2.area();

                        if (a1 * blocked_macros_area_ratio < a2) {
                            Box intersect_box(std::max(box1.xl, box2.xl),
                                              std::max(box1.yl, box2.yl),
                                              std::min(box1.xh, box2.xh),
                                              std::min(box1.yh, box2.yh));
                            if (intersect_box.xl < intersect_box.xh && intersect_box.yl < intersect_box.yh) {
                                if (intersect_box.height() > box1.height() * blocked_macros_direct_threshold) {
                                    if (box2.xl <= box1.xl) {
                                        intersect_directs[0] = 1;  // xl
                                    }
                                    if (box2.xh >= box1.xh) {
                                        intersect_directs[1] = 1;  // xh
                                    }
                                }
                                if (intersect_box.width() > box1.width() * blocked_macros_direct_threshold) {
                                    if (box2.yl <= box1.yl) {
                                        intersect_directs[2] = 1;  // yl
                                    }
                                    if (box2.yh >= box1.yh) {
                                        intersect_directs[3] = 1;  // yh
                                    }
                                }
                            }
                        }
                        if ((intersect_directs[0] && intersect_directs[1]) ||
                            (intersect_directs[2] && intersect_directs[3])) {
                            markers[node_id1] = true;
                            logger.debug("collect %d", node_id1);
                            break;
                        }
                    }
                }
            }
        }
    }

    fixed_macros_for_rough_legalize = fixed_macros;
    for (auto node_id : macros) {
        if (markers[node_id]) {
            macros_for_rough_legalize.push_back(node_id);
        } else {
            fixed_macros_for_rough_legalize.push_back(node_id);
        }
    }

    logger.info("Rough legalize small clusters with %lu macros", macros_for_rough_legalize.size());
    return hannanLegalize(db, macros_for_rough_legalize, fixed_macros_for_rough_legalize, 1);
}

bool macroLegalization(DPTorchRawDB& at_db, int num_bins_x, int num_bins_y) {
    LegalizationData db(at_db);
    db.set_num_bins(num_bins_x, num_bins_y);
    // collect macros
    std::vector<int> macros;
    for (int i = 0; i < db.num_movable_nodes; ++i) {
        if (db.is_dummy_fixed(i)) {
            // in some extreme case, some macros with 0 area should be ignored
            float area = db.node_size_x[i] * db.node_size_y[i];
            if (area > 0) {
                macros.push_back(i);
            }
        }
    }
    logger.info("Macro legalization: regard %lu cells as dummy fixed (movable macros)", macros.size());

    // in case there is no movable macros
    if (macros.empty()) {
        return true;
    }

    // fixed macros
    std::vector<int> fixed_macros;
    fixed_macros.reserve(db.num_nodes - db.num_movable_nodes);
    for (int i = db.num_movable_nodes; i < db.num_nodes; ++i) {
        // in some extreme case, some fixed macros with 0 area should be ignored
        float area = db.node_size_x[i] * db.node_size_y[i];
        if (area > 0) {
            fixed_macros.push_back(i);
        }
    }

    // store the best legalization solution found
    std::vector<float> best_x(macros.size());
    std::vector<float> best_y(macros.size());
    MacroLegalizeStats best_displace;
    best_displace.total_displace = std::numeric_limits<float>::max();
    best_displace.max_displace = std::numeric_limits<float>::max();
    best_displace.total_weighted_displace = std::numeric_limits<float>::max();
    best_displace.max_weighted_displace = std::numeric_limits<float>::max();

    // update current best solution
    auto update_best = [&](bool legal, const MacroLegalizeStats& displace) {
        if (legal && displace.total_displace < best_displace.total_displace) {
            for (unsigned int i = 0, ie = macros.size(); i < ie; ++i) {
                int macro_id = macros[i];
                best_x[i] = db.x[macro_id];
                best_y[i] = db.y[macro_id];
            }
            best_displace = displace;
        }
    };

    // first round rough legalization with Hannan grid for clusters
    bool small_clusters_flag = true;
    bool blocked_macros_flag = false;
    roughLegalize(db, macros, fixed_macros, small_clusters_flag, blocked_macros_flag);
    auto displace = compute_displace(db, macros);
    logger.info("Macro displacement total %g, max %g, weighted total %g, max %g",
                displace.total_displace,
                displace.max_displace,
                displace.total_weighted_displace,
                displace.max_weighted_displace);
    bool legal = check_macro_legality(db, macros, true);

    // try Hannan grid legalization if still not legal
    if (!legal) {
        legal = hannanLegalize(db, macros, fixed_macros, 10);
        auto displace = compute_displace(db, macros);
        logger.info("Macro displacement total %g, max %g, weighted total %g, max %g",
                    displace.total_displace,
                    displace.max_displace,
                    displace.total_weighted_displace,
                    displace.max_weighted_displace);
        legal = check_macro_legality(db, macros, true);
        update_best(legal, displace);

        // apply best solution
        if (best_displace.total_displace < std::numeric_limits<float>::max()) {
            logger.info(
                "use best macro displacement total %g, max %g, weighted "
                "total %g, max %g",
                best_displace.total_displace,
                best_displace.max_displace,
                best_displace.total_weighted_displace,
                best_displace.max_weighted_displace);
            for (unsigned int i = 0, ie = macros.size(); i < ie; ++i) {
                int macro_id = macros[i];
                db.x[macro_id] = best_x[i];
                db.y[macro_id] = best_y[i];
            }
        }
    }

    logger.info("Align macros to site and rows");
    // align the lower left corner to row and site
    for (unsigned int i = 0, ie = macros.size(); i < ie; ++i) {
        int node_id = macros[i];
        db.x[node_id] = db.align2site(db.x[node_id], db.node_size_x[node_id]);
        db.y[node_id] = db.align2row(db.y[node_id], db.node_size_y[node_id]);
    }

    legal = check_macro_legality(db, macros, false);

    return legal;
}

}  // namespace dp