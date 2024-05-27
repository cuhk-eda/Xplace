#include "common/common.h"
#include "common/db/Database.h"
#include "flute.h"
#include "io_parser/gp/GPDatabase.h"

namespace routedp {

inline int floorDiv(float a, float b, float rtol = 1e-4) { return std::floor((a + rtol * b) / b); }

inline int ceilDiv(float a, float b, float rtol = 1e-4) { return std::ceil((a - rtol * b) / b); }

inline int roundDiv(float a, float b) { return std::round(a / b); }

torch::Tensor dp_route_opt(torch::Tensor node_lpos_init_,
                           torch::Tensor node_size_,
                           float dieLX,
                           float dieHX,
                           float dieLY,
                           float dieHY,
                           float site_width,
                           float row_height,
                           std::shared_ptr<db::Database> rawdb_,
                           std::shared_ptr<gp::GPDatabase> gpdb_,
                           int K) {
    // We found that placing cells under M2 SNet will easily cause DRVs
    // this function will shift cells outside the SNet within an acceptable range
    db::Database& rawdb = *rawdb_;
    gp::GPDatabase& gpdb = *gpdb_;

    torch::Tensor at_node_size_x = node_size_.index({"...", 0}).clone().contiguous();
    torch::Tensor at_node_size_y = node_size_.index({"...", 1}).clone().contiguous();
    torch::Tensor at_init_x = node_lpos_init_.index({"...", 0}).clone().contiguous();
    torch::Tensor at_init_y = node_lpos_init_.index({"...", 1}).clone().contiguous();
    torch::Tensor at_x = node_lpos_init_.index({"...", 0}).clone().contiguous();
    torch::Tensor at_y = node_lpos_init_.index({"...", 1}).clone().contiguous();

    float* node_size_x = at_node_size_x.data_ptr<float>();
    float* node_size_y = at_node_size_y.data_ptr<float>();
    float* init_x = at_init_x.data_ptr<float>();
    float* init_y = at_init_y.data_ptr<float>();
    float* x = at_x.data_ptr<float>();
    float* y = at_y.data_ptr<float>();

    int num_bin_x = 1;  // row-based
    int nLayers = rawdb.getNumRLayers();

    float bin_size_x = (dieHX - dieLX) / num_bin_x;
    float bin_size_y = row_height;  // row height

    int num_bin_y = ceilDiv(dieHY - dieLY, bin_size_y);

    std::vector<std::vector<std::pair<int, int>>> bin2snets(nLayers * num_bin_x * num_bin_y);
    for (int snetId = 0; snetId < rawdb.snets.size(); snetId++) {
        db::SNet* snet = rawdb.snets[snetId];
        for (size_t shapeIdx = 0; shapeIdx < snet->shapes.size(); shapeIdx++) {
            auto& shape = snet->shapes[shapeIdx];
            int currLayer = shape.layer.rIndex;
            int bin_id_lx = std::max(((float)shape.lx / rawdb.siteW - dieLX) / bin_size_x, (float)0);
            int bin_id_hx = std::min((int)ceil(((float)shape.hx / rawdb.siteW - dieLX) / bin_size_x), num_bin_x);
            int bin_id_ly = std::max(((float)shape.ly / rawdb.siteW - dieLY) / bin_size_y, (float)0);
            int bin_id_hy = std::min((int)ceil(((float)shape.hy / rawdb.siteW - dieLY) / bin_size_y), num_bin_y);
            for (int bin_id_x = bin_id_lx; bin_id_x < bin_id_hx; bin_id_x++) {
                for (int bin_id_y = bin_id_ly; bin_id_y < bin_id_hy; bin_id_y++) {
                    int bin_id = currLayer * num_bin_x * num_bin_y + bin_id_x * num_bin_y + bin_id_y;
                    bin2snets[bin_id].emplace_back(snetId, shapeIdx);
                }
            }
        }
    }
    for (int bin_id = 0; bin_id < bin2snets.size(); bin_id++) {
        if (bin2snets[bin_id].size() == 0) continue;
        std::stable_sort(
            bin2snets[bin_id].begin(), bin2snets[bin_id].end(), [&](std::pair<int, int> p1, std::pair<int, int> p2) {
                int id1 = p1.first;
                int id2 = p2.first;
                db::SNet* snet1 = rawdb.snets[id1];
                db::SNet* snet2 = rawdb.snets[id2];
                int shapeIdx1 = p1.second;
                int shapeIdx2 = p2.second;
                db::Geometry& shape1 = snet1->shapes[shapeIdx1];
                db::Geometry& shape2 = snet2->shapes[shapeIdx2];
                float x1 = shape1.lx;
                float x2 = shape2.lx;
                float xx1 = shape1.hx;
                float xx2 = shape2.hx;
                return x1 < x2 || (x1 == x2 && (xx1 > xx2 || (xx1 == xx2 && id1 < id2)));
            });
    }

    std::vector<std::vector<int>> bin2cells(num_bin_x * num_bin_y);
    // add nodes
    for (int i = 0; i < gpdb.getNodes().size(); i++) {
        if (gpdb.getNodes()[i].getNodeType() == "IOPin" || gpdb.getNodes()[i].getNodeType() == "FloatIOPin") {
            continue;
        }
        if (std::round(node_size_y[i] / bin_size_y) <= 1) {
            int bin_id_x = (x[i] + node_size_x[i] / 2 - dieLX) / bin_size_x;
            int bin_id_y = (y[i] + node_size_y[i] / 2 - dieLY) / bin_size_y;

            bin_id_x = std::min(std::max(bin_id_x, 0), num_bin_x - 1);
            bin_id_y = std::min(std::max(bin_id_y, 0), num_bin_y - 1);

            int bin_id = bin_id_x * num_bin_y + bin_id_y;

            bin2cells[bin_id].emplace_back(i);
        } else {
            int bin_id_lx = std::max((x[i] - dieLX) / bin_size_x, (float)0);
            int bin_id_hx = std::min((int)ceil((x[i] + node_size_x[i] - dieLX) / bin_size_x), num_bin_x);
            int bin_id_ly = std::max((y[i] - dieLY) / bin_size_y, (float)0);
            int bin_id_hy = std::min((int)ceil((y[i] + node_size_y[i] - dieLY) / bin_size_y), num_bin_y);

            for (int bin_id_x = bin_id_lx; bin_id_x < bin_id_hx; bin_id_x++) {
                for (int bin_id_y = bin_id_ly; bin_id_y < bin_id_hy; bin_id_y++) {
                    int bin_id = bin_id_x * num_bin_y + bin_id_y;
                    bin2cells[bin_id].emplace_back(i);
                }
            }
        }
    }

    std::vector<bool> cellIsMove(gpdb.getNodes().size(), false);
    for (int i = 0; i < gpdb.getNodes().size(); i++) {
        auto node_type = gpdb.getNodes()[i].getNodeType();
        if (node_type == "Mov" || node_type == "FloatMov") {
            cellIsMove[i] = true;
        }
    }

    float totalDisplace = 0.0;
    int totalNumMoves = 0;

    for (int bin_id = 0; bin_id < bin2cells.size(); bin_id++) {
        auto& currBin2cellsTmp = bin2cells[bin_id];
        if (currBin2cellsTmp.size() == 0) {
            continue;
        }
        float rowDisplace = 0.0;
        int rowNumMoves = 0;

        int bin_id_x = bin_id / num_bin_y;
        int bin_id_y = bin_id % num_bin_y;

        int m2BinId = 1 * num_bin_x * num_bin_y + bin_id_x * num_bin_y + bin_id_y;
        auto& currM2Snets = bin2snets[m2BinId];
        if (currM2Snets.size() == 0) {
            continue;
        }

        // sort all nodes by row
        std::sort(currBin2cellsTmp.begin(), currBin2cellsTmp.end(), [&](int i, int j) {
            float x1 = x[i];
            float x2 = x[j];
            float w1 = node_size_x[i];
            float w2 = node_size_x[j];
            return x1 < x2 || (x1 == x2 && (w1 > w2 || (w1 == w2 && i < j)));
        });
        // remove fixed cell overlap in row
        bool errorFlag = false;
        std::vector<std::tuple<int, float, float>> currBin2cells;  // nodeId, lx, hx
        for (int i = 0; i < currBin2cellsTmp.size(); i++) {
            int this_id = currBin2cellsTmp[i];
            float this_lx = x[this_id];
            float this_hx = x[this_id] + node_size_x[this_id];
            if (currBin2cells.size() == 0) {
                currBin2cells.emplace_back(this_id, this_lx, this_hx);
                continue;
            } else {
                auto [last_id, last_lx, last_hx] = currBin2cells.back();
                if (std::max(this_lx, last_lx) < std::min(this_hx, last_hx)) {
                    if (cellIsMove[this_id] || cellIsMove[last_id]) {
                        // one of movable cells overlap
                        currBin2cells.emplace_back(this_id, this_lx, this_hx);
                        logger.error("Node %d (%.1f, %.1f) overlap with Node %d (%.1f, %.1f)",
                                     this_id,
                                     this_lx,
                                     this_hx,
                                     last_id,
                                     last_lx,
                                     last_hx);
                        errorFlag = true;
                    } else {
                        // two fixed cells overlap, merge them and mark it as true
                        float new_lx = std::min(this_lx, last_lx);
                        float new_hx = std::max(this_hx, last_hx);
                        currBin2cells[currBin2cells.size() - 1] = {last_id, new_lx, new_hx};
                    }
                } else {
                    currBin2cells.emplace_back(this_id, this_lx, this_hx);
                }
            }
        }
        if (errorFlag) continue;

        float binLx = std::max(bin_id_x * bin_size_x, (float)0);
        float binHx = std::min((bin_id_x + 1) * bin_size_x, (float)dieHX);
        float binLy = std::max(bin_id_y * bin_size_y, (float)0);
        float binHy = std::min((bin_id_y + 1) * bin_size_y, (float)dieHY);

        int cellPtr = 0;
        int snetPtr = 0;
        while (cellPtr != currBin2cells.size() && snetPtr != currM2Snets.size()) {
            if (!cellIsMove[std::get<0>(currBin2cells[cellPtr])]) {
                cellPtr++;
                continue;
            }
            db::SNet* snet = rawdb.snets[currM2Snets[snetPtr].first];
            int shapeIdx = currM2Snets[snetPtr].second;
            db::Geometry& shape = snet->shapes[shapeIdx];
            auto [node_id, node_lx, node_hx] = currBin2cells[cellPtr];
            float snetLx = (float)shape.lx / rawdb.siteW;
            float snetHx = (float)shape.hx / rawdb.siteW;
            if (std::max(node_lx, snetLx) < std::min(node_hx, snetHx)) {
                // overlap with snet
                std::vector<std::tuple<int, float, int>> movesL;  // cellId, x, currBin2cellsId
                std::vector<std::tuple<int, float, int>> movesR;  // cellId, x, currBin2cellsId
                // when more than one cells are located in the same SNet stripe, two cases:
                //   (1) move the first cell to left
                //   (2) move the all cells to right
                // Move left:
                int cellPtrL = cellPtr;
                float src_width_l = ceilDiv(node_hx - node_lx, site_width) * site_width;
                // record the position of cellPtrL + ptrOffsetL + 1
                float lhsX = std::max(floorDiv(snetLx - dieLX, site_width) * site_width + dieLX, dieLX);
                float blank_width_l = 0;
                int ptrOffsetL = 0;
                bool doMoveL = true;
                float displaceL = dieHX;
                doMoveL = doMoveL && (snetLx > dieLX);
                if (doMoveL && blank_width_l < src_width_l) {
                    for (ptrOffsetL = -1; ptrOffsetL >= -K; ptrOffsetL--) {
                        int targetPtr = cellPtrL + ptrOffsetL;
                        if (targetPtr >= 0) {
                            auto [node_id1, node_lx1, node_hx1] = currBin2cells[targetPtr];
                            float blank =
                                std::max(lhsX - (ceilDiv(node_hx1 - dieLX, site_width) * site_width + dieLX), (float)0);
                            blank_width_l += blank;
                            if (blank_width_l >= src_width_l) {
                                // src_width_l, blank_width_l, blank are both integer multiple of site_width
                                lhsX = lhsX - (src_width_l - blank_width_l + blank);
                                break;
                            }
                            lhsX = std::max(floorDiv(node_lx1 - dieLX, site_width) * site_width + dieLX, dieLX);
                            if (!cellIsMove[node_id1]) {
                                break;
                            }
                        } else {
                            float blank = std::max(floorDiv(lhsX - dieLX, site_width) * site_width, (float)0);
                            blank_width_l += blank;
                            if (blank_width_l >= src_width_l) {
                                lhsX = lhsX - (src_width_l - blank_width_l + blank);
                                break;
                            }
                            lhsX = dieLX;
                            break;
                        }
                    }
                }
                doMoveL = doMoveL && (blank_width_l >= src_width_l);
                if (doMoveL) {
                    // compute cell movement and displacement
                    // shift cell to the lhs of snet polygon
                    float lhsX_copy = lhsX;
                    displaceL = 0.0;
                    for (int targetPtr = cellPtrL + ptrOffsetL + 1; targetPtr <= cellPtrL; targetPtr++) {
                        auto [node_id1, node_lx1, node_hx1] = currBin2cells[targetPtr];
                        movesL.emplace_back(node_id1, lhsX_copy, targetPtr);
                        displaceL += std::abs(init_x[node_id1] - lhsX_copy);
                        lhsX_copy = ceilDiv(node_hx1 - dieLX, site_width) * site_width + dieLX;
                    }
                }

                // Move right:
                // find all cells overlap with current SNet
                int cellPtrR = cellPtr;
                int node_id1;
                float node_lx1, node_hx1;
                std::tie(node_id1, node_lx1, node_hx1) = currBin2cells[cellPtrR];
                bool doMoveR = true;
                float displaceR = dieHX;
                while (std::max(node_lx1, snetLx) < std::min(node_hx1, snetHx)) {
                    cellPtrR++;
                    if (cellPtrR == currBin2cells.size()) break;
                    std::tie(node_id1, node_lx1, node_hx1) = currBin2cells[cellPtrR];
                    if (!cellIsMove[node_id1]) {
                        doMoveR = false;
                        break;
                    }
                }
                cellPtrR--;
                float src_width_r = 0;
                for (int i = cellPtrL; i <= cellPtrR; i++) {
                    auto [node_id1, node_lx1, node_hx1] = currBin2cells[i];
                    src_width_r += ceilDiv(node_hx1 - node_lx1, site_width) * site_width;
                }
                float rhsX = std::min(ceilDiv(snetHx - dieLX, site_width) * site_width + dieLX, dieHX);
                float blank_width_r = 0;
                int ptrOffsetR = 0;
                doMoveR = doMoveR && (snetHx < dieHX);
                if (doMoveR && blank_width_r < src_width_r) {
                    for (ptrOffsetR = 1; ptrOffsetR <= K; ptrOffsetR++) {
                        int targetPtr = cellPtrR + ptrOffsetR;
                        if (targetPtr < currBin2cells.size()) {
                            auto [node_id1, node_lx1, node_hx1] = currBin2cells[targetPtr];
                            float blank = std::max((floorDiv(node_lx1 - dieLX, site_width) * site_width + dieLX) - rhsX,
                                                   (float)0);
                            blank_width_r += blank;
                            if (blank_width_r >= src_width_r) {
                                // src_width_r, blank_width_r, blank are both integer multiple of site_width
                                rhsX = rhsX + (src_width_r - blank_width_r + blank);
                                break;
                            }
                            rhsX = std::min(ceilDiv(node_hx1 - dieLX, site_width) * site_width + dieLX, dieHX);
                            if (!cellIsMove[node_id1]) {
                                break;
                            }
                        } else {
                            float blank = std::max(floorDiv(dieHX - rhsX, site_width) * site_width, (float)0);
                            blank_width_r += blank;
                            if (blank_width_r >= src_width_r) {
                                rhsX = rhsX + (src_width_r - blank_width_r + blank);
                                break;
                            }
                            rhsX = dieHX;
                            break;
                        }
                    }
                }
                doMoveR = doMoveR && (blank_width_r >= src_width_r);

                if (doMoveR) {
                    // compute cell movement and displacement
                    // shift cells to the rhs of snet polygon
                    float rhsX_copy = rhsX;
                    displaceR = 0.0;
                    for (int targetPtr = cellPtrR + ptrOffsetR - 1; targetPtr >= cellPtrL; targetPtr--) {
                        auto [node_id1, node_lx1, node_hx1] = currBin2cells[targetPtr];
                        float thisLhs = rhsX_copy - ceilDiv(node_hx1 - node_lx1, site_width) * site_width;
                        movesR.emplace_back(node_id1, thisLhs, targetPtr);
                        displaceR += std::abs(init_x[node_id1] - thisLhs);
                        rhsX_copy = thisLhs;
                    }
                }

                // do move and update currBin2cells by the new lx and new hx
                float maxDis = site_width * 50 + 1e-4;  // max displacement
                if ((doMoveL || doMoveR) && (std::round(displaceL) < maxDis || std::round(displaceR) < maxDis)) {
                    if (std::round(displaceL) <= std::round(displaceR)) {
                        for (auto [node_id1, targetX, targetPtr] : movesL) {
                            currBin2cells[targetPtr] = {node_id1, targetX, targetX + node_size_x[node_id1]};
                            x[node_id1] = targetX;
                        }
                    } else {
                        for (auto [node_id1, targetX, targetPtr] : movesR) {
                            currBin2cells[targetPtr] = {node_id1, targetX, targetX + node_size_x[node_id1]};
                            x[node_id1] = targetX;
                        }
                    }
                    logger.debug("SNetMove Node: %d PtrOffsetL: %d DisplaceL %.1f PtrOffsetR: %d DisplaceR %.1f",
                                 node_id,
                                 ptrOffsetL,
                                 displaceL,
                                 ptrOffsetR,
                                 displaceR);
                }
                cellPtr++;
            } else if (node_hx <= snetLx) {
                cellPtr++;
            } else if (node_lx >= snetHx) {
                snetPtr++;
            } else {
                cellPtr++;
            }
        }

        for (auto node_id : currBin2cellsTmp) {
            float dis = std::abs(x[node_id] - init_x[node_id]);
            if (std::lround(dis) > 0) {
                rowNumMoves++;
                rowDisplace += dis;
            }
        }
        totalDisplace += rowDisplace;
        totalNumMoves += rowNumMoves;
    }
    torch::Tensor new_node_lpos = torch::stack({at_x, at_y}, 1);
    logger.info("Route Move #Moves: %d Displacement: %.1f", totalNumMoves, totalDisplace);
    return new_node_lpos;
}

}  // namespace routedp

namespace Xplace {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("dp_route_opt", &routedp::dp_route_opt, "dp_route_opt"); }

}  // namespace Xplace
