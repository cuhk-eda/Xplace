#include "GRDatabase.h"

namespace gr {

GRDatabase::~GRDatabase() { logger.info("destruct grdb"); }

GRDatabase::GRDatabase(std::shared_ptr<db::Database> rawdb_, std::shared_ptr<gp::GPDatabase> gpdb_)
    : rawdb(*rawdb_), gpdb(*gpdb_) {
    logger.info("Init GRDatabase.");
    ISPD18 = (db::setting.LefFile.find("ispd18") != std::string::npos);
    ISPD19 = (db::setting.LefFile.find("ispd19") != std::string::npos);
    METAL5 = (db::setting.LefFile.find("metal5") != std::string::npos);
    if (grSetting.csrnScale <= 0) {
        csrnScale = 8 - ISPD19;
    } else {
        csrnScale = grSetting.csrnScale;
    }
    if (db::setting.BookshelfVariety != "" || db::setting.Format != "lefdef") {
        // NOTE: GGR is a LEFDEF based detailed-routability driven global placer and it is not
        //       designed for the old bookshelf designs. For bookshelf, please consider to use
        //       NCTU-GR to generate the routing congestion maps. Note that existing bookshelf
        //       designs cannot evaluate by academic/commercial detailed router.
        std::cout << "Bookshelf format is unsupported in GR. Terminated" << std::endl;
        exit(0);
    }

    // 1) init layers and tracks
    nLayers = rawdb.getNumRLayers();
    layerWidth.resize(nLayers);
    layerPitch.resize(nLayers);
    tracks.resize(nLayers);
    for (int l = 0; l < nLayers; l++) {
        auto rLayer = rawdb.getRLayer(l);
        layerWidth[l] = rLayer->width;
        layerPitch[l] = rLayer->pitch;
        for (auto& track : rLayer->tracks) {
            for (int i = 0; i < track.num; i++) {
                tracks[l].emplace_back(i * track.step + track.start);
            }
        }
        sort(tracks[l].begin(), tracks[l].end());
        tracks[l].erase(unique(tracks[l].begin(), tracks[l].end()), tracks[l].end());
    }
    m1direction = rawdb.getRLayer(0)->direction == 'v' ? 1 : 0;
    microns = rawdb.LefConvertFactor;
    if (nLayers > 1) {
        m2pitch = layerPitch[1];
    } else {
        m2pitch = layerPitch[0];
    }

    maxEOLSpacingVec.resize(nLayers, 0);
    maxEOLWidthVec.resize(nLayers, 0);
    defaultSpacing.resize(nLayers, 0);
    for (int l = 0; l < nLayers; l++) {
        auto rLayer = rawdb.getRLayer(l);
        int sp0 = rLayer->spacing;
        auto [sp1, width1, within1] = rLayer->maxEOLSpace;
        auto [sp2, width2, within2, parSpace2, parWithin2] = rLayer->maxEOLSpaceParallelEdge;
        maxEOLSpacingVec[l] = std::max(maxEOLSpacingVec[l], sp0);
        maxEOLSpacingVec[l] = std::max(maxEOLSpacingVec[l], sp1);
        maxEOLSpacingVec[l] = std::max(maxEOLSpacingVec[l], sp2);
        maxEOLWidthVec[l] = std::max(maxEOLWidthVec[l], width1);
        maxEOLWidthVec[l] = std::max(maxEOLWidthVec[l], width2);
        defaultSpacing[l] = getParallelRunSpace(l, layerWidth[l], 0);
    }

    // 2) init gcellgrid
    // 2.1) grid lines
    db::GCellGrid& gcellgrid = rawdb.gcellgrid;
    gridlines.resize(2);
    if (!gcellgrid.numX.size() || !gcellgrid.numY.size()) {
        if (grSetting.routeXSize <= 0 || grSetting.routeYSize <= 0) {
            grSetting.routeXSize = 512;
            grSetting.routeYSize = 512;
        }
    }
    if (grSetting.routeXSize <= 0 || grSetting.routeYSize <= 0) {
        gridlines[0].emplace_back(0);
        for (int idx = 0; idx < gcellgrid.numX.size(); idx++) {
            for (int i = 1; i < gcellgrid.numX[idx]; i++) {
                gridlines[0].emplace_back(gcellgrid.startX[idx] + i * gcellgrid.stepX[idx]);
            }
        }

        gridlines[1].emplace_back(0);
        for (int idx = 0; idx < gcellgrid.numY.size(); idx++) {
            for (int i = 1; i < gcellgrid.numY[idx]; i++) {
                gridlines[1].emplace_back(gcellgrid.startY[idx] + i * gcellgrid.stepY[idx]);
            }
        }
    } else {
        int stepX = rawdb.dieHX / grSetting.routeXSize;
        for (int i = 0; i < grSetting.routeXSize; i++) {
            gridlines[0].emplace_back(i * stepX);
        }
        gridlines[0].emplace_back(rawdb.dieHX);

        int stepY = rawdb.dieHY / grSetting.routeYSize;
        for (int i = 0; i < grSetting.routeYSize; i++) {
            gridlines[1].emplace_back(i * stepY);
        }
        gridlines[1].emplace_back(rawdb.dieHY);
    }
    sort(gridlines[0].begin(), gridlines[0].end());
    xSize = gridlines[0].size() - 1;
    sort(gridlines[1].begin(), gridlines[1].end());
    ySize = gridlines[1].size() - 1;

    if (grSetting.routeXSize <= 0 || grSetting.routeYSize <= 0) {
        int largeNumX = -1, largeNumY = -1;
        for (int idx = 0; idx < gcellgrid.numX.size(); idx++) {
            if (largeNumX < gcellgrid.numX[idx]) {
                largeNumX = gcellgrid.numX[idx];
                mainGcellStepX = gcellgrid.stepX[idx];
            }
        }
        for (int idx = 0; idx < gcellgrid.numY.size(); idx++) {
            if (largeNumY < gcellgrid.numY[idx]) {
                largeNumY = gcellgrid.numY[idx];
                mainGcellStepY = gcellgrid.stepY[idx];
            }
        }
    } else {
        mainGcellStepX = rawdb.dieHX / grSetting.routeXSize;
        mainGcellStepY = rawdb.dieHY / grSetting.routeYSize;
    }

    // 2.2) grid center points
    gridCenters.resize(2);
    for (unsigned dir = 0; dir <= 1; dir++) {
        gridCenters[dir].resize(gridlines[dir].size() - 1);
        for (int gidx = 0; gidx < gridlines[dir].size() - 1; gidx++) {
            gridCenters[dir][gidx] = (gridlines[dir][gidx] + gridlines[dir][gidx + 1]) / 2;
        }
    }
    nMaxGrid = (std::max(xSize, ySize) + 31) / 32 * 32;
    if (std::max(xSize, ySize) % 32 == 0) {
        nMaxGrid += 32;
    }
    gridGraphSize = nMaxGrid * nMaxGrid * nLayers;

    cgxsize = (xSize + csrnScale - 1) / csrnScale;
    cgysize = (ySize + csrnScale - 1) / csrnScale;

    logger.info("GridGraph (%d x %d x %d) CG SCALE = %d (%d x %d) nMaxGrid = %d",
                nLayers,
                xSize,
                ySize,
                csrnScale,
                cgxsize,
                cgysize,
                nMaxGrid);

    // 3) init routing capacity and routing wire distance
    setupCapacity();
    setupWireDist();
    // 4) init obs and mark obs
    setupObs();
    // 5) init gr nets
    setupGrNets();
    logger.info("Finish setting up grdb");
}

void GRDatabase::setupCapacity() {
    if (db::setting.BookshelfVariety != "") {
        return setupCapacityBookshelf();
    }
    capacity.resize(gridGraphSize, 0);
    for (int i = 0; i < nLayers; i++) {
        if ((i & 1) ^ m1direction) {
            for (int j = 0; j < xSize; j++) {
                int cap = lower_bound(tracks[i].begin(), tracks[i].end(), gridlines[0][j + 1]) -
                          lower_bound(tracks[i].begin(), tracks[i].end(), gridlines[0][j]);
                for (int k = 0; k < ySize; k++) {
                    capacity[encodeId(i, j, k)] = cap;
                }
            }
        } else {
            for (int k = 0; k < ySize; k++) {
                int cap = lower_bound(tracks[i].begin(), tracks[i].end(), gridlines[1][k + 1]) -
                          lower_bound(tracks[i].begin(), tracks[i].end(), gridlines[1][k]);
                for (int j = 0; j < xSize; j++) {
                    capacity[encodeId(i, j, k)] = cap;
                }
            }
        }
    }
}

void GRDatabase::setupCapacityBookshelf() {
    capacity.resize(gridGraphSize, 0);
    for (int i = 0; i < nLayers; i++) {
        int oricap = std::max(rawdb.bsRouteInfo.capH[i], rawdb.bsRouteInfo.capV[i]);
        float cap = oricap / (layerPitch[i]);
        if ((i & 1) ^ m1direction) {
            for (int j = 0; j < xSize; j++) {
                for (int k = 0; k < ySize; k++) {
                    capacity[encodeId(i, j, k)] = cap;
                }
            }
        } else {
            for (int k = 0; k < ySize; k++) {
                for (int j = 0; j < xSize; j++) {
                    capacity[encodeId(i, j, k)] = cap;
                }
            }
        }
    }
}

void GRDatabase::setupWireDist() {
    wireDist.resize(gridGraphSize, 1e9);
    for (int i = 0; i < nLayers; i++) {
        for (int j = 0; j < xSize; j++) {
            for (int k = 0; k < ySize; k++) {
                int idx = encodeId(i, j, k);
                if ((i & 1) ^ m1direction) {
                    if (k + 1 < ySize) {
                        wireDist[idx] = 0.5 * (gridlines[1][k + 2] - gridlines[1][k]);
                    }
                } else {
                    if (j + 1 < xSize) {
                        wireDist[idx] = 0.5 * (gridlines[0][j + 2] - gridlines[0][j]);
                    }
                }
            }
        }
    }
}

void GRDatabase::setupObs() {
    addFixObs();
    addMovObs();
    updateUsageLength();
}

void GRDatabase::updateUsageLength() {
    fixedUsage.resize(gridGraphSize, 0);
    fixedLength.resize(gridGraphSize, 0);

    int obsStartLayer = 1;
    for (int l = obsStartLayer; l < nLayers; l++) {
        int dir = (l & 1) ^ m1direction;
        int outerSize = (dir == 0 ? ySize : xSize);
        int innerSize = (dir == 0 ? xSize : ySize);
        for (int i = 0; i < outerSize; i++) {
            for (int j = 0; j < innerSize; j++) {
                int idx = l * nMaxGrid * nMaxGrid + i * nMaxGrid + j;
                fixedUsage[idx] = fixTmpUsage[idx] + movTmpUsage[idx];
                if (fixedUsage[idx] > 0.01) {
                    fixedLength[idx] = (fixTmpLength[idx] + movTmpLength[idx]) / fixedUsage[idx];
                }
            }
        }
    }
}

void GRDatabase::addFixObs() {
    logger.info("Marking fixed cell obs...");
    fixObs.clear();
    // 1) add IOPins
    for (auto iopin : rawdb.iopins) {
        if (iopin->type->shapes.size() > 0) {
            int posx = iopin->x;
            int posy = iopin->y;
            for (auto& shape : iopin->type->shapes) {
                auto [olx, oly, ohx, ohy] = getOrientOffset(iopin->orient(), shape.lx, shape.ly, shape.hx, shape.hy);
                int lx = posx + olx;
                int ly = posy + oly;
                int hx = posx + ohx;
                int hy = posy + ohy;
                fixObs.emplace_back(shape.layer.rIndex, lx, ly, hx, hy);
            }
        }
    }
    // 2) add SNets wires and vias
    for (auto snet : rawdb.snets) {
        for (auto shape : snet->shapes) {
            fixObs.emplace_back(shape.layer.rIndex, shape.lx, shape.ly, shape.hx, shape.hy);
        }
        for (auto via : snet->vias) {
            db::ViaRule& rule = via.type->rule;
            if (rule.hasViaRule) {
                int lenx = rule.cutSize.first * rule.numCutCols + rule.cutSpacing.first * (rule.numCutCols - 1);
                int leny = rule.cutSize.second * rule.numCutRows + rule.cutSpacing.second * (rule.numCutRows - 1);
                int dx = lenx / 2 + rule.botEnclosure.first;
                int dy = leny / 2 + rule.botEnclosure.second;
                if (rule.botLayer->rIndex > 0) {
                    fixObs.emplace_back(rule.botLayer->rIndex, via.x - dx, via.y - dy, via.x + dx, via.y + dy);
                }
                dx = lenx / 2 + rule.topEnclosure.first;
                dy = leny / 2 + rule.topEnclosure.second;
                if (rule.topLayer->rIndex > 0) {
                    fixObs.emplace_back(rule.topLayer->rIndex, via.x - dx, via.y - dy, via.x + dx, via.y + dy);
                }
            }
        }
    }
    // 3) Routing blkgs
    for (auto& blkg : rawdb.routeBlockages) {
        fixObs.emplace_back(blkg.layer.rIndex, blkg.lx, blkg.ly, blkg.hx, blkg.hy);
    }
    // 4) Fixed nodes
    for (auto cell : rawdb.cells) {
        if (cell->fixed()) {
            addCellObs(fixObs, cell);
        }
    }

    // update usage and length
    markObs(fixObs, fixTmpUsage, fixTmpLength);
}

void GRDatabase::addMovObs() {
    logger.info("Marking movable cell obs...");
    movObs.clear();
    for (auto cell : rawdb.cells) {
        if (!cell->fixed()) {
            addCellObs(movObs, cell);
        }
    }
    markObs(movObs, movTmpUsage, movTmpLength);
}

void GRDatabase::addCellObs(std::vector<RectOnLayer>& allObs, db::Cell* cell) {
    db::CellType* ctype = cell->ctype();
    int cellOrient = cell->orient();
    int dx = ctype->originX() + cell->lx();
    int dy = ctype->originY() + cell->ly();
    // Macro Obs
    for (auto& e : ctype->obs()) {
        if (e.layer.rIndex <= 0) continue;  // ignore M1 OBS and non-routing layer
        int lx = e.lx, ly = e.ly, hx = e.hx, hy = e.hy;
        switch (cellOrient) {
            case 2:  // S
                lx = ctype->width - e.hx;
                ly = ctype->height - e.hy;
                hx = ctype->width - e.lx;
                hy = ctype->height - e.ly;
                break;
            case 4:  // FN
                lx = ctype->width - e.hx;
                hx = ctype->width - e.lx;
                break;
            case 6:  // FS
                ly = ctype->height - e.hy;
                hy = ctype->height - e.ly;
                break;
            default:
                break;
        }
        allObs.emplace_back(e.layer.rIndex, lx + dx, ly + dy, hx + dx, hy + dy);
    }
    // Pin Box
    for (auto pintype : ctype->pins) {
        for (auto& e : pintype->shapes) {
            if (e.layer.rIndex <= 0) continue;  // ignore M1 OBS and non-routing layer
            int lx = e.lx, ly = e.ly, hx = e.hx, hy = e.hy;
            switch (cellOrient) {
                case 2:  // S
                    lx = ctype->width - e.hx;
                    ly = ctype->height - e.hy;
                    hx = ctype->width - e.lx;
                    hy = ctype->height - e.ly;
                    break;
                case 4:  // FN
                    lx = ctype->width - e.hx;
                    hx = ctype->width - e.lx;
                    break;
                case 6:  // FS
                    ly = ctype->height - e.hy;
                    hy = ctype->height - e.ly;
                    break;
                default:
                    break;
            }
            allObs.emplace_back(e.layer.rIndex, lx + dx, ly + dy, hx + dx, hy + dy);
        }
    }
}

std::tuple<int, int, int, int> GRDatabase::getOrientOffset(int orient, int lx, int ly, int hx, int hy) {
    std::tuple<int, int, int, int> offset;  // lx, ly, hx, hy
    // 0:N, 1:W, 2:S, 3:E, 4:FN, 5:FW, 6:FS, 7:FE, -1:NONE
    switch (orient) {
        case 0:  // N
            offset = {lx, ly, hx, hy};
            break;
        case 1:  // W
            offset = {-hy, lx, -ly, hx};
            break;
        case 2:  // S
            offset = {-hx, -hy, -lx, -ly};
            break;
        case 3:  // E
            offset = {ly, -hx, hy, -lx};
            break;
        case 4:  // FN
            offset = {-hx, ly, -lx, hy};
            break;
        case 5:  // FW
            offset = {ly, lx, hy, hx};
            break;
        case 6:  // FS
            offset = {lx, -hy, hx, -ly};
            break;
        case 7:  // FE
            offset = {-hy, -hx, -ly, -lx};
            break;
        default:
            offset = {lx, ly, hx, hy};
            break;
    }
    return offset;
}

int GRDatabase::encodeId(int l, int x, int y) {
    if (!(l & 1) ^ m1direction) std::swap(x, y);
    return l * nMaxGrid * nMaxGrid + x * nMaxGrid + y;
}

int GRDatabase::getEOLSpace(int width, int l) { return (width < maxEOLWidthVec[l]) ? maxEOLSpacingVec[l] : 0; }

int GRDatabase::getParallelRunSpace(int l, int width, int length) {
    auto rLayer = rawdb.getRLayer(l);
    if (rLayer->parWidth.size() == 0) return 0;  // TODO: default values ?
    int iWidth = rLayer->parWidth.size() - 1;
    while (iWidth > 0 && rLayer->parWidth[iWidth] >= width) iWidth--;
    int iLength = rLayer->parLength.size() - 1;
    while (iLength > 0 && rLayer->parLength[iLength] >= length) iLength--;
    return rLayer->parWidthSpace[iWidth][iLength];
}

utils::PointT<int> GRDatabase::getObsMargin(RectOnLayer box, AggrParaRunSpace aggr) {
    utils::PointT<int> margin;
    for (int dir = 0; dir < 2; dir++) {
        int range = box.getDirRange(1 - dir);
        int space = getEOLSpace(range, box.layer);
        if (!space) {
            int length = 0;
            if (aggr == AggrParaRunSpace::LARGER_LENGTH && range > 100 * layerPitch[box.layer]) {
                length = layerPitch[box.layer] * 2 + layerWidth[box.layer];
            }
            space = getParallelRunSpace(box.layer, std::min(box.hx - box.lx, box.hy - box.ly), length);
        }
        margin[dir] = space + layerWidth[box.layer] / 2 - ISPD19;
    }
    return margin;
}

utils::IntervalT<int> GRDatabase::rangeSearchTracks(const utils::IntervalT<int>& locRange, int layerIdx) {
    auto& t = tracks[layerIdx];
    int lpos = lower_bound(t.begin(), t.end(), locRange.low) - t.begin();
    lpos = std::min(static_cast<int>(t.size()) - 1, lpos);
    while (lpos > 0 && t[lpos - 1] >= locRange.low) lpos--;
    int hpos = upper_bound(t.begin(), t.end(), locRange.high) - t.begin() - 1;
    hpos = std::max(hpos, 0);
    return utils::IntervalT<int>(lpos, hpos);
}

void GRDatabase::markObs(std::vector<RectOnLayer>& allObs,
                         std::vector<float>& wireUsage,
                         std::vector<float>& wireTotalLength) {
    if (db::setting.BookshelfVariety != "") {
        return markObsBookShelf(allObs, wireUsage, wireTotalLength);
    }
    int obsStartLayer = 1;

    wireUsage.resize(gridGraphSize, 0);
    wireTotalLength.resize(gridGraphSize, 0);

    std::vector<std::vector<int>> layerToObjIdx(nLayers);
    for (unsigned i = 0; i < allObs.size(); i++) {
        int l = allObs[i].layer;
        if (l < obsStartLayer) continue;
        layerToObjIdx[allObs[i].layer].push_back(i);
    }
    for (int l = obsStartLayer; l < nLayers; l++) {
        auto& t = tracks[l];
        int dir = (l & 1) ^ m1direction;

        std::vector<std::vector<std::vector<std::pair<utils::IntervalT<int>, int>>>> markingBufferLUT;

        auto searchLowerBoundTrack = [&](int p) {
            int pos = lower_bound(t.begin(), t.end(), p) - t.begin();
            pos = std::min(static_cast<int>(t.size()) - 1, pos);
            while (pos > 0 && t[pos - 1] >= p) pos--;
            return pos;
        };

        markingBufferLUT.resize((dir == 0 ? ySize : xSize));
        int lutInnerSize = (dir == 0 ? xSize : ySize);
        for (auto& e : markingBufferLUT) {
            e.resize(lutInnerSize);
        }

        for (auto idx : layerToObjIdx[l]) {
            const auto& curObs = allObs[idx];

            AggrParaRunSpace aggr = ISPD19 ? AggrParaRunSpace::LARGER_LENGTH : AggrParaRunSpace::LARGER_WIDTH;
            utils::PointT<int> margin = getObsMargin(curObs, aggr);
            utils::BoxT<int> obsBox(
                curObs.lx - margin.x, curObs.ly - margin.y, curObs.hx + margin.x, curObs.hy + margin.y);

            if (obsBox.IsValid()) {
                if (obsBox.hx() <= gridlines[0][0] || obsBox.hy() <= gridlines[1][0] ||
                    obsBox.lx() >= gridlines[0][gridlines[0].size() - 1] ||
                    obsBox.ly() >= gridlines[1][gridlines[1].size() - 1]) {
                    logger.verbose("ignore obs that is outside gridgraph, obsBox: %d %d %d %d",
                                   obsBox.lx(),
                                   obsBox.hx(),
                                   obsBox.ly(),
                                   obsBox.hy());
                    continue;
                }
            }

            int xmin =
                std::upper_bound(gridlines[0].begin(), gridlines[0].end(), obsBox.lx()) - gridlines[0].begin() - 1;
            int xmax =
                std::lower_bound(gridlines[0].begin(), gridlines[0].end(), obsBox.hx()) - gridlines[0].begin() - 1;
            int ymin =
                std::upper_bound(gridlines[1].begin(), gridlines[1].end(), obsBox.ly()) - gridlines[1].begin() - 1;
            int ymax =
                std::lower_bound(gridlines[1].begin(), gridlines[1].end(), obsBox.hy()) - gridlines[1].begin() - 1;
            xmin = std::max(xmin, 0);
            ymin = std::max(ymin, 0);
            xmax = std::min(xmax, xSize - 1);
            ymax = std::min(ymax, ySize - 1);
            if (xmin > xmax || ymin > ymax) {
                logger.error("continue, obs: %d %d %d %d, obsBox: %d %d %d %d",
                             xmin,
                             xmax,
                             ymin,
                             ymax,
                             obsBox.lx(),
                             obsBox.hx(),
                             obsBox.ly(),
                             obsBox.hy());
                continue;
            }
            utils::BoxT<int> grBox(xmin, ymin, xmax, ymax);

            utils::IntervalT<int> trackIntvl = rangeSearchTracks(obsBox[1 - dir], l);
            if (!trackIntvl.IsValid()) continue;

            int jmin = std::max(grBox[dir].low - 1, 0);
            int jmax = std::min(grBox[dir].high, (dir == 0 ? xSize : ySize) - 2);
            for (int i = grBox[1 - dir].low; i <= grBox[1 - dir].high; i++) {
                utils::IntervalT<int> gridTrackIntvl(searchLowerBoundTrack(gridlines[1 - dir][i]),
                                                     searchLowerBoundTrack(gridlines[1 - dir][i + 1]) - 1);
                for (int j = jmin; j <= jmax; j++) {
                    utils::IntervalT<int> edgeIntvl = {gridCenters[dir][j], gridCenters[dir][j + 1]};
                    auto blockedLen = obsBox[dir].IntersectWith(edgeIntvl).range();
                    if (blockedLen > 0) {
                        utils::IntervalT<int> blockedIntvl = gridTrackIntvl.IntersectWith(trackIntvl);
                        if (blockedIntvl.IsValid()) {
                            markingBufferLUT[i][j].emplace_back(blockedIntvl, blockedLen);
                        }
                    }
                }
            }
        }

        for (int i = 0; i < markingBufferLUT.size(); i++) {
            utils::IntervalT<int> gridTrackIntvl;
            gridTrackIntvl.low = searchLowerBoundTrack(gridlines[1 - dir][i]);
            gridTrackIntvl.high = searchLowerBoundTrack(gridlines[1 - dir][i + 1]) - 1;
            for (int j = 0; j < markingBufferLUT[i].size(); j++) {
                if (markingBufferLUT[i][j].size() == 0) continue;
                const auto& buf = markingBufferLUT[i][j];
                std::vector<int> trackBlocked(gridTrackIntvl.range() + 1, 0);  // blocked track length
                for (auto& pair : buf) {
                    for (int k = pair.first.low; k <= pair.first.high; k++) {
                        trackBlocked[k - gridTrackIntvl.low] += pair.second;
                    }
                }
                int nBlocked = 0;
                int totalBlockedLen = 0;
                for (auto& len : trackBlocked) {
                    if (len > 0) {
                        nBlocked++;
                        totalBlockedLen += len;
                    }
                }
                wireUsage[l * nMaxGrid * nMaxGrid + i * nMaxGrid + j] = nBlocked;
                wireTotalLength[l * nMaxGrid * nMaxGrid + i * nMaxGrid + j] = totalBlockedLen;
            }
        }
    }
}

void GRDatabase::markObsBookShelf(std::vector<RectOnLayer>& allObs,
                                  std::vector<float>& wireUsage,
                                  std::vector<float>& wireTotalLength) {
    int obsStartLayer = 1;

    wireUsage.resize(gridGraphSize, 0);
    wireTotalLength.resize(gridGraphSize, 0);

    std::vector<std::vector<int>> layerToObjIdx(nLayers);
    for (unsigned i = 0; i < allObs.size(); i++) {
        int l = allObs[i].layer;
        if (l < obsStartLayer) continue;
        layerToObjIdx[allObs[i].layer].push_back(i);
    }

    std::vector<float> layer2oricap(nLayers, 0.0);
    for (int i = 0; i < nLayers; i++) {
        int oricap = std::max(rawdb.bsRouteInfo.capH[i], rawdb.bsRouteInfo.capV[i]);
        float cap = oricap / (layerPitch[i]);
        layer2oricap[i] = oricap;
    }
    // ignore M1 obs
    for (int l = obsStartLayer; l < nLayers; l++) {
        int dir = (l & 1) ^ m1direction;

        std::vector<std::vector<std::vector<utils::IntervalT<int>>>> markingBufferLUT;
        markingBufferLUT.resize((dir == 0 ? ySize : xSize));
        int lutInnerSize = (dir == 0 ? xSize : ySize);
        for (auto& e : markingBufferLUT) {
            e.resize(lutInnerSize);
        }

        for (auto idx : layerToObjIdx[l]) {
            const auto& curObs = allObs[idx];

            utils::BoxT<int> obsBox(curObs.lx, curObs.ly, curObs.hx, curObs.hy);

            int xmin =
                std::upper_bound(gridlines[0].begin(), gridlines[0].end(), obsBox.lx()) - gridlines[0].begin() - 1;
            int xmax =
                std::lower_bound(gridlines[0].begin(), gridlines[0].end(), obsBox.hx()) - gridlines[0].begin() - 1;
            int ymin =
                std::upper_bound(gridlines[1].begin(), gridlines[1].end(), obsBox.ly()) - gridlines[1].begin() - 1;
            int ymax =
                std::lower_bound(gridlines[1].begin(), gridlines[1].end(), obsBox.hy()) - gridlines[1].begin() - 1;
            xmin = std::max(xmin, 0);
            ymin = std::max(ymin, 0);
            xmax = std::min(xmax, xSize - 1);
            ymax = std::min(ymax, ySize - 1);
            if (xmin > xmax || ymin > ymax) {
                logger.error("continue obs %d %d %d %d", xmin, xmax, ymin, ymax);
                continue;
            }
            utils::BoxT<int> grBox(xmin, ymin, xmax, ymax);
            int jmin = std::max(grBox[dir].low - 1, 0);
            int jmax = std::min(grBox[dir].high, (dir == 0 ? xSize : ySize) - 2);
            for (int i = grBox[1 - dir].low; i <= grBox[1 - dir].high; i++) {
                utils::IntervalT<int> gridIntvl(gridlines[1 - dir][i], gridlines[1 - dir][i + 1]);
                utils::IntervalT<int> blockedIntvl = gridIntvl.IntersectWith(obsBox[1 - dir]);
                if (!blockedIntvl.IsValid()) continue;
                if (blockedIntvl.range() == 0) continue;
                for (int j = jmin; j <= jmax; j++) {
                    int edgePos = gridlines[dir][j + 1];
                    if (obsBox[dir].Contain(edgePos)) {
                        markingBufferLUT[i][j].emplace_back(blockedIntvl);
                    }
                }
            }
        }
        for (int i = 0; i < markingBufferLUT.size(); i++) {
            for (int j = 0; j < markingBufferLUT[i].size(); j++) {
                if (markingBufferLUT[i][j].size() == 0) continue;
                utils::IntervalT<int> gridIntvl(gridlines[1 - dir][i], gridlines[1 - dir][i + 1]);
                std::vector<utils::IntervalT<int>>& buf = markingBufferLUT[i][j];
                int ovlpLen = 0;
                if (buf.size() > 1) {
                    std::stable_sort(
                        buf.begin(), buf.end(), [](const utils::IntervalT<int>& lhs, const utils::IntervalT<int>& rhs) {
                            return lhs.low < rhs.low;
                        });
                    utils::IntervalT<int> tmpIntvl(buf[0].low, buf[0].high);
                    for (int bufIdx = 1; bufIdx < buf.size(); bufIdx++) {
                        utils::IntervalT<int>& curIntvl = buf[bufIdx];
                        if (curIntvl.low == tmpIntvl.low) {
                            tmpIntvl.high = std::max(tmpIntvl.high, curIntvl.high);
                        } else if (curIntvl.low > tmpIntvl.low) {
                            if (curIntvl.low <= tmpIntvl.high) {
                                tmpIntvl.high = std::max(tmpIntvl.high, curIntvl.high);
                            } else {
                                ovlpLen += tmpIntvl.range();
                                tmpIntvl.low = curIntvl.low;
                                tmpIntvl.high = curIntvl.high;
                            }
                        } else {
                            logger.error(
                                "continue Intvl %d %d %d %d", tmpIntvl.low, tmpIntvl.high, curIntvl.low, curIntvl.high);
                            continue;
                        }
                    }
                    ovlpLen += tmpIntvl.range();
                } else {
                    ovlpLen = buf[0].range();
                }
                // Follow perl script dac2012_evaluate_solution.pl
                float blocked = floor((float)ovlpLen * (1.0 - rawdb.bsRouteInfo.blockagePorosity));
                float availableSpace = ((float)gridIntvl.range() - blocked) / (float)gridIntvl.range();
                int adjustedCap = layer2oricap[l] * availableSpace;
                adjustedCap = std::max(0, adjustedCap);
                int numTracksAvailable = adjustedCap / layerPitch[l];
                float bcount = capacity[l * nMaxGrid * nMaxGrid + i * nMaxGrid + j] - (float)numTracksAvailable;
                // Assign value, we suppose tracks are completely blocked
                wireUsage[l * nMaxGrid * nMaxGrid + i * nMaxGrid + j] =
                    std::min(bcount, capacity[l * nMaxGrid * nMaxGrid + i * nMaxGrid + j]);
                wireTotalLength[l * nMaxGrid * nMaxGrid + i * nMaxGrid + j] =
                    wireDist[l * nMaxGrid * nMaxGrid + i * nMaxGrid + j] *
                    wireUsage[l * nMaxGrid * nMaxGrid + i * nMaxGrid + j];
            }
        }
    }
}

void GRDatabase::setupGrNets() {
    grNets.resize(rawdb.nets.size());
    int tempcnt = 0, tempcnt2 = 0;
    auto thread_func = [&](int threadIdx) {
        for (int netId = threadIdx; netId < rawdb.nets.size(); netId += db::setting.numThreads) {
            db::Net* rawdbNet = rawdb.nets[netId];
            std::vector<std::vector<std::tuple<int, int, int>>> pinAccessPoints(rawdbNet->pins.size());
            for (size_t pinIdx = 0; pinIdx < rawdbNet->pins.size(); pinIdx++) {
                std::vector<RectOnLayer> pin_shapes;
                db::Pin* net_pin = rawdbNet->pins[pinIdx];
                if (net_pin->iopin != nullptr) {
                    db::IOPin* iopin = net_pin->iopin;
                    int lx = iopin->lx();
                    int ly = iopin->ly();
                    for (auto& shape : iopin->type->shapes) {
                        auto [olx, oly, ohx, ohy] =
                            getOrientOffset(iopin->orient(), shape.lx, shape.ly, shape.hx, shape.hy);
                        pin_shapes.emplace_back(shape.layer.rIndex, lx + olx, ly + oly, lx + ohx, ly + ohy);
                    }
                } else if (net_pin->cell != nullptr) {
                    db::Cell* cell = net_pin->cell;
                    db::CellType* ctype = cell->ctype();
                    int dx = cell->lx() + ctype->originX(), dy = cell->ly() + ctype->originY();
                    int cellOrient = cell->orient();
                    for (auto& e : net_pin->type->shapes) {
                        int lx = e.lx, ly = e.ly, hx = e.hx, hy = e.hy;
                        switch (cellOrient) {
                            case 2:  // S
                                lx = ctype->width - e.hx;
                                ly = ctype->height - e.hy;
                                hx = ctype->width - e.lx;
                                hy = ctype->height - e.ly;
                                break;
                            case 4:  // FN
                                lx = ctype->width - e.hx;
                                hx = ctype->width - e.lx;
                                break;
                            case 6:  // FS
                                ly = ctype->height - e.hy;
                                hy = ctype->height - e.ly;
                                break;
                            default:
                                break;
                        }
                        pin_shapes.emplace_back(e.layer.rIndex, lx + dx, ly + dy, hx + dx, hy + dy);
                    }
                } else {
                    continue;
                }
                std::set<std::tuple<int, int, int>> vis;
                for (int shapeIdx = 0; shapeIdx < pin_shapes.size(); shapeIdx++) {
                    auto& e = pin_shapes[shapeIdx];
                    int xmin =
                        std::upper_bound(gridlines[0].begin(), gridlines[0].end(), e.lx) - gridlines[0].begin() - 1;
                    int xmax =
                        std::lower_bound(gridlines[0].begin(), gridlines[0].end(), e.hx) - gridlines[0].begin() - 1;
                    int ymin =
                        std::upper_bound(gridlines[1].begin(), gridlines[1].end(), e.ly) - gridlines[1].begin() - 1;
                    int ymax =
                        std::lower_bound(gridlines[1].begin(), gridlines[1].end(), e.hy) - gridlines[1].begin() - 1;
                    // boundary check
                    int elayer = std::min(std::max(e.layer, 0), nLayers - 1);
                    xmin = std::min(std::max(xmin, 0), xSize - 1);
                    ymin = std::min(std::max(ymin, 0), ySize - 1);
                    xmax = std::min(std::max(xmax, 0), xSize - 1);
                    ymax = std::min(std::max(ymax, 0), ySize - 1);
                    if (xmin > xmax || ymin > ymax) {
                        std::string instName = "";
                        std::string instType = "";
                        if (net_pin->iopin != nullptr) {
                            db::IOPin* iopin = net_pin->iopin;
                            instName = iopin->name;
                            instType = iopin->type->name();
                        } else if (net_pin->cell != nullptr) {
                            db::Cell* cell = net_pin->cell;
                            instName = cell->name();
                            instType = cell->ctype()->name;
                        }
                        // NOTE: some benchmarks have strange definition of pin shapes (lx ly hx hy).
                        // For example, in ispd18_test9, one of ADDFHX2 CI shapes is "RECT 2.59 0.40 3.67 0.36".
                        logger.error(
                            "continue netId: %d netName: %s net_pinId: %d | instName: %s instType: %s pinName: %s "
                            "pinShapeId: %d | grid: %d %d %d %d | coord: %d %d %d %d",
                            netId,
                            rawdbNet->name.c_str(),
                            pinIdx,
                            instName.c_str(),
                            instType.c_str(),
                            net_pin->type->name().c_str(),
                            shapeIdx,
                            xmin,
                            xmax,
                            ymin,
                            ymax,
                            e.lx,
                            e.hx,
                            e.ly,
                            e.hy);
                        continue;
                    }
                    for (int x = xmin; x <= xmax; x++) {
                        for (int y = ymin; y <= ymax; y++) {
                            auto t = std::make_tuple(elayer, x, y);
                            if (vis.find(t) != vis.end()) continue;
                            vis.insert(t);
                            pinAccessPoints[pinIdx].emplace_back(t);
                        }
                    }
                }
            }
            int xmin = nMaxGrid - 1, ymin = nMaxGrid - 1, xmax = 0, ymax = 0, lmin = nLayers - 1, lmax = 0;
            for (const auto& accessPoints : pinAccessPoints) {
                for (const auto [layer, x, y] : accessPoints) {
                    lmin = std::min(lmin, layer);
                    lmax = std::max(lmax, layer);
                    xmin = std::min(xmin, x);
                    xmax = std::max(xmax, x);
                    ymin = std::min(ymin, y);
                    ymax = std::max(ymax, y);
                }
            }
            int cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
            robin_hood::unordered_map<uint64_t, std::pair<utils::PointT<int>, utils::IntervalT<int>>>
                selectedAccessPoints;
            // std::map<uint64_t, std::pair<utils::PointT<int>, utils::IntervalT<int>>>
            //     selectedAccessPoints;
            robin_hood::unordered_map<uint64_t, std::vector<int>> accessPoint2pinIds;
            for (size_t pinIdx = 0; pinIdx < pinAccessPoints.size(); pinIdx++) {
                db::Pin* net_pin = rawdbNet->pins[pinIdx];
                const auto& accessPoints = pinAccessPoints[pinIdx];
                int minDistance = std::numeric_limits<int>::max();
                int bestIndex = -1;
                for (int index = 0; index < accessPoints.size(); index++) {
                    const auto [point_l, point_x, point_y] = accessPoints[index];
                    int distance = std::abs(cx - point_x) + std::abs(cy - point_y);
                    if (distance < minDistance) {
                        minDistance = distance;
                        bestIndex = index;
                    }
                }
                const auto [selected_l, selected_x, selected_y] = accessPoints[bestIndex];
                const utils::PointT<int> selectedPoint(selected_x, selected_y);
                const uint64_t hash = selectedPoint.x * ySize + selectedPoint.y;
                if (selectedAccessPoints.find(hash) == selectedAccessPoints.end()) {
                    selectedAccessPoints.emplace(hash, std::make_pair(selectedPoint, utils::IntervalT<int>()));
                }
                utils::IntervalT<int>& fixedLayerInterval = selectedAccessPoints[hash].second;
                for (const auto [point_l, point_x, point_y] : accessPoints) {
                    if (point_x == selectedPoint.x && point_y == selectedPoint.y) {
                        fixedLayerInterval.Update(point_l);
                    }
                }
                accessPoint2pinIds[hash].emplace_back(net_pin->gpdb_id);
            }
            for (auto& accessPoint : selectedAccessPoints) {
                utils::IntervalT<int>& fixedLayers = accessPoint.second.second;
                fixedLayers.high = std::min(fixedLayers.high + 2, nLayers - 1);
            }
            std::vector<std::vector<int>> grNetPins(selectedAccessPoints.size());
            std::vector<std::vector<int>> grPin2GpdbPins(selectedAccessPoints.size());
            size_t grNetPinId = 0;
            for (auto& accessPoint : selectedAccessPoints) {
                const uint64_t hash = accessPoint.first;
                const utils::IntervalT<int>& fixedLayers = accessPoint.second.second;
                grNetPins[grNetPinId].reserve(fixedLayers.range());
                int pinX = accessPoint.second.first.x, pinY = accessPoint.second.first.y;
                for (int pinL = fixedLayers.low; pinL <= fixedLayers.high; pinL++) {
                    grNetPins[grNetPinId].emplace_back(encodeId(pinL, pinX, pinY));
                }
                grPin2GpdbPins[grNetPinId] = std::move(accessPoint2pinIds[hash]);
                grNetPinId++;
            }
            grNets[netId].setBoundingBox(xmin, ymin, xmax, ymax);
            grNets[netId].setPins(grNetPins);
            grNets[netId].pin2gpdbPinIds = std::move(grPin2GpdbPins);
            if (!grNets[netId].needToRoute()) {
                grNets[netId].setNoRoute();
                // } else if (ymax - ymin <= 2 && xmax - xmin <= 2 && lmax - lmin <= 3) {
                //     grNets[netId].setNoRoute(), tempcnt++;
            } else if (ISPD18) {
                if (ymax - ymin <= 3 && xmax - xmin <= 3 && lmax - lmin <= 5) {
                    grNets[netId].setNoRoute(), tempcnt++;
                    // } else {
                    //     std::vector<int> fa(grNetPins.size());
                    //     std::function<int(int)> fu = [&] (int x) {
                    //         return x == fa[x] ? x : fa[x] = fu(fa[x]);
                    //     };
                    //     for(int i = 0; i < grNetPins.size(); i++)
                    //         fa[i] = i;
                    //     for(int i = 0; i < grNetPins.size(); i++)
                    //         for(int j = 0; j < grNetPins.size(); j++) if(fu(i) != fu(j)) {
                    //             for(auto e : dbnet.global_pins[i]) {
                    //                 int le = std::get<0> (e), xe = std::get<1> (e), ye = std::get<2> (e);
                    //                 for(auto f : dbnet.global_pins[j]) {
                    //                     int lf = std::get<0> (f), xf = std::get<1> (f), yf = std::get<2> (f);
                    //                     if(le + 3 < lf - 2 || lf + 3 < le - 2) continue;
                    //                     if(xe + 2 < xf - 1 || xf + 2 < xe - 1) continue;
                    //                     if(ye + 2 < yf - 1 || yf + 2 < ye - 1) continue;
                    //                     fa[fu(i)] = fu(j);
                    //                     break;
                    //                 }
                    //                 if(fu(i) == fu(j)) break;
                    //             }
                    //         }
                    //     int ok = 1;
                    //     for(int i = 1; i < grNetPins.size(); i++)
                    //         if(fu(i) != fu(0)) ok = 0;
                    //     if(ok)
                    //         grNets[netId].setNoRoute(), tempcnt2++;
                }
                // } else if (ymax - ymin <= 1 && xmax - xmin <= 1 && lmax - lmin <= 3) {
                //     grNets[netId].setNoRoute(), tempcnt++;
            }
        }
    };

    std::thread threads[db::setting.numThreads];
    for (int j = 0; j < db::setting.numThreads; j++) {
        threads[j] = std::thread(thread_func, j);
    }
    for (auto& t : threads) {
        t.join();
    }

    int gbpidId = 0;
    for (int netId = 0; netId < grNets.size(); netId++) {
        for (auto e : grNets[netId].getPins()) {
            grNets[netId].pin2gbpinId.emplace_back(gbpidId++);
        }
    }

    logger.info("INCORRECT noroute nets: %d %d", tempcnt, tempcnt2);
}

void GRDatabase::resetGrNetsRoute() {
    for (int netId = 0; netId < grNets.size(); netId++) {
        grNets[netId].resetRoute();
    }
}

std::pair<int, int> GRDatabase::reportGRStat() {
    int wirelength = 0;
    int numVias = 0;
    for (int netId = 0; netId < grNets.size(); netId++) {
        auto wires = grNets[netId].getWires();
        for (size_t i = 0; i < wires.size(); i += 2) {
            wirelength += wires[i + 1];
        }
        numVias += grNets[netId].getVias().size();
    }
    logger.info("GR wirelength: %d, #Vias: %d", wirelength, numVias);
    return std::make_pair(wirelength, numVias);
}

void GRDatabase::writeGuides(std::string outputFile) {
    constexpr int LLL = 500000000;

    logger.info("Writing guides to file %s", outputFile.c_str());
    FILE* file = fopen(outputFile.c_str(), "w");

    static char s[LLL];
    char temp[10];
    int cur = 0;
    std::vector<std::string> rlayerNames(rawdb.getNumRLayers());
    for (int l = 0; l < rawdb.getNumRLayers(); l++) {
        rlayerNames[l] = rawdb.getRLayer(l)->name();
    }
    auto number = [&](int num) {
        if (num == 0)
            s[cur++] = '0';
        else {
            int len = 0;
            while (num) temp[len++] = num % 10, num /= 10;
            for (int i = len - 1; i >= 0; i--) s[cur++] = temp[i] + '0';
        }
    };
    auto singleGuide = [&](int xmin, int xmax, int ymin, int ymax, int layer) {
        number(gridlines[0][xmin]);
        s[cur++] = ' ';
        number(gridlines[1][ymin]);
        s[cur++] = ' ';
        number(gridlines[0][xmax + 1]);
        s[cur++] = ' ';
        number(gridlines[1][ymax + 1]);
        s[cur++] = ' ';
        for (auto e : rlayerNames[layer]) {
            s[cur++] = e;
        }
        s[cur++] = '\n';
    };

    auto printGrGuides = [&](int netId) {
        auto wires = grNets[netId].getWires();
        for (size_t i = 0; i < wires.size(); i += 2) {
            int p = wires[i];
            int l = p / nMaxGrid / nMaxGrid, x = p % (nMaxGrid * nMaxGrid) / nMaxGrid, y = p % nMaxGrid;
            if (!(l & 1) ^ m1direction) std::swap(x, y);
            int xmin = x, xmax = x, ymin = y, ymax = y;
            if ((l & 1) ^ m1direction) {
                ymax += wires[i + 1];
            } else {
                xmax += wires[i + 1];
            }
            if (l >= nLayers || x + (!(l & 1) ^ m1direction) * wires[i + 1] >= xSize ||
                y + ((l & 1) ^ m1direction) * wires[i + 1] >= ySize) {
                logger.error("Net %d OUT OF BOUNDARY", netId);
                exit(0);
            }
            singleGuide(xmin, xmax, ymin, ymax, l);
        }
        auto vias = grNets[netId].getVias();
        for (auto p : vias) {
            int l = p / nMaxGrid / nMaxGrid, x = p % (nMaxGrid * nMaxGrid) / nMaxGrid, y = p % nMaxGrid;
            if (!(l & 1) ^ m1direction) std::swap(x, y);
            singleGuide(x, x, y, y, l);
            singleGuide(x, x, y, y, l + 1);
            if (l + 2 < nLayers) singleGuide(x, x, y, y, l + 2);
        }
        auto pins = grNets[netId].getPins();
        for (auto& temp : pins)
            for (auto& p : temp) {
                int l = p / nMaxGrid / nMaxGrid, x = p % (nMaxGrid * nMaxGrid) / nMaxGrid, y = p % nMaxGrid;
                if (!(l & 1) ^ m1direction) std::swap(x, y);
                // int  xmin = x, xmax = x, ymin = y, ymax = y;
                int lmin = std::max(0, l - 2), lmax = std::min(nLayers - 1, l + 2);
                int xmin = std::max(0, x - 1), xmax = std::min(xSize - 1, x + 1);
                int ymin = std::max(0, y - 1), ymax = std::min(ySize - 1, y + 1);
                for (int i = lmin; i <= lmax; i++) singleGuide(xmin, xmax, ymin, ymax, i);
            }
    };
    for (int netId = 0; netId < grNets.size(); netId++) {
        int rawdbNetId = gpdb.getNets()[netId].getOriDBId();
        for (auto e : rawdb.nets[rawdbNetId]->name) {
            s[cur++] = e;
        }
        s[cur++] = '\n';
        s[cur++] = '(';
        s[cur++] = '\n';
        printGrGuides(netId);
        s[cur++] = ')';
        s[cur++] = '\n';
        if (cur * 1.1 > LLL) {
            fwrite(s, sizeof(char), cur, file);
            cur = 0;
        }
        if (cur > LLL) {
            logger.error("LLL too small. Please increase LLL.");
            exit(0);
        }
    }
    fwrite(s, sizeof(char), cur, file);
    fclose(file);
}

}  // namespace gr