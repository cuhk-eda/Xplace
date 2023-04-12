#pragma once
#include "GRSetting.h"
#include "GrNet.h"
#include "common/common.h"
#include "common/db/Database.h"
#include "io_parser/gp/GPDatabase.h"

namespace gr {

enum class AggrParaRunSpace { DEFAULT, LARGER_WIDTH, LARGER_LENGTH };

class RectOnLayer {
public:
    int layer = -1;
    int lx, ly, hx, hy;

    RectOnLayer() {}
    RectOnLayer(const int layer_, const int lx_, const int ly_, const int hx_, const int hy_)
        : layer(layer_), lx(lx_), ly(ly_), hx(hx_), hy(hy_) {}

    int getDirRange(unsigned i) { return (i == 0) ? hx - lx : hy - ly; }
};

// class GRNet {
//     std::vector<std::vector<std::tuple<int, int, int>>> global_pins;
//     std::vector<std::vector<RectOnLayer>> pins;  // GCell locations
// };

class GRDatabase {
public:
    db::Database& rawdb;
    gp::GPDatabase& gpdb;

    // GR Obstacle
    std::vector<RectOnLayer> fixObs;
    std::vector<RectOnLayer> movObs;  // for movable nodes
    std::vector<std::vector<int>> tracks;

    // Metal Layer
    std::vector<int> layerWidth;
    std::vector<int> layerPitch;
    std::vector<int> defaultSpacing;
    std::vector<int> maxEOLSpacingVec;  // from all spacing types
    std::vector<int> maxEOLWidthVec;    // from all spacing types

    // Design
    int nLayers;
    int xSize;
    int ySize;
    int nMaxGrid;
    int gridGraphSize;
    int m1direction;  // layer 0 direction, 'v': 1, 'h': 0
    double m2pitch;
    int microns;

    int mainGcellStepX, mainGcellStepY;
    std::vector<std::vector<int>> gridlines;
    std::vector<std::vector<int>> gridCenters;

    int ISPD19 = 0;
    int ISPD18 = 0;
    int METAL5 = 0;
    int csrnScale = 0;
    int cgxsize, cgysize;

    // GR variables
    std::vector<float> capacity, wireDist;         // init once
    std::vector<float> fixTmpUsage, fixTmpLength;  // init once
    std::vector<float> movTmpUsage, movTmpLength;  // update dynamically
    std::vector<float> fixedUsage, fixedLength;    // variables for GR

    std::vector<GrNet> grNets;
    std::vector<int> gpdbPinId2gbPinId;

    GRDatabase(std::shared_ptr<db::Database> rawdb_, std::shared_ptr<gp::GPDatabase> gpdb_);
    ~GRDatabase();

    void setupCapacity();
    void setupCapacityBookshelf();
    void setupWireDist();

    void addFixObs();
    void addMovObs();
    void updateUsageLength();
    void setupObs();

    void setupGrNets();
    void resetGrNetsRoute();
    void resetGrNets() { grNets.clear(); };

    std::pair<int, int> reportGRStat();
    void writeGuides(std::string outputFile);

    int encodeId(int l, int x, int y);
    tuple<int, int, int, int> getOrientOffset(int orient, int lx, int ly, int hx, int hy);
    int getEOLSpace(int width, int l);
    int getParallelRunSpace(int l, int width, int length);
    utils::PointT<int> getObsMargin(RectOnLayer box, AggrParaRunSpace aggr);
    utils::IntervalT<int> rangeSearchTracks(const utils::IntervalT<int>& locRange, int layerIdx);
    void markObs(std::vector<RectOnLayer>& allObs, std::vector<float>& wireUsage, std::vector<float>& wireTotalLength);
    void markObsBookShelf(std::vector<RectOnLayer>& allObs,
                          std::vector<float>& wireUsage,
                          std::vector<float>& wireTotalLength);
    void addCellObs(std::vector<RectOnLayer>& allObs, db::Cell* cell);
};
}  // namespace gr