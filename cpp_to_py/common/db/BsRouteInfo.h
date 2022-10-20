#pragma once

namespace db {
class BsRouteInfo {
    // specially restore ICCAD2012/DAC2012 bookshelf routing info
public:
    BsRouteInfo() {}
    bool hasInfo = false;
    double blockagePorosity = 0.0;
    vector<int> capV;  // numTracksV = capV[i] / layerPitch[i]
    vector<int> capH;
    vector<int> viaSpace;

private:
};
}  // namespace db
