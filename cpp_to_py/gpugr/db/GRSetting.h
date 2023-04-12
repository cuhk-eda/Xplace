#pragma once
#include <string>

namespace gr {

class GRSetting {
public:
    // 1. SystemSetting
    int deviceId = 0;

    // 2. Gridgraph setting
    int routeXSize = 0;
    int routeYSize = 0;
    int csrnScale = 0;

    // 3. The number of Rip-up and Reroute iterations (if 0, only PR is invoked)
    int rrrIters = 0;

    std::string routeGuideFile = "";

    void reset();
};

extern GRSetting grSetting;
}  //   namespace gr
