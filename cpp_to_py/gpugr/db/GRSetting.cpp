#include "GRSetting.h"

namespace gr {

void GRSetting::reset() {
    deviceId = 0;

    routeXSize = 0;
    routeYSize = 0;
    csrnScale = 0;

    rrrIters = 0;

    routeGuideFile = ""; 
}

GRSetting grSetting;

}  //   namespace gr