#pragma once
#include "common/common.h"

namespace db {

class Setting {
public:
    // 1. SystemSetting
    int numThreads = 1;

    // 2. DBSetting
    bool EdgeSpacing = true;
    bool EnableFence = true;
    bool EnablePG = true;
    bool EnableIOPin = true;
    bool liteMode = true;
    bool random_place = false;

    // 3. IOSetting
    std::string Format = "";

    std::string BookshelfVariety = "";
    std::string BookshelfAux = "";
    std::string BookshelfPl = "";

    std::string DefFile = "";
    std::string LefFile = "";
    std::string LefCell = "";
    std::string LefTech = "";

    std::string Constraints = "";
    std::string Verilog = "";
    std::string Size = "";

    std::string OutputFile = "";

    void reset();
};

extern Setting setting;
}  //   namespace db
