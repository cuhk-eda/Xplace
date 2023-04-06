#include "common/db/Database.h"

using namespace db;

bool Database::readConstraints(const std::string& file) {
    string buffer;
    ifstream ifs(file.c_str());
    if (!ifs.good()) {
        logger.error("cannot open constraint file: %s", file.c_str());
        return false;
    }
    while (ifs >> buffer) {
        unsigned equal = buffer.find("=");
        string key = buffer.substr(0, equal);
        if (key == "maximum_utilization") {
            if (!maxDensity) {
                unsigned unit = buffer.find("%");
                string value = buffer.substr(equal + 1, unit - equal - 1);
                maxDensity = atof(value.c_str()) / 100.0;
            } else {
                logger.warning("use input max util %f", maxDensity);
            }
        } else if (key == "maximum_movement") {
            if (!maxDisp) {
                unsigned unit = buffer.find("rows");
                string value = buffer.substr(equal + 1, unit - equal - 1);
                maxDisp = atof(value.c_str());
            } else {
                logger.warning("use input max disp %f", maxDisp);
            }
        }
    }
    ifs.close();
    return true;
}

bool Database::readSize(const std::string& file) {
    ifstream fs(file.c_str());
    if (!fs.good()) {
        logger.error("cannot open size file: %s", file.c_str());
        return false;
    }

    if (siteW == 0 && rows.size() > 0) {
        siteW = rows[0]->xStep();
    }
    if (siteH == 0 && celltypes.size() > 0) {
        siteH = INT_MAX;
        for (CellType* celltype : celltypes) {
            if (!celltype->stdcell) {
                continue;
            }
            siteH = std::min(siteH, celltype->height);
        }
    }
    if (siteW == 0 || siteH == 0) {
        logger.info("not enough information to retrieve placement site size");
        return false;
    }

#ifndef NDEBUG
    logger.info("reading %s", file.c_str());
#endif

    std::set<CellType*> sized;
    do {
        string name;
        int w, h;
        fs >> name >> w >> h;
        if (name != "") {
            Cell* cell = getCell(name);
            if (cell == NULL) {
                logger.error("cell not found : %s", name.c_str());
                break;
            } else {
                CellType* celltype = cell->ctype();
                if (sized.find(celltype) == sized.end()) {
                    celltype->width = w * siteW;
                    celltype->height = h * siteH;
                    sized.insert(celltype);
                }
            }
        }
    } while (!fs.eof());

    fs.close();
    return true;
}
