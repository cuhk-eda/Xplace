#include "Database.h"

using namespace db;

/***** Database *****/
Database::Database() {
    clear();
    _buffer = new char[_bufferCapacity];
}

Database::~Database() {
    delete[] _buffer;
    _buffer = nullptr;
    clear();
    // for regions.push_back(new Region("default"));
    CLEAR_POINTER_LIST(regions);
    logger.info("destruct rawdb");
}

void Database::load() {
    // ----- design related options -----

    if (setting.BookshelfAux != "" && setting.BookshelfPl != "") {
        setting.Format = "bookshelf";
        readBSAux(setting.BookshelfAux, setting.BookshelfPl);
    }

    if (setting.LefFile != "") {
        setting.Format = "lefdef";
        readLEF(setting.LefFile);
    } else if ((setting.LefCell != "") && (setting.LefTech != "")) {
        setting.Format = "lefdef";
        readLEF(setting.LefTech);
        readLEF(setting.LefCell);
    }

    if (setting.DefFile != "") {
        setting.Format = "lefdef";
        readDEF(setting.DefFile);
        readDEFPG(setting.DefFile);
    }

    if (setting.Size != "") {
        readSize(setting.Size);
    }

    if (setting.Constraints != "") {
        readConstraints(setting.Constraints);
    }

    // verilog is unused now
    // if (setting.Verilog != "") {
    //     readVerilog(setting.Verilog);
    // }
    logger.info("Finish loading rawdb");
}

void Database::reset() {
    delete[] _buffer;
    _buffer = nullptr;
    clear();
    _buffer = new char[_bufferCapacity];
    name_celltypes.clear();
    name_cells.clear();
    name_nets.clear();
    name_iopins.clear();
    name_viatypes.clear();

    layers.clear();
    sites.clear();

    routeBlockages.clear();
    placeBlockages.clear();

    powerNet = PowerNet();

    siteMap = SiteMap();
    gcellgrid = GCellGrid();
    edgetypes = EdgeTypes();
    dbIssues.clear();

    dieLX = 0;
    dieLY = 0;
    dieHX = 0;
    dieHY = 0;

    coreLX = 0;
    coreLY = 0;
    coreHX = 0;
    coreHY = 0;

    siteW = 0;
    siteH = 0;
    nSitesX = 0;
    nSitesY = 0;

    maxDensity = 0;
    maxDisp = 0;
}

void Database::clear() {
    clearDesign();
    clearLibrary();
    clearTechnology();
}

void Database::clearTechnology() {
    CLEAR_POINTER_LIST(viatypes);
    CLEAR_POINTER_MAP(ndrs);
}

void Database::clearDesign() {
    CLEAR_POINTER_LIST(cells);
    CLEAR_POINTER_LIST(iopins);
    CLEAR_POINTER_LIST(nets);
    CLEAR_POINTER_LIST(rows);
    CLEAR_POINTER_LIST(regions);
    CLEAR_POINTER_LIST(snets);
    CLEAR_POINTER_LIST(tracks);

    DBU_Micron = -1.0;
    designName = "";
    regions.push_back(new Region("default"));
}

void Database::save(const std::string& given_prefix) {
    std::string filename;
    if (given_prefix != "") {
        filename = given_prefix;
        if (setting.Format == "bookshelf") {
            filename += ".pl";
        } else if (setting.Format == "lefdef") {
            filename += ".def";
        }
    } else {
        filename = setting.OutputFile;
    }
    if (setting.Format == "bookshelf" && filename != "") {
        writeBSPl(filename);
    } else if (setting.Format == "lefdef" && filename != "") {
        writeICCAD2017(setting.DefFile, filename);
    } else {
        std::cout << "[Error] Cannot save placement! filename: " << filename << " format: " << setting.Format
                  << std::endl;
    }
}

bool Database::placed() {
    for (Cell* cell : cells) {
        if (!cell->placed()) {
            return false;
        }
    }
    return true;
}

bool Database::detailedRouted() {
    for (Net* net : nets) {
        if (!net->detailedRouted()) {
            return false;
        }
    }
    return true;
}

bool Database::globalRouted() {
    for (Net* net : nets) {
        if (!net->globalRouted()) {
            return false;
        }
    }
    return true;
}

void Database::SetupLayers() {
#ifndef NDEBUG
    int maxRLayer = 0;
    int maxCLayer = 0;
    for (const Layer& layer : layers) {
        if (layer.isRouteLayer()) {
            assert(layer.rIndex == maxRLayer++);
        } else if (layer.isCutLayer()) {
            assert(layer.cIndex == maxCLayer++);
        }
    }
#endif
}

void Database::SetupCellLibrary() {
    for (auto celltype : celltypes) {
        if (!celltype->stdcell) {
            continue;
        }
        for (const PinType* pin : celltype->pins) {
            if (pin->type() != 'p' && pin->type() != 'g') {
                continue;
            }
            for (const Geometry& shape : pin->shapes) {
                if (shape.ly < 0 && shape.hy > 0) {
                    celltype->_botPower = pin->type();
                }
                if (shape.ly < celltype->height && shape.hy > celltype->height) {
                    celltype->_topPower = pin->type();
                }
            }
            if (celltype->_botPower != 'x' && celltype->_topPower != 'x') {
                break;
            }
        }
    }
}

/*
site height as the smallest row height
site width as the smallest step x in rows
*/
void Database::SetupFloorplan() {
    coreLX = INT_MAX;
    coreHX = INT_MIN;
    coreLY = INT_MAX;
    coreHY = INT_MIN;
    siteW = INT_MAX;
    siteH = INT_MAX;

    std::sort(rows.begin(), rows.end(), [](const Row* a, const Row* b) {
        return (a->y() == b->y()) ? (a->x() < b->x()) : (a->y() < b->y());
    });

    for (CellType* celltype : celltypes) {
        if (!celltype->stdcell) {
            continue;
        }
        siteH = std::min(siteH, celltype->height);
    }

    for (Row* row : rows) {
        siteW = std::min(siteW, row->xStep());
        coreLX = std::min(row->x(), coreLX);
        coreLY = std::min(row->y(), coreLY);
        coreHX = std::max(row->x() + (int)row->width(), coreHX);
        coreHY = std::max(row->y() + siteH, coreHY);
    }

    for (Site& site : sites) {
        if (site.siteClassName() == "CORE") {
            if (siteW != (unsigned)site.width()) {
                logger.warning("siteW %d in DEF is inconsistent with siteW %d in LEF.",
                               static_cast<int>(siteW),
                               static_cast<int>(site.width()));
            }
            if (siteH != site.height()) {
                logger.warning("siteH %d in DEF is inconsistent with siteH %d in LEF.",
                               static_cast<int>(siteH),
                               static_cast<int>(site.height()));
            }
            break;
        }
    }

    nSitesX = (coreHX - coreLX) / siteW;
    nSitesY = (coreHY - coreLY) / siteH;
    if (!maxDisp) {
        maxDisp = nSitesX;
    }
}

void Database::SetupRegions() {
    // setup default region
    regions[0]->addRect(coreLX, coreLY, coreHX, coreHY);

    if (!setting.EnableFence) {
        for (size_t i = 1; i < regions.size(); ++i) {
            delete regions[i];
        }
        regions.resize(1);
    }

    unsigned numRegions = regions.size();
    // setup region members
    // convert rect to horizontal slices
    for (unsigned char i = 0; i != numRegions; ++i) {
        regions[i]->id = i;
        regions[i]->resetRects();
    }
    // associate cells to regions according to cell name
    for (Region* region : regions) {
        //	each member group name
        for (string member : region->members) {
            if (member[member.length() - 1] == '*') {
                // member group name involve wildcard
                member = member.substr(0, member.length() - 1);
                for (Cell* cell : cells) {
                    if (cell->name().substr(0, member.length()) == member) {
                        cell->region = region;
                    }
                }
            } else {
                // member group name is a particular cell name
                Cell* cell = getCell(member);
                if (!cell) {
                    logger.error("cell name (%s) not found for group (%s)", member.c_str(), region->name().c_str());
                }
                cell->region = region;
            }
        }
    }
    // associate remaining cells to default region
    for (Cell* cell : cells) {
        if (!cell->region) {
            cell->region = regions[0];
        }
    }
}

void Database::SetupSiteMap() {
    // set up site map
    siteMap.siteL = coreLX;
    siteMap.siteR = coreHX;
    siteMap.siteB = coreLY;
    siteMap.siteT = coreHY;
    siteMap.siteStepX = siteW;
    siteMap.siteStepY = siteH;
    siteMap.siteNX = nSitesX;
    siteMap.siteNY = nSitesY;
    siteMap.initSiteMap(nSitesX, nSitesY);

    // mark site partially overlapped by fence
    int nRegions = regions.size();
    for (int i = 1; i < nRegions; i++) {
        // skipped the default region
        Region* region = regions[i];

        logger.verbose("region : %s", region->name().c_str());
        // partially overlap at left/right
        vector<Rectangle> hSlices = region->rects;
        Rectangle::sliceH(hSlices);
        for (int j = 0; j < (int)hSlices.size(); j++) {
            int lx = hSlices[j].lx;
            int hx = hSlices[j].hx;
            int olx = binOverlappedL(lx, siteMap.siteL, siteMap.siteR, siteMap.siteStepX);
            int ohx = binOverlappedR(hx, siteMap.siteL, siteMap.siteR, siteMap.siteStepX);
            int clx = binContainedL(lx, siteMap.siteL, siteMap.siteR, siteMap.siteStepX);
            int chx = binContainedR(hx, siteMap.siteL, siteMap.siteR, siteMap.siteStepX);
            int sly = binOverlappedL(hSlices[j].ly, siteMap.siteB, siteMap.siteT, siteMap.siteStepY);
            int shy = binOverlappedR(hSlices[j].hy, siteMap.siteB, siteMap.siteT, siteMap.siteStepY);
            if (olx != clx) {
                for (int y = sly; y <= shy; y++) {
                    siteMap.blockRegion(olx, y);
                }
            }
            if (ohx != chx) {
                for (int y = sly; y <= shy; y++) {
                    siteMap.blockRegion(ohx, y);
                }
            }
        }
        //  partially overlap at bottom/top
        vector<Rectangle> vSlices = region->rects;
        Rectangle::sliceV(vSlices);
        for (const Rectangle& vSlice : vSlices) {
            int ly = vSlice.ly;
            int hy = vSlice.hy;
            const unsigned oly = binOverlappedL(ly, siteMap.siteB, siteMap.siteT, siteMap.siteStepY);
            const unsigned ohy = binOverlappedR(hy, siteMap.siteB, siteMap.siteT, siteMap.siteStepY);
            const int cly = binContainedL(ly, siteMap.siteB, siteMap.siteT, siteMap.siteStepY);
            const int chy = binContainedR(hy, siteMap.siteB, siteMap.siteT, siteMap.siteStepY);
            const unsigned slx = binOverlappedL(vSlice.lx, siteMap.siteL, siteMap.siteR, siteMap.siteStepX);
            const unsigned shx = binOverlappedR(vSlice.hx, siteMap.siteL, siteMap.siteR, siteMap.siteStepX);
            if ((int)oly != cly) {
                for (unsigned x = slx; x <= shx; ++x) {
                    siteMap.blockRegion(x, oly);
                }
            }
            if ((int)ohy != chy) {
                for (unsigned x = slx; x <= shx; ++x) {
                    siteMap.blockRegion(x, ohy);
                }
            }
        }

        for (const Rectangle& rect : hSlices) {
            siteMap.setRegion(rect.lx, rect.ly, rect.hx, rect.hy, region->id);
        }
        for (const Rectangle& rect : vSlices) {
            siteMap.setRegion(rect.lx, rect.ly, rect.hx, rect.hy, region->id);
        }
    }

    // mark all sites blocked
    siteMap.setSites(coreLX, coreLY, coreHX, coreHY, SiteMap::SiteBlocked);

    // mark rows as non-blocked
    for (const Row* row : rows) {
        int lx = row->x();
        int ly = row->y();
        int hx = row->x() + row->width();
        int hy = row->y() + siteH;
        siteMap.unsetSites(lx, ly, hx, hy, SiteMap::SiteBlocked);
    }

    // mark blocked sites
    for (const Cell* cell : cells) {
        if (!cell->fixed()) {
            continue;
        }
        int lx = cell->lx();
        int ly = cell->ly();
        int hx = cell->hx();
        int hy = cell->hy();

        siteMap.setSites(lx, ly, hx, hy, SiteMap::SiteBlocked);
        siteMap.blockRegion(lx, ly, hx, hy);
    }

    for (const Rectangle& placeBlockage : placeBlockages) {
        int lx = placeBlockage.lx;
        int ly = placeBlockage.ly;
        int hx = placeBlockage.hx;
        int hy = placeBlockage.hy;
        siteMap.setSites(lx, ly, hx, hy, SiteMap::SiteBlocked);
        siteMap.blockRegion(lx, ly, hx, hy);
    }

    for (const SNet* snet : snets) {
        for (const Geometry& geo : snet->shapes) {
            switch (geo.layer.rIndex) {
                case 1:
                    if (geo.layer.direction == 'v') {
                        siteMap.setSites(geo.lx, geo.ly, geo.hx, geo.hy, SiteMap::SiteM2Blocked);
                    }
                    break;
                case 2:
                    siteMap.setSites(geo.lx, geo.ly, geo.hx, geo.hy, SiteMap::SiteM3Blocked);
                    break;
                default:
                    break;
            }
        }
    }

    for (const db::IOPin* iopin : iopins) {
        for (const Geometry& shape : iopin->type->shapes) {
            switch (shape.layer.rIndex) {
                case 0:
                    break;
                case 1:
                    siteMap.setSites(shape.lx + iopin->x,
                                     shape.ly + iopin->y,
                                     shape.hx + iopin->x,
                                     shape.hy + iopin->y,
                                     SiteMap::SiteM2BlockedIOPin);
                    break;
                case 2:
                    break;
                default:
                    break;
            }
        }
    }

    siteMap.nSites = siteMap.siteNX * siteMap.siteNY;
    siteMap.nPlaceable = 0;
    siteMap.nRegionSites.resize(regions.size(), 0);
    for (int y = 0; y < siteMap.siteNY; y++) {
        for (int x = 0; x < siteMap.siteNX; x++) {
            if (siteMap.getSiteMap(x, y, SiteMap::SiteBlocked) || siteMap.getSiteMap(x, y, SiteMap::SiteM2Blocked) ||
                siteMap.getSiteMap(x, y, SiteMap::SiteM2BlockedIOPin)) {
                continue;
            }
            siteMap.nPlaceable++;
            unsigned char region = siteMap.getRegion(x, y);
            if (region != Region::InvalidRegion) {
                siteMap.nRegionSites[region]++;
            }
        }
    }

    logger.verbose("core area: %ld", siteMap.nSites);
    logger.verbose(
        "placeable: %ld (%lf%%)", siteMap.nPlaceable, (double)siteMap.nPlaceable / (double)siteMap.nSites * 100.0);
    for (int i = 0; i < (int)regions.size(); i++) {
        logger.verbose("region %d : %ld (%lf%%)",
                       i,
                       siteMap.nRegionSites[i],
                       (double)siteMap.nRegionSites[i] / (double)siteMap.nPlaceable);
    }
}

void Database::SetupRows() {
    // verify row flipping conflict
    bool flipCheckPass = true;
    std::vector<char> flip(nSitesY, 0);
    for (Row* row : rows) {
        char isFlip = (row->flip() ? 1 : 2);
        int y = (row->y() - coreLY) / siteH;
        if (flip[y] == 0) {
            flip[y] = isFlip;
        } else if (flip[y] != isFlip) {
            logger.error("row flip conflict %d : %d", y, isFlip);
            flipCheckPass = false;
        }
    }

    if (!flipCheckPass) {
        logger.error("row flip checking fail");
    }

    if (rows.size() != nSitesY) {
        logger.error("resize rows %d->%d", (int)rows.size(), nSitesY);
        for (Row*& row : rows) {
            delete row;
            row = nullptr;
        }
    }
    rows.resize(nSitesY, nullptr);

    // NOTE: currently we only support one step size
    const int stepX = (coreHX - coreLX) / nSitesX;
    // NOTE: currently we only support horizontal row

    for (unsigned y = 0; y != nSitesY; ++y) {
        rows[y] = new Row("core_SITE_ROW_" + to_string(y), "core", coreLX, coreLY + y * siteH);
        rows[y]->xStep(stepX);
        rows[y]->yStep(0);
        rows[y]->xNum(nSitesX);
        rows[y]->yNum(1);
        rows[y]->flip(flip[y] == 1);
    }

    // set row power-rail
    bool topNormal = true;
    bool botNormal = true;
    bool shrNormal = true;
    for (unsigned y = 0; y < nSitesY; y++) {
        Row* row = rows[y];
        int ly = row->y();
        int hy = row->y() + siteH;
        if (!powerNet.getRowPower(ly, hy, row->_topPower, row->_botPower)) {
            if (topNormal && row->topPower() == 'x') {
                if (y + 1 == nSitesY) {
                    logger.warning("Top power rail of the row at y=%d is not connected to power rail", row->y());
                } else {
                    logger.error("Top power rail of the row at y=%d is not connected to power rail", row->y());
                    topNormal = false;
                }
            }
            if (botNormal && row->botPower() == 'x') {
                if (y) {
                    logger.error("Bottom power rail of the row at y=%d is not connected to power rail", row->y());
                    botNormal = false;
                } else {
                    logger.warning("Bottom power rail of the row at y=%d is not connected to power rail", row->y());
                }
            }
        }
        if (shrNormal && row->topPower() == row->botPower()) {
            logger.error(
                "Top and Bottom power rail of the row at y=%d share the same power %c", row->y(), row->topPower());
            shrNormal = false;
        }
    }
}

void Database::SetupRowSegments() {
    for (Row* row : rows) {
        int xL = row->getSiteL(this->coreLX, this->siteW);
        int xR = row->getSiteR(this->coreLX, this->siteW);
        int y = row->getSiteB(this->coreLY, this->siteH);
        RowSegment segment;
        bool b1 = true;
        bool b2 = true;
        Region* r1 = NULL;
        Region* r2 = NULL;
        for (int x = xL; x <= xR; x++) {
            if (x == xR) {
                b2 = true;
                r2 = NULL;
            } else {
                b2 = siteMap.getSiteMap(x, y, SiteMap::SiteBlocked);
                if (setting.EnablePG) {
                    b2 = b2 || siteMap.getSiteMap(x, y, SiteMap::SiteM2Blocked);
                }
                if (setting.EnableIOPin) {
                    b2 = b2 || siteMap.getSiteMap(x, y, SiteMap::SiteM2BlockedIOPin);
                }
                if (setting.EnableFence) {
                    r2 = getRegion(siteMap.getRegion(x, y));
                } else {
                    r2 = regions[0];
                }
            }
            if ((b1 || !r1) && !b2) {
                segment.x = coreLX + x * siteW;
                segment.w = siteW;
                segment.region = r2;
            } else if (!b1 && r1 && (b2 || !r2)) {
                row->segments.push_back(segment);
            } else if (!b1 && !b2 && (r1 == r2)) {
                segment.w += siteW;
            } else if (!b1 && !b2 && (r1 != r2)) {
                row->segments.push_back(segment);
                segment.x = coreLX + x * siteW;
                segment.w = siteW;
                segment.region = r2;
            }
            b1 = b2;
            r1 = r2;
        }
    }
}

void Database::setup() {
    SetupLayers();
    SetupCellLibrary();
    SetupFloorplan();
    SetupRegions();
    if (!setting.liteMode) {
        SetupSiteMap();
        SetupRows();
        SetupRowSegments();
    }
    logger.info("Finish setting up rawdb");
}

Layer& Database::addLayer(const string& name, const char type) {
    layers.emplace_back(name, type);
    Layer& newlayer = layers.back();
    if (layers.size() == 1) {
        if (type == 'r') {
            newlayer.rIndex = 0;
        }
    } else {
        Layer& oldlayer = layers[layers.size() - 2];
        oldlayer._above = &newlayer;
        newlayer._below = &oldlayer;
        if (type == 'r') {
            newlayer.rIndex = oldlayer.cIndex + 1;
        } else {
            newlayer.cIndex = oldlayer.rIndex;
        }
    }
    return newlayer;
}

Site& Database::addSite(const string& name, const string& siteClassName, const int w, const int h) {
    for (unsigned i = 0; i < sites.size(); i++) {
        if (name == sites[i].name()) {
            logger.warning("site re-defined: %s", name.c_str());
            return sites[i];
        }
    }
    sites.emplace_back(name, siteClassName, w, h);
    return sites.back();
}

ViaType* Database::addViaType(const string& name, bool isDef) {
    ViaType* viatype = getViaType(name);
    if (viatype) {
        logger.warning("via type re-defined: %s", name.c_str());
        return viatype;
    }
    viatype = new ViaType(name, isDef);
    name_viatypes[name] = viatype;
    viatypes.push_back(viatype);
    return viatype;
}

CellType* Database::addCellType(const string& name, unsigned libcell) {
    CellType* celltype = getCellType(name);
    if (celltype) {
        logger.warning("cell type re-defined: %s", name.c_str());
        return celltype;
    }
    celltype = new CellType(name, libcell);
    name_celltypes.emplace(name, celltype);
    celltypes.push_back(celltype);
    return celltype;
}

Cell* Database::addCell(const string& name, CellType* type) {
    Cell* cell = getCell(name);
    if (cell) {
        logger.warning("cell re-defined: %s", name.c_str());
        if (!cell->ctype()) {
            cell->ctype(type);
        }
        return cell;
    }
    cell = new Cell(name, type);
    name_cells.emplace(name, cell);
    cells.push_back(cell);
    return cell;
}

IOPin* Database::addIOPin(const string& name, const string& netName, const char direction) {
    IOPin* iopin = getIOPin(name);
    if (iopin) {
        logger.warning("IO pin re-defined: %s", name.c_str());
        return iopin;
    }
    iopin = new IOPin(name, netName, direction);
    name_iopins[name] = iopin;
    iopins.push_back(iopin);
    return iopin;
}

Net* Database::addNet(const string& name, const NDR* ndr) {
    Net* net = getNet(name);
    if (net) {
        logger.warning("Net re-defined: %s", name.c_str());
        return net;
    }
    net = new Net(name, ndr);
    name_nets[name] = net;
    nets.push_back(net);
    return net;
}

Row* Database::addRow(const string& name,
                      const string& macro,
                      const int x,
                      const int y,
                      const unsigned xNum,
                      const unsigned yNum,
                      const bool flip,
                      const unsigned xStep,
                      const unsigned yStep) {
    Row* newrow = new Row(name, macro, x, y, xNum, yNum, flip, xStep, yStep);
    rows.push_back(newrow);
    return newrow;
}

Track* Database::addTrack(char direction, double start, double num, double step) {
    Track* newtrack = new Track(direction, start, num, step);
    tracks.push_back(newtrack);
    return newtrack;
}

Region* Database::addRegion(const string& name, const char type) {
    Region* region = getRegion(name);
    if (region) {
        logger.warning("Region re-defined: %s", name.c_str());
        return region;
    }
    region = new Region(name, type);
    regions.push_back(region);
    return region;
}

NDR* Database::addNDR(const string& name, const bool hardSpacing) {
    NDR* ndr = getNDR(name);
    if (ndr) {
        logger.warning("NDR re-defined: %s", name.c_str());
        return ndr;
    }
    ndr = new NDR(name, hardSpacing);
    ndrs.emplace(name, ndr);
    return ndr;
}

SNet* Database::addSNet(const string& name) {
    SNet* newsnet = new SNet(name);
    snets.push_back(newsnet);
    return newsnet;
}

long long Database::getCellArea(Region* region) const {
    long long cellArea = 0;
    for (const Cell* cell : cells) {
        if (region && cell->region != region) {
            continue;
        }
        int w = cell->width() / siteW;
        int h = cell->height() / siteH;
        cellArea += w * h;
    }
    return cellArea;
}

long long Database::getFreeArea(Region* region) const {
    unsigned nRegions = getNumRegions();
    long long freeArea = 0;
    for (unsigned i = 0; i != nRegions; ++i) {
        if (region && region != regions[i]) {
            continue;
        }
        freeArea += siteMap.nRegionSites[i];
    }
    return freeArea;
}

long long Database::getHPWL() {
    long long hpwl = 0;
    int nNets = getNumNets();
    for (int i = 0; i < nNets; i++) {
        int nPins = nets[i]->pins.size();
        if (nPins < 2) {
            continue;
        }
        int lx = INT_MAX;
        int ly = INT_MAX;
        int hx = INT_MIN;
        int hy = INT_MIN;
        for (int j = 0; j < nPins; j++) {
            Pin* pin = nets[i]->pins[j];
            int x, y;
            pin->getPinCenter(x, y);
            lx = min(lx, x);
            ly = min(ly, y);
            hx = max(hx, x);
            hy = max(hy, y);
        }
        hpwl += (hx - lx) + (hy - ly);
    }
    return hpwl;
}

/* get layer by name */
Layer* Database::getLayer(const string& name) {
    for (Layer& layer : layers) {
        if (layer.name() == name) {
            return &layer;
        }
    }
    return nullptr;
}

/* get routing layer by index : 0=M1 */
Layer* Database::getRLayer(const int index) {
    for (Layer& layer : layers) {
        if (layer.rIndex == index) {
            return &layer;
        }
    }
    return nullptr;
}

/* get cut layer by index : 0=M1/2 */
const Layer* Database::getCLayer(const unsigned index) const {
    for (const Layer& layer : layers) {
        if (layer.cIndex == static_cast<int>(index)) {
            return &layer;
        }
    }
    return nullptr;
}

/* get cell type by name */
CellType* Database::getCellType(const string& name) {
    robin_hood::unordered_map<string, CellType*>::iterator mi = name_celltypes.find(name);
    if (mi == name_celltypes.end()) {
        return nullptr;
    }
    return mi->second;
}

Cell* Database::getCell(const string& name) {
    robin_hood::unordered_map<string, Cell*>::iterator mi = name_cells.find(name);
    if (mi == name_cells.end()) {
        return nullptr;
    }
    return mi->second;
}

Net* Database::getNet(const string& name) {
    robin_hood::unordered_map<string, Net*>::iterator mi = name_nets.find(name);
    if (mi == name_nets.end()) {
        return nullptr;
    }
    return mi->second;
}

Region* Database::getRegion(const string& name) {
    for (Region* region : regions) {
        if (region->name() == name) {
            return region;
        }
    }
    return nullptr;
}

Region* Database::getRegion(const unsigned char id) {
    if (id == Region::InvalidRegion) {
        return nullptr;
    }
    return regions[id];
}

NDR* Database::getNDR(const string& name) const {
    map<string, NDR*>::const_iterator mi = ndrs.find(name);
    if (mi == ndrs.end()) {
        return nullptr;
    }
    return mi->second;
}

IOPin* Database::getIOPin(const string& name) const {
    robin_hood::unordered_map<string, IOPin*>::const_iterator mi = name_iopins.find(name);
    if (mi == name_iopins.end()) {
        return nullptr;
    }
    return mi->second;
}

ViaType* Database::getViaType(const string& name) const {
    robin_hood::unordered_map<string, ViaType*>::const_iterator mi = name_viatypes.find(name);
    if (mi == name_viatypes.end()) {
        return nullptr;
    }
    return mi->second;
}

int Database::getContainedSites(
    const int lx, const int ly, const int hx, const int hy, int& slx, int& sly, int& shx, int& shy) const {
    slx = binContainedL(lx, coreLX, coreHX, siteW);
    sly = binContainedL(ly, coreLY, coreHY, siteH);
    shx = binContainedR(hx, coreLX, coreHX, siteW);
    shy = binContainedR(hy, coreLY, coreHY, siteH);
    if (slx > shx || sly > shy) {
        return 0;
    }
    return (shx - slx + 1) * (shy - sly + 1);
}

int Database::getOverlappedSites(
    const int lx, const int ly, const int hx, const int hy, int& slx, int& sly, int& shx, int& shy) const {
    slx = binOverlappedL(lx, coreLX, coreHX, siteW);
    sly = binOverlappedL(ly, coreLY, coreHY, siteH);
    shx = binOverlappedR(hx, coreLX, coreHX, siteW);
    shy = binOverlappedR(hy, coreLY, coreHY, siteH);
    if (slx > shx || sly > shy) {
        return 0;
    }
    return (shx - slx + 1) * (shy - sly + 1);
}

unsigned Database::getNumRLayers() const {
    unsigned numRLayers = 0;
    for (const Layer& layer : layers) {
        if (layer.rIndex != -1) {
            numRLayers++;
        }
    }
    return numRLayers;
}

unsigned Database::getNumCLayers() const {
    unsigned numCLayers = 0;
    for (const Layer& layer : layers) {
        if (layer.cIndex != -1) {
            numCLayers++;
        }
    }
    return numCLayers;
}

void Database::errorCheck(bool autoFix) {
    vector<Row*>::iterator ri = rows.begin();
    vector<Row*>::iterator re = rows.end();
    bool e_row_exceed_die = false;
    bool w_non_uniform_site_width = false;
    bool w_non_horizontal_row = false;
    int sitewidth = -1;
    for (; ri != re; ++ri) {
        Row* row = *ri;
        if (sitewidth < 0) {
            sitewidth = row->xStep();
        } else if ((unsigned)sitewidth != row->xStep()) {
            w_non_uniform_site_width = true;
        }
        if (row->yNum() != 1) {
            w_non_horizontal_row = true;
        }
        if (row->x() < dieLX) {
            if (autoFix) {
                int exceedSites = ceil((dieLX - row->x()) / (double)row->xStep());
                row->shrinkXNum(exceedSites);
                row->shiftX(exceedSites * row->xStep());
            }
            e_row_exceed_die = true;
        }
        if (row->x() + (int)row->width() > dieHX) {
            if (autoFix) {
                row->xNum((dieHX - row->x()) / row->xStep());
            }
            e_row_exceed_die = true;
        }
    }
    if (e_row_exceed_die) {
        dbIssues.push_back(E_ROW_EXCEED_DIE);
    }
    if (w_non_uniform_site_width) {
        dbIssues.push_back(W_NON_UNIFORM_SITE_WIDTH);
    }
    if (w_non_horizontal_row) {
        dbIssues.push_back(W_NON_HORIZONTAL_ROW);
    }

    bool e_no_net_driving_pin = false;
    bool e_multiple_net_driving_pin = false;
    for (int i = 0; i < (int)nets.size(); i++) {
        for (int j = 0; j < (int)nets[i]->pins.size(); j++) {
            if (nets[i]->pins[j]->type->direction() != 'o') {
                e_no_net_driving_pin = true;
                break;
            }
        }
        for (int j = 0; j < (int)nets[i]->pins.size(); j++) {
            if (nets[i]->pins[j]->type->direction() != 'i') {
                e_multiple_net_driving_pin = true;
                break;
            }
        }
    }
    if (e_no_net_driving_pin) {
        dbIssues.push_back(E_NO_NET_DRIVING_PIN);
    }
    if (e_multiple_net_driving_pin) {
        dbIssues.push_back(E_MULTIPLE_NET_DRIVING_PIN);
    }

    for (int i = 0; i < (int)dbIssues.size(); i++) {
        switch (dbIssues[i]) {
            case E_ROW_EXCEED_DIE:
                logger.warning("row is placed out of die area");
                break;
            case W_NON_UNIFORM_SITE_WIDTH:
                logger.warning("non uniform site width detected");
                break;
            case W_NON_HORIZONTAL_ROW:
                logger.warning("non horizontal row detected");
                break;
            case E_NO_NET_DRIVING_PIN:
                logger.warning("missing net driving pin");
                break;
            case E_MULTIPLE_NET_DRIVING_PIN:
                logger.warning("multiple net driving pin");
                break;
            default:
                break;
        }
    }
}

void Database::checkPlaceError() {
    logger.info("starting checking...");
    int nError = 0;
    vector<Cell*> cells = this->cells;
    sort(cells.begin(), cells.end(), [](const Cell* a, const Cell* b) { return a->lx() < b->lx(); });
    int nCells = cells.size();
    for (int i = 0; i < nCells; i++) {
        Cell* cell_i = cells[i];
        // int lx = cell_i->x;
        int hx = cell_i->hx();
        int ly = cell_i->ly();
        int hy = cell_i->hy();
        for (int j = i + 1; j < nCells; j++) {
            Cell* cell_j = cells[j];
            if (cell_j->lx() >= hx) {
                break;
            }
            if (cell_j->ly() >= hy || cell_j->hy() <= ly) {
                continue;
            }
            nError++;
        }
    }

    logger.info("#overlap=%d", nError);
}

void Database::checkDRCError() {
    logger.info("starting checking...");
    vector<int> nOverlapErrors(3);
    vector<int> nSpacingErrors(3);

    class Metal {
    public:
        Rectangle rect;
        const Cell* cell;
        const IOPin* iopin;
        const SNet* snet;
    };

    vector<int> minSpace(3);
    vector<vector<Metal>> metals(3);

    for (unsigned i = 0; i != 3; ++i) {
        minSpace[i] = getRLayer(i)->spacing;
    }

    for (const Cell* cell : cells) {
        unsigned nPins = cell->numPins();
        for (unsigned j = 0; j != nPins; ++j) {
            Pin* pin = cell->pin(j);
            for (const Geometry& geo : pin->type->shapes) {
                Metal metal;
                metal.rect.lx = cell->lx() + geo.lx;
                metal.rect.hx = cell->lx() + geo.hx;
                metal.rect.ly = cell->ly() + geo.ly;
                metal.rect.hy = cell->ly() + geo.hy;
                metal.cell = cell;
                metal.iopin = nullptr;
                metal.snet = nullptr;
                const int rIndex = geo.layer.rIndex;
                if (rIndex >= 0 && rIndex <= 2) {
                    metals[rIndex].push_back(metal);
                }
            }
        }
    }

    for (const IOPin* iopin : iopins) {
        for (const Geometry& geo : iopin->type->shapes) {
            Metal metal;
            metal.rect.lx = iopin->x + geo.lx;
            metal.rect.hx = iopin->x + geo.hx;
            metal.rect.ly = iopin->y + geo.ly;
            metal.rect.hy = iopin->y + geo.hy;
            metal.cell = nullptr;
            metal.iopin = iopin;
            metal.snet = nullptr;
            const int rIndex = geo.layer.rIndex;
            if (rIndex >= 0 && rIndex <= 2) {
                metals[rIndex].push_back(metal);
            }
        }
    }

    for (const SNet* snet : snets) {
        for (const Geometry& geo : snet->shapes) {
            Metal metal;
            metal.rect.lx = geo.lx;
            metal.rect.hx = geo.hx;
            metal.rect.ly = geo.ly;
            metal.rect.hy = geo.hy;
            metal.cell = nullptr;
            metal.iopin = nullptr;
            metal.snet = snet;
            const int rIndex = geo.layer.rIndex;
            if (rIndex >= 0 && rIndex <= 2) {
                metals[rIndex].push_back(metal);
            }
        }
    }

    for (unsigned i = 0; i != 3; ++i) {
        logger.info("m%d = %u", i + 1, metals[i].size());
        sort(metals[i].begin(), metals[i].end(), [](const Metal& a, const Metal& b) {
            return (a.rect.lx == b.rect.lx) ? (a.rect.ly < b.rect.ly) : (a.rect.lx < b.rect.lx);
        });
    }

    for (unsigned L = 0; L != 3; ++L) {
        unsigned nMetals = metals[L].size();
        int minS = minSpace[L];
        for (unsigned i = 0; i != nMetals; ++i) {
            Metal& mi = metals[L][i];
            // int lx = mi.rect.lx;
            int hx = mi.rect.hx;
            int ly = mi.rect.ly;
            int hy = mi.rect.hy;
            for (unsigned j = i + 1; j != nMetals; ++j) {
                const Metal& mj = metals[L][j];
                if (mj.rect.lx - minS >= hx) {
                    break;
                }
                if (mj.rect.ly - minS >= hy || mj.rect.hy + minS <= ly) {
                    continue;
                }
                if ((mi.cell != NULL && mi.cell == mj.cell) || (mi.iopin != NULL && mi.iopin == mj.iopin) ||
                    (mi.snet != NULL && mi.snet == mj.snet)) {
                    continue;
                }
                ++(nSpacingErrors[L]);
                if (mj.rect.lx >= hx) {
                    continue;
                }
                if (mj.rect.ly >= hy || mj.rect.hy <= ly) {
                    continue;
                }
                ++(nOverlapErrors[L]);
            }
        }
    }

    /*
    vector<Cell*> cells = this->cells;
    sort(cells.begin(), cells.end(), Cell::CompareXInc);
    int nCells = cells.size();
    for(int i=0; i<nCells; i++){
        Cell *cell_i = cells[i];
        //int lx = cell_i->x;
        int hx = cell_i->x + cell_i->width();
        int ly = cell_i->y;
        int hy = cell_i->y + cell_i->height();
        for(int j=i+1; j<nCells; j++){
            Cell *cell_j = cells[j];
            if(cell_j->x >= hx){
                break;
            }
            if(cell_j->y >= hy || cell_j->y + cell_j->height() <= ly){
                continue;
            }
            nError++;
        }
    }
    */

    for (unsigned i = 0; i != 3; ++i) {
        logger.info("#M%u overlaps = %d", i + 1, nOverlapErrors[i]);
        logger.info("#M%u spacings = %d", i + 1, nSpacingErrors[i]);
    }
}
