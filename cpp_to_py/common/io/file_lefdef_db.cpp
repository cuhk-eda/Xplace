#include "common/db/Database.h"

// tcl have another definition of EXTERN
#undef EXTERN

#include "def58/include/defiUtil.hpp"
#include "def58/include/defrReader.hpp"
#include "lef58/include/lefrReader.hpp"

using namespace db;

bool isFlipX(int orient) {
    switch (orient) {
        case 0:
            return false;  // N
        case 2:
            return true;  // S
        case 4:
            return true;  // FN
        case 6:
            return false;  // FS
    }
    return false;
}
bool isFlipY(int orient) {
    switch (orient) {
        case 0:
            return false;  // N
        case 2:
            return true;  // S
        case 4:
            return false;  // FN
        case 6:
            return true;  // FS
    }
    return false;
}

string getOrient(bool flipX, bool flipY) {
    if (flipX) {
        if (flipY) {
            return "S";
        } else {
            return "FN";
        }
    } else {
        if (flipY) {
            return "FS";
        } else {
            return "N";
        }
    }
}

#define DIR_UP 1
#define DIR_DOWN 2
#define DIR_LEFT 4
#define DIR_RIGHT 8

unsigned char pointDir(const int lx, const int ly, const int hx, const int hy) {
    if ((lx == hx) == (ly == hy)) {
        return 0;
    }
    if (lx < hx) {
        return DIR_RIGHT;
    }
    if (lx > hx) {
        return DIR_LEFT;
    }
    if (ly < hy) {
        return DIR_UP;
    }
    return DIR_DOWN;
}

class Point {
public:
    int x, y;
    unsigned char outdir = 0;

    Point(const int x = 0, const int y = 0, const unsigned char dir = 0) : x(x), y(y), outdir(dir) {}

    inline bool operator<(const Point& T) const {
        if (y < T.y) {
            return true;
        }
        if (y > T.y) {
            return false;
        }
        if (x < T.x) {
            return true;
        }
        if (x > T.x) {
            return false;
        }
        if ((outdir | DIR_DOWN) && T.outdir | DIR_UP) {
            return true;
        }
        if ((outdir | DIR_UP) && T.outdir | DIR_DOWN) {
            return false;
        }
        if ((outdir | DIR_LEFT) && T.outdir | DIR_RIGHT) {
            return true;
        }
        return false;
    }

    inline bool operator==(const Point& T) const { return x == T.x && y == T.y; }
    inline bool operator!=(const Point& T) const { return !(*this == T); }
};

int readLefUnits(lefrCallbackType_e c, lefiUnits* unit, lefiUserData ud);
int readLefProp(lefrCallbackType_e c, lefiProp* prop, lefiUserData ud);
int readLefLayer(lefrCallbackType_e c, lefiLayer* leflayer, lefiUserData ud);
int readLefVia(lefrCallbackType_e c, lefiVia* lvia, lefiUserData ud);
int readLefMacroBegin(lefrCallbackType_e c, const char* macroName, lefiUserData ud);
int readLefObs(lefrCallbackType_e c, lefiObstruction* obs, lefiUserData ud);
int readLefPin(lefrCallbackType_e c, lefiPin* pin, lefiUserData ud);
int readLefSite(lefrCallbackType_e c, lefiSite* site, lefiUserData ud);
int readLefMacro(lefrCallbackType_e c, lefiMacro* macro, lefiUserData ud);
int readLefMacroEnd(lefrCallbackType_e c, const char* macroName, lefiUserData ud);

int readDefUnits(defrCallbackType_e c, double d, defiUserData ud);
int readDefVersion(defrCallbackType_e c, double d, defiUserData ud);
int readDefDesign(defrCallbackType_e c, const char* name, defiUserData ud);
int readDefDieArea(defrCallbackType_e c, defiBox* dbox, defiUserData ud);
int readDefRow(defrCallbackType_e c, defiRow* drow, defiUserData ud);
int readDefTrack(defrCallbackType_e c, defiTrack* dtrack, defiUserData ud);
int readDefGcellGrid(defrCallbackType_e c, defiGcellGrid* dgrid, defiUserData ud);
int readDefVia(defrCallbackType_e c, defiVia* dvia, defiUserData ud);
int readDefNdr(defrCallbackType_e c, defiNonDefault* nd, defiUserData ud);
int readDefComponentStart(defrCallbackType_e c, int num, defiUserData ud);
int readDefComponent(defrCallbackType_e c, defiComponent* co, defiUserData ud);
int readDefPin(defrCallbackType_e c, defiPin* dpin, defiUserData ud);
int readDefBlockage(defrCallbackType_e c, defiBlockage* dblk, defiUserData ud);
int readDefSNetStart(defrCallbackType_e c, int num, defiUserData ud);
int readDefSNet(defrCallbackType_e c, defiNet* dnet, defiUserData ud);
int readDefNetStart(defrCallbackType_e c, int num, defiUserData ud);
int readDefNet(defrCallbackType_e c, defiNet* dnet, defiUserData ud);
int readDefRegion(defrCallbackType_e c, defiRegion* dreg, defiUserData ud);
int readDefGroupName(defrCallbackType_e c, const char* cl, defiUserData ud);
int readDefGroupMember(defrCallbackType_e c, const char* cl, defiUserData ud);
int readDefGroup(defrCallbackType_e c, defiGroup* dgp, defiUserData ud);
inline void fastCopy(char* t, const char* s, size_t n);

bool Database::readLEF(const std::string& file) {
    FILE* fp;
    if (!(fp = fopen(file.c_str(), "r"))) {
        logger.error("Unable to open LEF file: %s", file.c_str());
        return false;
    }

#ifndef NDEBUG
    logger.info("reading %s", file.c_str());
#endif

    lefrSetUnitsCbk(readLefUnits);
    lefrSetPropCbk(readLefProp);
    lefrSetLayerCbk(readLefLayer);
    lefrSetViaCbk(readLefVia);
    lefrSetMacroCbk(readLefMacro);
    lefrSetMacroBeginCbk(readLefMacroBegin);
    lefrSetObstructionCbk(readLefObs);
    lefrSetPinCbk(readLefPin);
    lefrSetSiteCbk(readLefSite);
    lefrSetMacroEndCbk(readLefMacroEnd);
    lefrInit();
    lefrReset();
    int res = lefrRead(fp, file.c_str(), (void*)this);
    if (res) {
        logger.error("Error in reading LEF");
        return false;
    }
    lefrReleaseNResetMemory();
    //  lefrUnsetCallbacks();
    lefrUnsetLayerCbk();
    lefrUnsetNonDefaultCbk();
    lefrUnsetViaCbk();
    fclose(fp);
    return true;
}

bool Database::readDEF(const std::string& file) {
    FILE* fp;
    if (!(fp = fopen(file.c_str(), "r"))) {
        logger.error("Unable to open DEF file: %s", file.c_str());
        return false;
    }

#ifndef NDEBUG
    logger.info("reading %s", file.c_str());
#endif

    defrSetDesignCbk(readDefDesign);
    defrSetUnitsCbk(readDefUnits);
    defrSetVersionCbk(readDefVersion);
    defrSetDieAreaCbk(readDefDieArea);
    defrSetRowCbk(readDefRow);
    defrSetTrackCbk(readDefTrack);
    defrSetGcellGridCbk(readDefGcellGrid);
    defrSetViaCbk(readDefVia);
    defrSetNonDefaultCbk(readDefNdr);
    defrSetComponentStartCbk(readDefComponentStart);
    defrSetComponentCbk(readDefComponent);
    defrSetPinCbk(readDefPin);
    defrSetBlockageCbk(readDefBlockage);

    defrSetSNetStartCbk(readDefSNetStart);
    defrSetSNetCbk(readDefSNet);
    //  defrSetSNetWireCbk(readDefSnetwire);
    defrSetNetStartCbk(readDefNetStart);
    defrSetNetCbk(readDefNet);
    // augment nets with path data
    defrSetAddPathToNet();

    defrSetRegionCbk(readDefRegion);
    //  defrSetGroupNameCbk(readDefGroupName);
    defrSetGroupMemberCbk(readDefGroupMember);
    defrSetGroupCbk(readDefGroup);

    defrInit();
    defrReset();
    int res = defrRead(fp, file.c_str(), (void*)this, 1);
    if (res) {
        logger.error("Error in reading DEF");
        return false;
    }
    defrReleaseNResetMemory();
    defrUnsetCallbacks();
    fclose(fp);
    return true;
}

bool Database::readDEFPG(const std::string& file) {
    //  shape of pre-routed PG mesh is not supported by the reader
    string buffer;
    ifstream ifs(file.c_str());
    if (!ifs.good()) {
        logger.error("Unable to open DEF PG file: %s", file.c_str());
        return false;
    }

#ifndef NDEBUG
    logger.info("reading %s", file.c_str());
#endif

    Database* db = this;
    int n_snets = 0;

    //  ignore lines until the keyword
    do {
        ifs >> buffer;
        if (buffer == "SPECIALNETS") {
            ifs >> n_snets >> buffer;
            break;
        }
    } while (!ifs.eof());

    for (int i = 0; i < n_snets; i++) {
        //  SNet* snet = NULL;
        do {
            ifs >> buffer;
            if (buffer == "-") {
                string snetname;
                ifs >> snetname;
                // snet = db->addSNet(snetname);
            }
            if (buffer == "ROUTED" || buffer == "NEW") {
                string layername;
                int wirewidth;
                string vianame;
                string shape;
                char direction;
                int fx, fy;
                //  int tx, ty;
                string tx_str, ty_str;
                ifs >> layername >> wirewidth;
                if (wirewidth == 0) {
                    // via
                    ifs >> buffer >> buffer >> shape >> buffer >> fx >> fy >> buffer >> vianame;
                    ViaType* viatype = db->getViaType(vianame);
                    if (!viatype) {
                        logger.error("Via type is not defined: %s", vianame.c_str());
                        return false;
                    }
                    //  snet->addVia(viatype, fx, fy);
                } else {
                    // wire
                    ifs >> buffer >> buffer >> shape >> buffer >> fx >> fy >> buffer >> buffer >> tx_str >> ty_str >>
                        buffer;
                    if (tx_str == "*") {
                        //  tx = fx;
                    } else {
                        //  tx = atoi(tx_str.c_str());
                    }
                    if (ty_str == "*") {
                        //  ty = fy;
                    } else {
                        //  ty = atoi(ty_str.c_str());
                    }
                    //  if (tx == fx) {
                    //      direction = 'v';
                    //  } else {
                    //      direction = 'h';
                    //  }
                    Layer* layer = db->getLayer(layername);
                    if (!layer) {
                        logger.error("Layer is not defined: %s", layername.c_str());
                        return false;
                    }
                    //  int lx, ly, hx, hy;
                    if (direction == 'h') {
                        //  lx = min(fx, tx);
                        //  ly = fy - wirewidth / 2;
                        //  hx = max(fx, tx);
                        //  hy = ly + wirewidth;
                    } else {
                        //  lx = fx - wirewidth / 2;
                        //  ly = min(fy, ty);
                        //  hx = lx + wirewidth;
                        //  hy = max(fy, ty);
                    }
                    //  snet->addShape(*layer, lx, ly, hx, hy);
                    if ((layer->rIndex == 0 || layer->rIndex == 1) && direction == 'h') {
                        //  db->powerNet.addRail(snet, lx, hx, fy);
                    }
                }
            }
            if (buffer == "USE") {
                string type;
                ifs >> type;
                if (type == "POWER") {
                    //  snet->type = 'p';
                } else if (type == "GROUND") {
                    //  snet->type = 'g';
                } else {
                    logger.error("unknown use: %s", type.c_str());
                }
            }
            if (buffer == ";") {
                break;
            }
        } while (!ifs.eof());
    }
    ifs.close();
    return true;
}

bool Database::writeComponents(ofstream& ofs) {
    int nCells = cells.size();
    ofs << "COMPONENTS " << nCells << " ;" << endl;
    // ofs << "COMPONENTS " << nCells << " ;" << endl;
    for (int i = 0; i < nCells; i++) {
        Cell* cell = cells[i];
#ifdef WRITE_BUFFER
        ostringstream oss;
#else
        ofstream& oss = ofs;
#endif
        oss << "   - " << cell->name() << " " << cell->ctype()->name << endl;
        // ofs << "   - " << cell->name() << " " << cell->ctype()->name << endl;
        if (cell->fixed()) {
            oss << "      + FIXED ( " << cell->lx() << " " << cell->ly() << " ) "
                << getOrient(cell->flipX(), cell->flipY()) << " ;" << endl;
            // ofs << "      + FIXED ( " << cell->lx() << " " << cell->ly() << " ) "
            //    << getOrient(cell->flipX(), cell->flipY())
            //    << " ;" << endl;
        } else if (cell->placed()) {
            oss << "      + PLACED ( " << cell->lx() << " " << cell->ly() << " ) "
                << getOrient(cell->flipX(), cell->flipY()) << " ;" << endl;
            // ofs << "      + PLACED ( " << cell->lx() << " " << cell->ly() << " ) "
            //    << getOrient(cell->flipX(), cell->flipY())
            //    << " ;" << endl;
        } else {
            oss << "      + UNPLACED ;" << endl;
            // ofs << "      + UNPLACED ;" << endl;
        }
#ifdef WRITE_BUFFER
        string lines = oss.str();
        writeBuffer(ofs, lines);
#endif
    }
#ifdef WRITE_BUFFER
    writeBufferFlush(ofs);
#endif
    ofs << "END COMPONENTS\n\n";
    return true;
}

bool Database::writeICCAD2017(const string& inputDef, const string& outputDef) {
    ifstream ifs(inputDef.c_str());
    if (!ifs.good()) {
        logger.error("Unable to create/open DEF: %s", inputDef.c_str());
        return false;
    }

#ifndef NDEBUG
    logger.info("reading %s", inputDef.c_str());
#endif

    ofstream ofs(outputDef.c_str());
    if (!ofs.good()) {
        logger.error("Unable to create/open DEF: %s", outputDef.c_str());
        return false;
    }
    logger.info("writing %s", outputDef.c_str());

    string line;
    while (getline(ifs, line)) {
        istringstream iss(line);
        string s;
        if (!(iss >> s)) {
            ofs << line << endl;
            continue;
        } else if (s != "COMPONENTS") {
            ofs << line << endl;
            continue;
        }
        writeComponents(ofs);
        while (getline(ifs, line)) {
            istringstream iss(line);
            if (iss >> s && s == "END") {
                break;
            }
        }
        // process pair (a,b)
    }
    return true;
}

bool Database::writeICCAD2017(const string& outputDef) {
    ofstream ofs(outputDef.c_str(), ios::app);
    if (!ofs.good()) {
        logger.error("Unable to create/open DEF: %s", outputDef.c_str());
        return false;
    }
    logger.info("writing %s", outputDef.c_str());

    writeComponents(ofs);  // just replace the information of components, while others keep remain.

    ofs << "END DESIGN\n\n" << endl;

    ofs.close();
    return true;
}

bool Database::writeDEF(const string& file) {
    ofstream ofs(file.c_str());
    if (!ofs.good()) {
        logger.error("Unable to create/open DEF: %s", file.c_str());
        return false;
    }
    logger.info("writing %s", file.c_str());

    ofs << "VERSION 5.8 ;" << endl;
    ofs << "DIVIDERCHAR \"/\" ;" << endl;
    ofs << "BUSBITCHARS \"[]\" ;" << endl;
    ofs << "DESIGN " << designName << " ;" << endl;
    ofs << "UNITS DISTANCE MICRONS " << (int)DBU_Micron << " ; \n\n";
    ofs << "DIEAREA ( " << dieLX << ' ' << dieLY << " ) ( " << dieHX << ' ' << dieHY << " ) ;\n\n";

    for (Row* row : rows) {
        ofs << "ROW " << row->name() << ' ' << row->macro() << ' ' << row->x() << ' ' << row->y() << ' ';
        ofs << getOrient(false, row->flip());
        ofs << " DO " << row->xNum() << " BY " << row->yNum() << " STEP " << row->xStep() << ' ' << row->yStep()
            << " ;\n";
    }
    ofs << endl;

    for (Track* track : tracks) {
        ofs << "TRACKS " << track->macro() << ' ' << track->start << " DO " << track->num << " STEP " << track->step
            << " LAYER ";
        for (unsigned i = 0; i != track->numLayers(); ++i) {
            ofs << track->layer(i) << ' ';
        }
        ofs << ";\n";
    }
    ofs << endl;

    ostringstream ossv;
    unsigned nVias = 0;
    for (ViaType* via : viatypes) {
        if (!via->isDef()) {
            continue;
        }
        ++nVias;
        ossv << "\t- " << via->name;
        for (Geometry geo : via->rects) {
            ossv << "\n\t\t+ RECT " << geo.layer.name() << " ( " << geo.lx << ' ' << geo.ly << " ) ( " << geo.hx << ' '
                 << geo.hy << " )";
        }
        ossv << " ;\n";
    }
    ofs << "VIAS " << nVias << " ;\n";
    ofs << ossv.str();
    ofs << "END VIAS\n\n";

    ofs << "NONDEFAULTRULES " << ndrs.size() << " ;\n";
    for (const auto& [name, ndr] : ndrs) {
        //  const NDR* ndr = p.second;
        ofs << "\t- " << name;
        if (ndr->hardSpacing()) {
            ofs << "\n\t\t+ HARDSPACING";
        }
        for (const WireRule& rule : ndr->rules) {
            ofs << "\n\t\t+ LAYER " << rule.layer()->name() << " WIDTH " << rule.width;
            if (rule.space) {
                ofs << " SPACING " << rule.space;
            }
        }
        for (ViaType* via : ndr->vias) {
            ofs << "\n\t\t+ VIA " << via->name;
        }
        ofs << " ;\n";
    }
    ofs << "END NONDEFAULTRULES\n\n";

    ostringstream ossr;
    unsigned nRegions = 0;
    for (Region* region : regions) {
        if (region->name() == "default") {
            continue;
        }
        ++nRegions;
        ossr << "\t- " << region->name();
        for (const Rectangle& rect : region->rects) {
            ossr << "\t( " << rect.lx << ' ' << rect.ly << " ) ( " << rect.hx << ' ' << rect.hy << " )";
        }
        switch (region->type()) {
            case 'f':
                ossr << "\t+ TYPE FENCE ;\n";
                break;
            case 'g':
                ossr << "\t+ TYPE GUIDE ;\n";
                break;
            default:
                logger.error("region type not recognized: %c", region->type());
                ossr << " ;\n";
                break;
        }
    }
    ofs << "REGIONS " << nRegions << " ;\n";
    ofs << ossr.str();
    ofs << "END REGIONS\n\n";

    writeComponents(ofs);

    ofs << "PINS " << iopins.size() << " ;\n";
    for (IOPin* iopin : iopins) {
        ofs << "\t- " << iopin->name << " + NET " << iopin->netName();
        switch (iopin->type->direction()) {
            case 'f':
                ofs << "\n\n\t+ DIRECTION FEEDTHRU";
                break;
            case 'i':
                ofs << "\n\t\t+ DIRECTION OUTPUT";
                break;
            case 'o':
                ofs << "\n\t\t+ DIRECTION INPUT";
                break;
            case 'x':
                ofs << "\n\t\t+ DIRECTION INOUT";
                break;
            default:
                logger.error("iopin direction not recognized: %c", iopin->type->direction());
                break;
        }
        if (iopin->x != INT_MIN || iopin->y != INT_MIN) {
            ofs << "\n\t\t+ PLACED ( " << iopin->x << ' ' << iopin->y << " ) ";
        }
    }

    ofs << "END DESIGN" << endl;

    ofs.close();
    return true;
}

/***********************************/
/* .DEF File buffer Writing Scheme */
/***********************************/

/// a buffered writing scheme
bool Database::writeBuffer(ofstream& ofs, const string& line) {
    const char* b = line.c_str();
    size_t n = line.size();
    while (_bufferSize + n > _bufferCapacity)  // output exceeds the capacity
    {
        size_t nw = _bufferCapacity - _bufferSize;
        if (nw) {
            n -= nw;
            fastCopy(_buffer + _bufferSize, b, nw);
            b += nw;
        }

        ofs.write(_buffer, _bufferCapacity);  // From _buffer write _bufferCapacity charcters to ostream
        _bufferSize = 0;
    }

    if (n)  // wait until buffer is full then output
    {
        fastCopy(_buffer + _bufferSize, b, n);
        _bufferSize += n;
    }

    return true;
}

void Database::writeBufferFlush(ofstream& ofs) {
    if (_bufferSize)  // remember to write the rest content in the buffer
    {
        ofs.write(_buffer, _bufferSize);
        _bufferSize = 0;
    }
}

/// copy char* as unsigned long* and deal with boundary cases
inline void fastCopy(char* t, const char* s, size_t n) {
    if (n >= sizeof(unsigned long)) {
        unsigned long* tl = reinterpret_cast<unsigned long*>(t);
        const unsigned long* sl = reinterpret_cast<const unsigned long*>(s);

        while (n >= sizeof(unsigned long))  // copy char* as unsigned long*
        {
            *tl++ = *sl++;
            n -= sizeof(unsigned long);
        }

        t = reinterpret_cast<char*>(tl);
        s = reinterpret_cast<const char*>(sl);
    }

    while (n-- > 0)  // boundary cases
    {
        *t++ = *s++;
    }
}

/****************/
/* LEF Callback */
/****************/

int readLefUnits(lefrCallbackType_e c, lefiUnits* unit, lefiUserData ud) {
    Database* db = (Database*)ud;
    if (unit->lefiUnits::hasDatabase() && strcmp(unit->lefiUnits::databaseName(), "MICRONS") == 0) {
        db->LefConvertFactor = unit->lefiUnits::databaseNumber();
    }
    return 0;
}

//-----Property-----
int readLefProp(lefrCallbackType_e c, lefiProp* prop, lefiUserData ud) {
    Database* db = (Database*)ud;
    if (prop->lefiProp::hasString() && !strcmp(prop->lefiProp::propName(), "LEF58_CELLEDGESPACINGTABLE")) {
        stringstream sstable(prop->lefiProp::string());
        string buffer;

        while (!sstable.eof()) {
            sstable >> buffer;
            if (buffer == "EDGETYPE") {
                string type1, type2, type3;
                double microndist;
                sstable >> type1 >> type2 >> type3;
                if (type3 == "EXCEPTABUTTED") {
                    logger.warning("ignore EXCEPTABUTTED between %s and %s", type1.c_str(), type2.c_str());
                    sstable >> microndist;
                } else {
                    microndist = stod(type3);
                }
                int dbdist = 0;
                if (setting.EdgeSpacing) {
                    dbdist = (int)round(microndist * db->LefConvertFactor);
                }
                int type1_idx = db->edgetypes.getEdgeType(type1);
                int type2_idx = db->edgetypes.getEdgeType(type2);

                if (type1_idx < 0) {
                    type1_idx = db->edgetypes.types.size();
                    db->edgetypes.types.push_back(type1);
                }
                if (type2_idx < 0) {
                    type2_idx = db->edgetypes.types.size();
                    db->edgetypes.types.push_back(type2);
                }

                int numEdgeTypes = db->edgetypes.types.size();
                if (numEdgeTypes > (int)db->edgetypes.distTable.size()) {
                    db->edgetypes.distTable.resize(numEdgeTypes);
                }
                for (int i = 0; i < numEdgeTypes; i++) {
                    db->edgetypes.distTable[i].resize(numEdgeTypes, 0);
                }
                db->edgetypes.distTable[type1_idx][type2_idx] = dbdist;
                db->edgetypes.distTable[type2_idx][type1_idx] = dbdist;
            }
            if (buffer == ";") {
                break;
            }
        }
        // default edge type dist
        for (unsigned i = 0; i < db->edgetypes.types.size(); i++) {
            db->edgetypes.distTable[0][i] = 0;
            db->edgetypes.distTable[i][0] = 0;
        }
    }
    return 0;
}

//-----Layer-----
int readLefLayer(lefrCallbackType_e c, lefiLayer* leflayer, lefiUserData ud) {
    Database* db = (Database*)ud;
    const int convertFactor = db->LefConvertFactor;

    const string name(leflayer->name());
    char type = 'x';
    if (!strcmp(leflayer->type(), "ROUTING")) {
        type = 'r';
    } else if (!strcmp(leflayer->type(), "CUT")) {
        if (db->layers.empty()) {
            logger.warning("remove cut layer %s below the first metal layer", name.c_str());
            return 0;
        }
        type = 'c';
    } else {
        return 0;
    }

    Layer& layer = db->addLayer(name, type);

    switch (type) {
        case 'r':
            // routing layer
            if (leflayer->hasDirection()) {
                if (strcmp(leflayer->direction(), "HORIZONTAL") == 0) {
                    layer.direction = 'h';
                } else if (strcmp(leflayer->direction(), "VERTICAL") == 0) {
                    layer.direction = 'v';
                }
            } else {
                layer.direction = 'x';
            }

            if (leflayer->hasXYPitch()) {
                if (layer.direction == 'v') {
                    layer.pitch = lround(leflayer->pitchX() * convertFactor);
                } else {
                    layer.pitch = lround(leflayer->pitchY() * convertFactor);
                }
            }
            if (leflayer->hasPitch()) {
                layer.pitch = lround(leflayer->pitch() * convertFactor);
            }

            if (leflayer->hasXYOffset()) {
                if (layer.direction == 'v') {
                    layer.offset = lround(leflayer->offsetX() * convertFactor);
                } else {
                    layer.offset = lround(leflayer->offsetY() * convertFactor);
                }
            }
            if (leflayer->hasOffset()) {
                layer.offset = lround(leflayer->offset() * convertFactor);
            }

            if (leflayer->hasWidth()) {
                layer.width = lround(leflayer->width() * convertFactor);
            }

            if (leflayer->lefiLayer::hasArea()) {
                layer.area = lround(leflayer->lefiLayer::area() * convertFactor);
            }

            if (leflayer->lefiLayer::hasMinwidth()) {
                layer.minWidth = lround(leflayer->lefiLayer::minwidth() * convertFactor);
            }

            if (leflayer->lefiLayer::hasMaxwidth()) {
                layer.maxWidth = lround(leflayer->lefiLayer::maxwidth() * convertFactor);
            }

            if (layer.width > 0 && layer.pitch > 0) {
                layer.spacing = layer.pitch - layer.width;
            } else if (layer.width < 0 && layer.pitch > 0) {
                layer.width = layer.pitch / 2;
                layer.spacing = layer.pitch - layer.width;
            } else if (layer.pitch < 0 && layer.width > 0) {
                layer.pitch = layer.width * 2;
                layer.spacing = layer.width;
            }

            if (leflayer->lefiLayer::hasSpacingNumber()) {
                for (int i = 0; i < leflayer->lefiLayer::numSpacing(); i++) {
                    // spaceType: 0 -> minSpacing, 1 -> maxEOLSpace, 2 -> maxEOLSpaceParallelEdge
                    int spaceType = 0;
                    int spacing, width, within, parSpace, parWithin;
                    spacing = lround(leflayer->lefiLayer::spacing(i) * convertFactor);
                    if (leflayer->lefiLayer::hasSpacingEndOfLine(i)) {
                        width = lround(leflayer->lefiLayer::spacingEolWidth(i) * convertFactor);
                        within = lround(leflayer->lefiLayer::spacingEolWithin(i) * convertFactor);
                        spaceType++;
                        if (leflayer->lefiLayer::hasSpacingParellelEdge(i)) {
                            parSpace = lround(leflayer->lefiLayer::spacingParSpace(i) * convertFactor);
                            parWithin = lround(leflayer->lefiLayer::spacingParWithin(i) * convertFactor);
                            spaceType++;
                        }
                    }
                    switch (spaceType) {
                        case 0:
                            layer.spacing = spacing;
                            break;
                        case 1:
                            layer.maxEOLSpace = {spacing, width, within};
                            break;
                        case 2:
                            layer.maxEOLSpaceParallelEdge = {spacing, width, within, parSpace, parWithin};
                            break;
                        default:
                            layer.spacing = spacing;
                            break;
                    }
                }
            }

            for (int i = 0; i < leflayer->lefiLayer::numSpacingTable(); i++) {
                // TODO: we only support ParallelRunLength currently
                lefiSpacingTable* spTable = leflayer->lefiLayer::spacingTable(i);
                if (spTable->lefiSpacingTable::isParallel()) {
                    lefiParallel* parallel = spTable->lefiSpacingTable::parallel();
                    layer.parLength.resize(parallel->lefiParallel::numLength());
                    for (int j = 0; j < parallel->lefiParallel::numLength(); j++) {
                        layer.parLength[j] = lround(parallel->lefiParallel::length(j) * convertFactor);
                    }
                    layer.parWidth.resize(parallel->lefiParallel::numWidth());
                    layer.parWidthSpace.resize(parallel->lefiParallel::numWidth());
                    for (int j = 0; j < parallel->lefiParallel::numWidth(); j++) {
                        layer.parWidth[j] = lround(parallel->lefiParallel::width(j) * convertFactor);
                        layer.parWidthSpace[j].resize(parallel->lefiParallel::numLength());
                        for (int k = 0; k < parallel->lefiParallel::numLength(); k++) {
                            layer.parWidthSpace[j][k] =
                                lround(parallel->lefiParallel::widthSpacing(j, k) * convertFactor);
                        }
                    }
                }
            }

            return 0;
        case 'c':
            // cut (via) layer
            if (leflayer->hasSpacingNumber()) {
                switch (leflayer->numSpacing()) {
                    case 0:
                        logger.warning("layer has no spacing: %s", name.c_str());
                        return 0;
                    case 1:
                        layer.spacing = leflayer->spacing(0);
                        return 0;
                    default:
                        layer.spacing = leflayer->spacing(0);
                        logger.warning("layer has multiple spacing: %s", name.c_str());
                        return 0;
                }
            }
            return 0;
        default:
            return 0;
    }
}

//-----Via-----
int readLefVia(lefrCallbackType_e c, lefiVia* lvia, lefiUserData ud) {
    Database* db = (Database*)ud;
    int convertFactor = db->LefConvertFactor;

    string name(lvia->lefiVia::name());
    ViaType* via = db->addViaType(name, false);
    for (int i = 0; i < lvia->lefiVia::numLayers(); i++) {
        string layername(lvia->lefiVia::layerName(i));
        Layer* layer = db->getLayer(layername);
        if (!layer) {
            logger.error("layer not found: %s", layername.c_str());
        }
        for (int j = 0; j < lvia->lefiVia::numRects(i); ++j) {
            via->addRect(*layer,
                         lround(lvia->lefiVia::xl(i, j) * convertFactor),
                         lround(lvia->lefiVia::yl(i, j) * convertFactor),
                         lround(lvia->lefiVia::xh(i, j) * convertFactor),
                         lround(lvia->lefiVia::yh(i, j) * convertFactor));
        }
    }
    return 0;
}

//-----CellType-----
int readLefMacroBegin(lefrCallbackType_e c, const char* macroName, lefiUserData ud) {
    Database* db = (Database*)ud;

    string name(macroName);
    db->addCellType(name, db->celltypes.size());
    return 0;
}

int readLefObs(lefrCallbackType_e c, lefiObstruction* obs, lefiUserData ud) {
    Database* db = (Database*)ud;
    int convertFactor = db->LefConvertFactor;
    CellType* celltype = db->celltypes.back();  // get the last inserted celltype

    Layer* layer = nullptr;
    lefiGeometries* geom = obs->lefiObstruction::geometries();
    int nitem = geom->lefiGeometries::numItems();
    for (int i = 0; i < nitem; i++) {
        if (geom->lefiGeometries::itemType(i) == lefiGeomLayerE) {
            const string layername(geom->lefiGeometries::getLayer(i));
            layer = db->getLayer(layername);
        } else if (geom->lefiGeometries::itemType(i) == lefiGeomRectE && layer != NULL) {
            lefiGeomRect* rect = geom->lefiGeometries::getRect(i);
            celltype->addObs(*layer,
                             lround(rect->xl * convertFactor),
                             lround(rect->yl * convertFactor),
                             lround(rect->xh * convertFactor),
                             lround(rect->yh * convertFactor));
        }
    }
    return 0;
}

int readLefPin(lefrCallbackType_e c, lefiPin* pin, lefiUserData ud) {
    Database* db = (Database*)ud;
    int convertFactor = db->LefConvertFactor;
    CellType* celltype = db->celltypes.back();  // get the last inserted celltype

    const string name(pin->name());
    if (name == "VBB" || name == "VPP") {
        return 0;
    }

    char direction = 'x';
    char type = 's';
    if (pin->hasUse()) {
        const string use(pin->use());
        if (use == "ANALOG") {
            //  Pin is used for analog connectivity.
            type = 'a';
        } else if (use == "CLOCK") {
            //  Pin is used for clock net connectivity.
            type = 'c';
        } else if (use == "GROUND") {
            //  Pin is used for connectivity to the chip-
            //  level ground distribution network.
            type = 'g';
        } else if (use == "POWER") {
            //  Pin is used for connectivity to the chip-
            //  level power distribution network.
            type = 'p';
        } else if (use == "SIGNAL") {
            //  Pin is used for regular net connectivity.
        } else {
            logger.error("unknown use: %s.%s", celltype->name.c_str(), use.c_str());
        }
    }

    if (pin->hasDirection()) {
        const string sDir(pin->direction());
        if (sDir == "OUTPUT") {
            direction = 'o';
        } else if (sDir == "OUTPUT TRISTATE") {
            direction = 'o';
            logger.warning(
                "treat pin %s.%s direction %s as OUTPUT", celltype->name.c_str(), name.c_str(), sDir.c_str());
        } else if (sDir == "INPUT") {
            direction = 'i';
        } else if (sDir == "INOUT") {
            if (name != "VDD" && name != "vdd" && name != "VSS" && name != "vss") {
                logger.warning("unknown pin %s.%s direction: %s", celltype->name.c_str(), name.c_str(), sDir.c_str());
            }
        } else {
            logger.error("unknown pin %s.%s direction: %s", celltype->name.c_str(), name.c_str(), sDir.c_str());
        }
    }

    // switch (direction) {
    //     case 'i':
    //         rtimer.lLib.cells[celltype->libcell()].addIPin(name);
    //         break;
    //     case 'o':
    //         rtimer.lLib.cells[celltype->libcell()].addOPin(name);
    //         break;
    //     default:
    //         break;
    // }

    if (pin->hasTaperRule()) {
        logger.warning("pin %s has taper rule %s", name.c_str(), pin->taperRule());
    }

    PinType* pintype = celltype->addPin(name, direction, type);

    set<Point> nodes;
    Layer* layer = nullptr;
    lefiGeomRect* rect = nullptr;
    lefiGeomPolygon* polygon = nullptr;
    lefiGeometries* geom = pin->port(0);
    vector<pair<int, int>> poly;
    unsigned bj = 0;
    bool has45 = false;
    for (unsigned i = 0; i != (unsigned)geom->numItems(); ++i) {
        switch (geom->itemType(i)) {
            case lefiGeomUnknown:
                logger.warning("lefiGeomUnknown: %s.%s", celltype->name.c_str(), name.c_str());
                break;
            case lefiGeomLayerE:
                layer = db->getLayer(string(geom->getLayer(i)));
                assert(layer);
                break;
            case lefiGeomRectE:
                rect = geom->getRect(i);
                pintype->addShape(*layer,
                                  lround(rect->xl * convertFactor),
                                  lround(rect->yl * convertFactor),
                                  lround(rect->xh * convertFactor),
                                  lround(rect->yh * convertFactor));
                break;
            case lefiGeomPolygonE:
                polygon = geom->getPolygon(i);
                poly.clear();
                bj = 0;
                has45 = false;
                for (unsigned j = 0; static_cast<int>(j) != polygon->numPoints; ++j) {
                    poly.emplace_back(lround(polygon->x[j] * convertFactor), lround(polygon->y[j] * convertFactor));
                    if (polygon->y[j] < polygon->y[bj]) {
                        bj = j;
                    }
                    const unsigned jPost = (j + 1) % polygon->numPoints;
                    if (polygon->x[j] != polygon->x[jPost] && polygon->y[j] != polygon->y[jPost]) {
                        has45 = true;
                    }
                }
                if (has45) {
                    poly.clear();
                    const unsigned jPre = (bj + polygon->numPoints - 1) % polygon->numPoints;
                    const unsigned jPost = (bj + 1) % polygon->numPoints;
                    bool isClockwise = false;
                    if (polygon->x[jPre] > polygon->x[jPost]) {
                        isClockwise = true;
                    }
                    for (unsigned k = 0; static_cast<int>(k) != polygon->numPoints; ++k) {
                        poly.emplace_back(lround(polygon->x[k] * convertFactor), lround(polygon->y[k] * convertFactor));
                        const unsigned kPost = (k + 1) % polygon->numPoints;
                        if (polygon->x[k] == polygon->x[kPost] || polygon->y[k] == polygon->y[kPost]) {
                            continue;
                        }
                        if (isClockwise ^ (polygon->x[k] < polygon->x[kPost]) ^ (polygon->y[k] < polygon->y[kPost])) {
                            poly.emplace_back(lround(polygon->x[k] * convertFactor),
                                              lround(polygon->y[kPost] * convertFactor));
                        } else {
                            poly.emplace_back(lround(polygon->x[kPost] * convertFactor),
                                              lround(polygon->y[k] * convertFactor));
                        }
                    }
                }
                for (unsigned j = 0; j != poly.size(); ++j) {
                    unsigned jPre = (j + poly.size() - 1) % poly.size();
                    unsigned jPost = (j + 1) % poly.size();
                    unsigned char dir = pointDir(poly[j].first, poly[j].second, poly[jPre].first, poly[jPre].second);
                    dir |= pointDir(poly[j].first, poly[j].second, poly[jPost].first, poly[jPost].second);
                    if (dir == (DIR_DOWN | DIR_UP) || dir == (DIR_LEFT | DIR_RIGHT)) {
                        continue;
                    }
                    nodes.emplace(poly[j].first, poly[j].second, dir);
                }
                while (nodes.size()) {
                    set<Point>::iterator pk;
                    set<Point>::iterator pl;
                    set<Point>::iterator pm;
                    set<Point>::iterator pn;

                    while (nodes.size()) {
                        pk = nodes.begin();
                        pl = nodes.begin();
                        ++pl;

                        if (pk->x != pl->x) {
                            break;
                        }

                        nodes.erase(nodes.begin());
                        nodes.erase(nodes.begin());
                    }

                    if (nodes.empty()) {
                        break;
                    }

                    if (nodes.size() < 4) {
                        cout << "Error when partitioning rectangles." << endl;
                        cout << "Remaining points: ";
                        for (const Point& p : nodes) {
                            printf("(%d, %d), ", p.x, p.y);
                        }
                        cout << endl;
                        break;
                    }

                    for (pm = nodes.begin(); pm->y <= pk->y || pm->x < pk->x || pm->x >= pl->x; ++pm) {
                    }

                    for (pn = pm; pn->x < pl->x && pn->y <= pm->y; ++pn) {
                    }

                    pintype->addShape(*layer, pk->x, pk->y, pl->x, pm->y);

                    Point ul(pk->x, pm->y, DIR_DOWN | DIR_RIGHT);
                    Point ur(pl->x, pm->y, DIR_DOWN | DIR_LEFT);

                    // Insert or erase the upper-right point of the rectangle.
                    if (*pn == ur && pn->outdir != (DIR_UP | DIR_RIGHT)) {
                        nodes.erase(pn);
                    } else {
                        nodes.insert(ur);
                    }

                    // Insert or erase the upper-left point of the rectangle.
                    pm = nodes.find(ul);
                    if (pm == nodes.end()) {
                        nodes.insert(ul);
                    } else {
                        nodes.erase(pm);
                    }

                    nodes.erase(nodes.begin());
                    nodes.erase(nodes.begin());
                }
                break;
            default:
                logger.warning(
                    "unknown lefiGeomEnum %u: %s.%s", geom->itemType(i), celltype->name.c_str(), name.c_str());
                break;
        }
    }
    return 0;
}

int readLefSite(lefrCallbackType_e c, lefiSite* site, lefiUserData ud) {
    Database* db = reinterpret_cast<Database*>(ud);
    int convertFactor = db->LefConvertFactor;
    int width = lround(site->sizeX() * convertFactor);
    int height = lround(site->sizeY() * convertFactor);
    logger.info("site name: %s class: %s siteW: %d siteH: %d", site->name(), site->siteClass(), width, height);
    db->addSite(site->name(), site->siteClass(), width, height);
    return 0;
}

int readLefMacro(lefrCallbackType_e c, lefiMacro* macro, lefiUserData ud) {
    Database* db = reinterpret_cast<Database*>(ud);
    int convertFactor = db->LefConvertFactor;
    CellType* celltype = db->celltypes.back();

    celltype->width = round(macro->sizeX() * convertFactor);
    celltype->height = round(macro->sizeY() * convertFactor);

    if (macro->lefiMacro::hasClass()) {
        char clsname[64] = {0};
        strcpy(clsname, macro->macroClass());
        if (!strcmp(clsname, "CORE")) {
            celltype->cls = 'c';
            celltype->stdcell = true;
        } else if (!strcmp(clsname, "BLOCK")) {
            celltype->cls = 'b';
        } else {
            celltype->cls = clsname[0];
            logger.warning("Class type is not defined: %s", clsname);
        }
    } else {
        celltype->cls = 'c';
    }

    if (macro->lefiMacro::hasOrigin()) {
        celltype->setOrigin(macro->originX() * convertFactor, macro->originY() * convertFactor);
    }

    if (macro->hasXSymmetry()) {
        celltype->setXSymmetry();
    }
    if (macro->hasYSymmetry()) {
        celltype->setYSymmetry();
    }
    if (macro->has90Symmetry()) {
        celltype->set90Symmetry();
    }

    if (macro->hasSiteName()) {
        celltype->siteName(string(macro->siteName()));
    }

    for (int i = 0; i < macro->numProperties(); ++i) {
        if (!strcmp(macro->propName(i), "LEF58_EDGETYPE")) {
            stringstream ssedgetype(macro->propValue(i));
            string buffer;
            while (!ssedgetype.eof()) {
                ssedgetype >> buffer;
                if (buffer == "EDGETYPE") {
                    string edgeside;
                    string edgetype;
                    ssedgetype >> edgeside >> edgetype;
                    if (edgeside == "LEFT") {
                        celltype->edgetypeL = db->edgetypes.getEdgeType(edgetype);
                    } else if (edgeside == "RIGHT") {
                        celltype->edgetypeR = db->edgetypes.getEdgeType(edgetype);
                    } else if (edgeside == "BOTTOM") {
                        static bool missBot = true;
                        if (missBot) {
                            logger.warning("unknown edge side: %s", edgeside.c_str());
                            missBot = false;
                        }
                    } else if (edgeside == "TOP") {
                        static bool missTop = true;
                        if (missTop) {
                            logger.warning("unknown edge side: %s", edgeside.c_str());
                            missTop = false;
                        }
                    } else {
                        logger.warning("unknown edge side: %s", edgeside.c_str());
                    }
                }
            }
        }
    }
    return 0;
}

int readLefMacroEnd(lefrCallbackType_e c, const char* macroName, lefiUserData ud) {
    //  TODO: sort by pin name.
    return 0;
}

/******************/
/*  DEF Callback  */
/******************/

//-----Unit-----
int readDefUnits(defrCallbackType_e c, double d, defiUserData ud) {
    reinterpret_cast<Database*>(ud)->DBU_Micron = d;
    return 0;
}

//-----Version-----
int readDefVersion(defrCallbackType_e c, double d, defiUserData ud) {
    reinterpret_cast<Database*>(ud)->version = d;
    return 0;
}

//-----Design-----
int readDefDesign(defrCallbackType_e c, const char* name, defiUserData ud) {
    reinterpret_cast<Database*>(ud)->designName = name;
    return 0;
}

//-----Die Area-----
int readDefDieArea(defrCallbackType_e c, defiBox* dbox, defiUserData ud) {
    Database* db = (Database*)ud;
    db->dieLX = dbox->xl();
    db->dieLY = dbox->yl();
    db->dieHX = dbox->xh();
    db->dieHY = dbox->yh();
    return 0;
}

//-----Row-----
int readDefRow(defrCallbackType_e c, defiRow* drow, defiUserData ud) {
    ((Database*)ud)
        ->addRow(string(drow->name()),
                 string(drow->macro()),
                 drow->x(),
                 drow->y(),
                 drow->xNum(),
                 drow->yNum(),
                 isFlipY(drow->orient()),
                 drow->xStep(),
                 drow->yStep());
    return 0;
}

//-----Track-----
int readDefTrack(defrCallbackType_e c, defiTrack* dtrack, defiUserData ud) {
    Database* db = (Database*)ud;
    char direction = 'x';
    if (strcmp(dtrack->macro(), "X") == 0) {
        direction = 'v';
    } else if (strcmp(dtrack->macro(), "Y") == 0) {
        direction = 'h';
    }
    Track* track = db->addTrack(direction, dtrack->x(), dtrack->xNum(), dtrack->xStep());
    for (int i = 0; i < dtrack->numLayers(); i++) {
        const string layername(dtrack->layer(i));
        track->addLayer(layername);
    }
    for (int i = 0; i < dtrack->numLayers(); i++) {
        const string layername(dtrack->layer(i));
        Layer* layer = db->getLayer(layername);
        if (layer) {
            if (track->direction == 'x' || layer->direction == 'x') {
                layer->track = *track;
            } else if (layer->direction == track->direction) {
                layer->track = *track;
            } else if (layer->direction != track->direction) {
                layer->nonPreferDirTrack = *track;
            } else {
                logger.error("wrong definition of tracks for layer %s", layername.c_str());
            }
        } else {
            logger.error("layer name not found: %s", layername.c_str());
        }
    }
    return 0;
}

//-----GcellGrid-----
int readDefGcellGrid(defrCallbackType_e c, defiGcellGrid* dgrid, defiUserData ud) {
    Database* db = (Database*)ud;
    const string macro(dgrid->macro());
    if (macro == "X") {
        db->gcellgrid.numX.emplace_back(dgrid->xNum());
        db->gcellgrid.stepX.emplace_back(dgrid->xStep());
        db->gcellgrid.startX.emplace_back(dgrid->x());
    } else if (macro == "Y") {
        db->gcellgrid.numY.emplace_back(dgrid->xNum());
        db->gcellgrid.stepY.emplace_back(dgrid->xStep());
        db->gcellgrid.startY.emplace_back(dgrid->x());
    }
    return 0;
}

//-----Via-----
int readDefVia(defrCallbackType_e c, defiVia* dvia, defiUserData ud) {
    Database* db = (Database*)ud;

    const string name(dvia->name());
    ViaType* via = db->getViaType(name);
    if (via) {
        via->isDef(true);
    } else {
        via = db->addViaType(name, true);
    }

    char* dvialayer = nullptr;
    int lx = 0;
    int ly = 0;
    int hx = 0;
    int hy = 0;
    for (int i = 0; i != dvia->numLayers(); ++i) {
        dvia->layer(i, &dvialayer, &lx, &ly, &hx, &hy);
        Layer* layer = db->getLayer(string(dvialayer));
        if (layer) {
            via->addRect(*layer, lx, ly, hx, hy);
        } else {
            logger.info("layer name not found: %s", dvialayer);
        }
    }

    if (dvia->defiVia::numPolygons()) {
        // TODO
    }

    if (dvia->defiVia::hasViaRule()) {
        ViaRule& rule = via->rule;
        char* vrn;                             // VIARULE Name
        int xs, ys;                            // CUTSIZE
        char *botlayer, *cutlayer, *toplayer;  // LAYERS
        int xcs, ycs;                          // CUTSPACING
        int xbe, ybe, xte, yte;                // ENCLOSURE
        int cr, cc;                            // ROWCOL
        int xo, yo;                            // ORIGIN
        int xbo, ybo, xto, yto;                // OFFSET

        (void)dvia->defiVia::viaRule(
            &vrn, &xs, &ys, &botlayer, &cutlayer, &toplayer, &xcs, &ycs, &xbe, &ybe, &xte, &yte);

        rule.hasViaRule = true;
        rule.name = string(vrn);

        rule.cutSize = std::make_pair(xs, ys);
        rule.cutSpacing = std::make_pair(xcs, ycs);
        rule.botEnclosure = std::make_pair(xbe, ybe);
        rule.topEnclosure = std::make_pair(xte, yte);
        rule.botLayer = db->getLayer(string(botlayer));
        rule.cutLayer = db->getLayer(string(cutlayer));
        rule.topLayer = db->getLayer(string(toplayer));

        if (dvia->defiVia::hasRowCol()) {
            (void)dvia->defiVia::rowCol(&cr, &cc);
            rule.numCutRows = cr;
            rule.numCutCols = cc;
        }

        if (dvia->defiVia::hasOrigin()) {
            (void)dvia->defiVia::origin(&xo, &yo);
            rule.originOffset = std::make_pair(xo, yo);
        }

        if (dvia->defiVia::hasOffset()) {
            (void)dvia->defiVia::offset(&xbo, &ybo, &xto, &yto);
            rule.botOffset = std::make_pair(xbo, ybo);
            rule.topOffset = std::make_pair(xto, yto);
        }

        if (dvia->defiVia::hasCutPattern()) {
            const string cutPattern(dvia->defiVia::cutPattern());
        }
    }

    return 0;
}

//-----NonDefaultRule-----

int readDefNdr(defrCallbackType_e c, defiNonDefault* nd, defiUserData ud) {
    Database* db = (Database*)ud;
    NDR* ndr = db->addNDR(string(nd->name()), nd->hasHardspacing());

    for (int i = 0; i < nd->numLayers(); i++) {
        int space = 0;
        if (nd->hasLayerSpacing(i)) {
            space = nd->layerSpacingVal(i);
        }
        ndr->rules.emplace_back(db->getLayer(string(nd->layerName(i))), nd->defiNonDefault::layerWidthVal(i), space);
    }
    for (int i = 0; i < nd->numVias(); i++) {
        string vianame(nd->viaName(i));
        ViaType* viatype = db->getViaType(vianame);
        if (!viatype) {
            logger.warning("NDR via type not found: %s", vianame.c_str());
        }
        ndr->vias.push_back(viatype);
    }
    return 0;
}

//-----Cell-----
int readDefComponentStart(defrCallbackType_e c, int num, defiUserData ud) {
    reinterpret_cast<Database*>(ud)->reserveCells(num);
    return 0;
}

int readDefComponent(defrCallbackType_e c, defiComponent* co, defiUserData ud) {
    Database* db = (Database*)ud;

    Cell* cell = db->addCell(co->id(), db->getCellType(co->name()));

    if (co->isUnplaced()) {
        cell->fixed(false);
        cell->unplace();
    } else if (co->isPlaced()) {
        cell->place(co->placementX(), co->placementY(), isFlipX(co->placementOrient()), isFlipY(co->placementOrient()));
        cell->fixed(false);
        if (co->placementOrient() % 2 == 1) {
            // 0:N, 1:W, 2:S, 3:E, 4:FN, 5:FW, 6:FS, 7:FE
            logger.warning(
                "Cell [%s]'s placementOrient [%d] is not supported.", cell->name().c_str(), co->placementOrient());
        }
    } else if (co->isFixed()) {
        cell->place(co->placementX(), co->placementY(), isFlipX(co->placementOrient()), isFlipY(co->placementOrient()));
        cell->fixed(true);
        if (co->placementOrient() % 2 == 1) {
            // 0:N, 1:W, 2:S, 3:E, 4:FN, 5:FW, 6:FS, 7:FE
            logger.warning(
                "Cell [%s]'s placementOrient [%d] is not supported.", cell->name().c_str(), co->placementOrient());
        }
    }
    return 0;
}

//-----Pin-----
int readDefPin(defrCallbackType_e c, defiPin* dpin, defiUserData ud) {
    Database* db = (Database*)ud;

    char direction = 'x';
    if (dpin->direction()) {
        if (!strcmp(dpin->direction(), "INPUT")) {
            // INPUT to the chip, output from external
            direction = 'o';
        } else if (!strcmp(dpin->direction(), "OUTPUT")) {
            // OUTPUT to the chip, input to external
            direction = 'i';
        } else {
            logger.warning("unknown pin signal direction: %s", dpin->direction());
        }
    } else {
        string pinName(dpin->pinName());
        logger.warning("Pin %s has no pin signal direction", pinName.c_str());
    }

    IOPin* iopin = db->addIOPin(string(dpin->pinName()), string(dpin->netName()), direction);

    if (dpin->hasPlacement()) {
        iopin->x = dpin->placementX();
        iopin->y = dpin->placementY();
        iopin->_orient = dpin->orient();
    }

    if (dpin->hasLayer()) {
        for (int i = 0; i < dpin->numLayer(); i++) {
            Layer* layer = db->getLayer(string(dpin->layer(i)));
            int lx, ly, hx, hy;
            dpin->bounds(i, &lx, &ly, &hx, &hy);
            iopin->type->addShape(*layer, lx, ly, hx, hy);
        }
    }

    return 0;
}

//-----Blockages-----
int readDefBlockage(defrCallbackType_e c, defiBlockage* dblk, defiUserData ud) {
    Database* db = (Database*)ud;

    if (dblk->hasLayer()) {
        string layername(dblk->layerName());
        Layer* layer = db->getLayer(layername);
        if (!layer) {
            logger.error("layer not found: %s", layername.c_str());
            return 1;
        }
        for (int i = 0; i < dblk->numRectangles(); ++i) {
            db->routeBlockages.emplace_back(*layer, dblk->xl(i), dblk->yl(i), dblk->xh(i), dblk->yh(i));
        }
    } else if (dblk->hasPlacement()) {
        for (int i = 0; i < dblk->defiBlockage::numRectangles(); ++i) {
            db->placeBlockages.emplace_back(dblk->xl(i), dblk->yl(i), dblk->xh(i), dblk->yh(i));
        }
    }
    return 0;
}

//-----Net-----
int readDefSNetStart(defrCallbackType_e c, int num, defiUserData ud) {
    reinterpret_cast<Database*>(ud)->reserveSNets(num);
    return 0;
}

int readDefSNet(defrCallbackType_e c, defiNet* dnet, defiUserData ud) {
    Database* db = (Database*)ud;

    SNet* snet = db->addSNet(dnet->name());

    if (dnet->hasUse()) {
        const string use(dnet->use());
        if (use == "POWER") {
            snet->type = 'p';
        } else if (use == "GROUND") {
            snet->type = 'g';
        } else {
            logger.error("unknown use: %s", use.c_str());
        }
    }

    int path = DEFIPATH_DONE;
    string layername = "";
    Layer* layer = nullptr;
    string vianame = "";
    ViaType* viatype = nullptr;
    int fx = 0;
    int fy = 0;
    int fz = 0;
    int tx = 0;
    int ty = 0;
    int tz = 0;
    int lx = 0;
    int ly = 0;
    int hx = 0;
    int hy = 0;
    int ext = 0;
    int nextExt = 0;
    int wirewidth = 0;
    for (unsigned i = 0; static_cast<int>(i) < dnet->numWires(); ++i) {
        const defiWire* dwire = dnet->wire(i);
        unsigned nNodes = 0;
        for (unsigned j = 0; static_cast<int>(j) < dwire->numPaths(); ++j) {
            const defiPath* dpath = dwire->path(j);
            dpath->initTraverse();
            while ((path = dpath->next()) != DEFIPATH_DONE) {
                switch (path) {
                    case DEFIPATH_LAYER:
                        layername = dpath->getLayer();
                        layer = db->getLayer(layername);
                        if (!layer) {
                            logger.error("Layer is not defined: %s", layername.c_str());
                            return false;
                        }
                        break;
                    case DEFIPATH_VIA:
                        vianame = dpath->getVia();
                        viatype = db->getViaType(vianame);
                        if (!viatype) {
                            logger.error("Via type is not defined: %s", vianame.c_str());
                            return false;
                        }
                        snet->addVia(viatype, fx, fy);
                        break;
                    case DEFIPATH_WIDTH:
                        wirewidth = dpath->getWidth();
                        nNodes = 0;
                        break;
                    case DEFIPATH_POINT:
                        if (nNodes) {
                            dpath->getPoint(&tx, &ty);
                            tz = -1;
                            ext = (fz == -1) ? 0 : fz;
                            nextExt = (tz == -1) ? 0 : tz;
                            if (fy == ty) {
                                lx = fx < tx ? fx - ext : tx - nextExt;
                                hx = fx < tx ? tx + nextExt : fx + ext;
                                ly = fy - wirewidth / 2;
                                hy = fy + wirewidth / 2;
                            } else if (fx == tx) {
                                lx = fx - wirewidth / 2;
                                hx = fx + wirewidth / 2;
                                ly = fy < ty ? fy - ext : ty - nextExt;
                                hy = fy < ty ? ty + nextExt : fy + ext;
                            } else {
                                // WARN
                            }
                            snet->addShape(*layer, lx, ly, hx, hy);
                            if ((layer->rIndex == 0 || layer->rIndex == 1) && fy == ty) {
                                db->powerNet.addRail(snet, lx, hx, fy);
                            }
                            fx = tx;
                            fy = ty;
                            fz = tz;
                        } else {
                            dpath->getPoint(&fx, &fy);
                            fz = -1;
                        }
                        ++nNodes;
                        break;
                    case DEFIPATH_FLUSHPOINT:
                        if (nNodes) {
                            dpath->getFlushPoint(&tx, &ty, &tz);
                            ext = (fz == -1) ? 0 : fz;
                            nextExt = (tz == -1) ? 0 : tz;
                            if (fy == ty) {
                                lx = fx < tx ? fx - ext : tx - nextExt;
                                hx = fx < tx ? tx + nextExt : fx + ext;
                                ly = fy - wirewidth / 2;
                                hy = fy + wirewidth / 2;
                            } else if (fx == tx) {
                                lx = fx - wirewidth / 2;
                                hx = fx + wirewidth / 2;
                                ly = fy < ty ? fy - ext : ty - nextExt;
                                hy = fy < ty ? ty + nextExt : fy + ext;
                            } else {
                                // WARN
                            }
                            snet->addShape(*layer, lx, ly, hx, hy);
                            if ((layer->rIndex == 0 || layer->rIndex == 1) && fy == ty) {
                                db->powerNet.addRail(snet, lx, hx, fy);
                            }
                            fx = tx;
                            fy = ty;
                            fz = tz;
                        } else {
                            dpath->getFlushPoint(&fx, &fy, &fz);
                        }
                        ++nNodes;
                        break;
                }
            }
        }
    }
    return 0;
}

int readDefNetStart(defrCallbackType_e c, int num, defiUserData ud) {
    reinterpret_cast<Database*>(ud)->reserveNets(num);
    return 0;
}

int readDefNet(defrCallbackType_e c, defiNet* dnet, defiUserData ud) {
    Database* db{reinterpret_cast<Database*>(ud)};
    NDR* ndr{nullptr};

    if (dnet->hasNonDefaultRule()) {
        string designrulename(dnet->nonDefaultRule());
        ndr = db->getNDR(designrulename);
        if (!ndr) {
            logger.warning("NDR rule is not defined: %s", designrulename.c_str());
        }
    }
    if ((unsigned)dnet->numConnections() == 0) {
        string netName(dnet->name());
        logger.warning("Net %s is 0-Pin net. Ignore.", netName.c_str());
        return 0;
    }

    Net* net = db->addNet(dnet->name(), ndr);

    for (unsigned i = 0; i != (unsigned)dnet->numConnections(); ++i) {
        Pin* pin = nullptr;
        if (strcmp(dnet->instance(i), "PIN") == 0) {
            string iopinname(dnet->pin(i));
            IOPin* iopin = db->getIOPin(iopinname);
            if (!iopin) {
                logger.warning("IO pin is not defined: %s", iopinname.c_str());
            }
            pin = iopin->pin;
            if (pin->is_connected) {
                string netName(dnet->name());
                logger.warning("IO Pin is re-connected: %s %s", netName.c_str(), iopinname.c_str());
            }
            iopin->is_connected = true;
        } else {
            string cellname(dnet->instance(i));
            string pinname(dnet->pin(i));
            Cell* cell = db->getCell(cellname);
            if (!cell) {
                logger.warning("Cell is not defined: %s", cellname.c_str());
            }
            pin = cell->pin(pinname);
            if (!pin) {
                string netName(dnet->name());
                logger.warning("Pin is not defined: %s %s %s", netName.c_str(), cellname.c_str(), pinname.c_str());
            }
            if (pin->is_connected) {
                string netName(dnet->name());
                logger.warning("Pin is re-connected: %s %s %s", netName.c_str(), cellname.c_str(), pinname.c_str());
            }
            cell->is_connected = true;
        }
        pin->net = net;
        pin->is_connected = true;
        net->addPin(pin);
    }

    int path = DEFIPATH_DONE;
    const Layer* layer = nullptr;
    int fromx = 0;
    int fromy = 0;
    int fromz = 0;
    int tox = 0;
    int toy = 0;
    int toz = 0;
    unsigned nNodes = 0;
    for (unsigned i = 0; static_cast<int>(i) < dnet->numWires(); ++i) {
        const defiWire* dwire = dnet->wire(i);
        for (unsigned j = 0; static_cast<int>(j) < dwire->numPaths(); ++j) {
            const defiPath* dpath = dwire->path(j);
            dpath->initTraverse();
            while ((path = dpath->next()) != DEFIPATH_DONE) {
                switch (path) {
                    case DEFIPATH_LAYER:
                        nNodes = 0;
                        layer = db->getLayer(string(dpath->getLayer()));
                        break;
                    case DEFIPATH_VIA:
                        //  printf("%s ", dpath->getVia());
                        break;
                    case DEFIPATH_VIAROTATION:
                        break;
                    case DEFIPATH_WIDTH:
                        //  printf("%d ", dpath->getWidth());
                        break;
                    case DEFIPATH_POINT:
                        if (nNodes) {
                            dpath->getPoint(&tox, &toy);
                            toz = -1;
                            net->addWire(layer, fromx, fromy, fromz, tox, toy, toz);
                            fromx = tox;
                            fromy = toy;
                            fromz = toz;
                        } else {
                            dpath->getPoint(&fromx, &fromy);
                        }
                        ++nNodes;
                        break;
                    case DEFIPATH_FLUSHPOINT:
                        if (nNodes) {
                            dpath->getFlushPoint(&tox, &toy, &toz);
                            net->addWire(layer, fromx, fromy, fromz, tox, toy, toz);
                            fromx = tox;
                            fromy = toy;
                            fromz = toz;
                        } else {
                            dpath->getFlushPoint(&fromx, &fromy, &fromz);
                        }
                        ++nNodes;
                        break;
                    case DEFIPATH_TAPER:
                        //  printf("TAPER ");
                        break;
                    case DEFIPATH_TAPERRULE:
                        //  printf("TAPERRULE %s ", dpath->getTaperRule());
                        break;
                    case DEFIPATH_STYLE:
                        //  printf("STYLE %d ", dpath->getStyle());
                        break;
                }
            }
        }
    }

    return 0;
}

//-----Region-----
int readDefRegion(defrCallbackType_e c, defiRegion* dreg, defiUserData ud) {
    Database* db = (Database*)ud;

    char type = 'f';
    if (dreg->hasType()) {
        if (!strcmp(dreg->type(), "FENCE")) {
            type = 'f';
        } else if (!strcmp(dreg->type(), "GUIDE")) {
            type = 'g';
        } else {
            logger.warning("Unknown region type: %s", dreg->type());
        }
    } else {
        logger.warning("Region is defined without type, use default region type = FENCE");
    }

    Region* region = db->addRegion(string(dreg->name()), type);

    for (unsigned i = 0; (int)i < dreg->numRectangles(); ++i) {
        region->addRect(dreg->xl(i), dreg->yl(i), dreg->xh(i), dreg->yh(i));
    }
    return 0;
}

//-----Group-----
// #define GROUP_MARKER 9999999 //first mark all group member cell with this marker, then replace the value in one scan
int readDefGroupName(defrCallbackType_e c, const char* cl, defiUserData ud) { return 0; }

int readDefGroupMember(defrCallbackType_e c, const char* cl, defiUserData ud) {
    ((Database*)ud)->regions[0]->members.emplace_back(cl);
    return 0;
}

int readDefGroup(defrCallbackType_e c, defiGroup* dgp, defiUserData ud) {
    Database* db = (Database*)ud;
    if (dgp->hasRegionName()) {
        string regionname(dgp->regionName());
        Region* region = db->getRegion(regionname);
        if (!region) {
            logger.warning("Region is not defined: %s", regionname.c_str());
            return 1;
        }
        region->members = db->regions[0]->members;
        db->regions[0]->members.clear();
    }
    return 0;
}
