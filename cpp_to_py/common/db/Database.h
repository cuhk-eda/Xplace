#pragma once

#include "DatabaseClass.h"
#include "Setting.h"
std::string validate_token(std::string& name);

namespace db {

#define CLEAR_POINTER_LIST(list) \
    {                            \
        for (auto obj : list) {  \
            delete obj;          \
        }                        \
        list.clear();            \
    }

#define CLEAR_POINTER_MAP(map) \
    {                          \
        for (auto p : map) {   \
            delete p.second;   \
        }                      \
        map.clear();           \
    }

class Database {
public:
    enum IssueType {
        E_ROW_EXCEED_DIE,
        E_OVERLAP_ROWS,
        W_NON_UNIFORM_SITE_WIDTH,
        W_NON_HORIZONTAL_ROW,
        E_MULTIPLE_NET_DRIVING_PIN,
        E_NO_NET_DRIVING_PIN
    };

    robin_hood::unordered_map<string, CellType*> name_celltypes;
    robin_hood::unordered_map<string, Cell*> name_cells;
    robin_hood::unordered_map<string, Net*> name_nets;
    robin_hood::unordered_map<string, IOPin*> name_iopins;
    robin_hood::unordered_map<string, ViaType*> name_viatypes;
    robin_hood::unordered_map<string, Layer*> name_layers;
    bool verilog_read = false;

    string module_name = "";
    VerilogParser* verilog_parser = nullptr;

    vector<Layer> layers;
    vector<Site*> lefsites;
    vector<ViaType*> viatypes;
    vector<CellType*> celltypes;

    vector<Cell*> cells;
    vector<IOPin*> iopins;
    vector<Net*> nets;
    vector<Row*> rows;
    vector<Region*> regions;
    std::map<string, NDR*> ndrs;
    vector<SNet*> snets;
    vector<Track*> tracks;

    vector<Geometry> routeBlockages;
    vector<Rectangle> placeBlockages;

    PowerNet* powerNet = nullptr;

private:
    static const size_t _bufferCapacity = 128 * 1024;
    size_t _bufferSize = 0;
    char* _buffer = nullptr;

public:
    unsigned siteW = 0;
    int siteH = 0;
    unsigned nSitesX = 0;
    unsigned nSitesY = 0;

    SiteMap* siteMap = nullptr;
    GCellGrid* gcellgrid = nullptr;

    BsRouteInfo* bsRouteInfo = nullptr;
    EdgeTypes* edgetypes = nullptr;

    int dieLX, dieLY, dieHX, dieHY;
    int coreLX, coreLY, coreHX, coreHY;

    double maxDensity = 0;
    double maxDisp = 0;

    int LefConvertFactor;
    double DBU_Micron;
    double version;
    string designName;

    vector<IssueType> dbIssues;

public:
    Database();
    ~Database();
    void clear();
    void clearTechnology();
    void clearLibrary();
    void clearDesign();

    Layer& addLayer(const string& name, const char type = 'x');
    Site* addSite(const string& name, const string& siteClassName, const int w, const int h);
    ViaType* addViaType(const string& name, bool isDef);
    ViaType* addViaType(const string& name) { return addViaType(name, false); }
    CellType* addCellType(const string& name, unsigned libcell);
    void reserveCells(const size_t n) { cells.reserve(n); }
    Cell* addCell(const string& name, CellType* type = nullptr);
    IOPin* addIOPin(const string& name = "", const string& netName = "", const char direction = 'x');
    void reserveNets(const size_t n) { nets.reserve(n); }
    Net* addNet(const string& name = "", const NDR* ndr = nullptr);
    Row* addRow(const string& name,
                const string& macro,
                const int x,
                const int y,
                const unsigned xNum = 0,
                const unsigned yNum = 0,
                const int orient = 0,
                const bool flip = false,
                const unsigned xStep = 0,
                const unsigned yStep = 0);
    Track* addTrack(char direction, double start, double num, double step);
    Region* addRegion(const string& name = "", const char type = 'x');
    NDR* addNDR(const string& name, const bool hardSpacing);
    void reserveSNets(const size_t n) { snets.reserve(n); }
    SNet* addSNet(const string& name);

    Layer* getRLayer(const int index);
    const Layer* getCLayer(const unsigned index) const;
    Layer* getLayer(const string& name);
    CellType* getCellType(const string& name);
    Cell* getCell(const string& name);
    Net* getNet(const string& name);
    Region* getRegion(const string& name);
    Region* getRegion(const unsigned char id);
    NDR* getNDR(const string& name) const;
    IOPin* getIOPin(const string& name) const;
    ViaType* getViaType(const string& name) const;
    SNet* getSNet(const string& name);

    unsigned getNumRLayers() const;
    unsigned getNumCLayers() const;
    unsigned getNumLayers() const;
    unsigned getNumCells() const { return cells.size(); }
    unsigned getNumNets() const { return nets.size(); }
    unsigned getNumRegions() const { return regions.size(); }
    unsigned getNumIOPins() const { return iopins.size(); }
    unsigned getNumCellTypes() const { return celltypes.size(); }

    int getCellTypeSpace(const CellType* L, const CellType* R) const;
    int getCellTypeSpace(const Cell* L, const Cell* R) const;
    int getContainedSites(
        const int lx, const int ly, const int hx, const int hy, int& slx, int& sly, int& shx, int& shy) const;
    int getOverlappedSites(
        const int lx, const int ly, const int hx, const int hy, int& slx, int& sly, int& shx, int& shy) const;

    long long getHPWL();
    long long getCellArea(Region* region = nullptr) const;
    long long getFreeArea(Region* region = nullptr) const;

    bool placed();
    bool globalRouted();
    bool detailedRouted();

    void errorCheck(bool autoFix = true);
    void checkPlaceError();
    void checkDRCError();

    void load();
    void setup();  // call after read
    void reset();
    void save(const std::string& given_prefix);

    /* defined in io/file_lefdef_db.cpp */
public:
    bool readLEF(const std::string& file);
    bool readDEF(const std::string& file);
    bool readDEFPG(const string& file);
    bool writeDEF(const std::string& file);
    bool writeICCAD2017(const string& inputDef, const string& outputDef);
    bool writeICCAD2017(const string& outputDef);
    bool writeComponents(std::ofstream& ofs);
    bool writeNets(std::ofstream& ofs);
    bool writeBuffer(std::ofstream& ofs, const string& line);
    void writeBufferFlush(std::ofstream& ofs);

    bool readBSAux(const std::string& auxFile, const std::string& plFile);
    bool readBSNodes(const std::string& file);
    bool readBSNets(const std::string& file);
    bool readBSScl(const std::string& file);
    bool readBSRoute(const std::string& file);
    bool readBSShapes(const std::string& file);
    bool readBSWts(const std::string& file);
    bool readBSPl(const std::string& file);
    bool writeBSPl(const std::string& file);

    bool readVerilog(const std::string& file);
    bool readVerilog_yy(const std::string &file);
    bool readLiberty(const std::string& file);
    bool write_verilog(const std::string& file);

    bool readConstraints(const std::string& file);
    bool readSize(const std::string& file);

private:
    void SetupLayers();
    void SetupCellLibrary();
    void SetupFloorplan();
    void SetupRegions();
    void SetupSiteMap();
    void SetupRows();
    void SetupRowSegments();
};

}  // namespace db
