#include "common/db/Database.h"

using namespace db;

class BookshelfData {
public:
    int nCells;
    int nTypes;
    unsigned nNets;
    int nRows;

    std::string format;
    robin_hood::unordered_map<string, int> cellMap;
    robin_hood::unordered_map<string, int> typeMap;
    vector<string> cellName;
    vector<int> cellX;
    vector<int> cellY;
    vector<char> cellFixed;
    vector<string> typeName;
    vector<int> cellType;
    vector<int> typeWidth;
    vector<int> typeHeight;
    vector<char> typeFixed;
    vector<vector<vector<int>>> typeShapes;
    robin_hood::unordered_map<string, int> typePinMap;
    vector<int> typeNPins;
    vector<vector<string>> typePinName;
    vector<vector<char>> typePinDir;
    vector<vector<double>> typePinX;
    vector<vector<double>> typePinY;
    vector<string> netName;
    vector<vector<int>> netCells;
    vector<vector<int>> netPins;
    vector<int> rowX;
    vector<int> rowY;
    vector<int> rowSites;
    vector<int> rowXStep;
    vector<int> rowHeight;
    int siteWidth;
    int siteHeight;

    // ICCAD/DAC 2012
    int gridNX;
    int gridNY;
    int gridNZ;
    vector<int> capV;
    vector<int> capH;
    vector<int> wireWidth;
    vector<int> wireSpace;
    vector<int> viaSpace;
    int gridOriginX;
    int gridOriginY;
    int tileW;
    int tileH;
    double blockagePorosity;
    vector<int> IOPinRouteLayer;
    vector<pair<int, vector<int>>> routeBlkgs;  // cellID, BlockedLayers

    BookshelfData() {
        nCells = 0;
        nTypes = 0;
        nNets = 0;
        nRows = 0;
    }

    void clearData() {
        cellMap.clear();
        typeMap.clear();
        cellName.clear();
        cellX.clear();
        cellY.clear();
        cellFixed.clear();
        typeName.clear();
        cellType.clear();
        typeWidth.clear();
        typeHeight.clear();
        typeShapes.clear();
        typePinMap.clear();
        typeNPins.clear();
        typePinName.clear();
        typePinDir.clear();
        typePinX.clear();
        typePinY.clear();
        netName.clear();
        netCells.clear();
        netPins.clear();
        rowX.clear();
        rowY.clear();
        rowSites.clear();
        rowXStep.clear();
        rowHeight.clear();
    }

    void scaleData(int scale) {
        for (int i = 0; i < nCells; i++) {
            cellX[i] *= scale;
            cellY[i] *= scale;
        }
        for (int i = 0; i < nTypes; i++) {
            typeWidth[i] *= scale;
            typeHeight[i] *= scale;
            for (int j = 0; j < typeNPins[i]; j++) {
                typePinX[i][j] *= scale;
                typePinY[i][j] *= scale;
            }
            for (int j = 0; j < typeShapes[i].size(); j++) {
                for (int k = 0; k < typeShapes[i][j].size(); k++) {
                    typeShapes[i][j][k] *= scale;
                }
            }
        }
        for (int i = 0; i < nRows; i++) {
            rowX[i] *= scale;
            rowY[i] *= scale;
            rowXStep[i] *= scale;
            rowHeight[i] *= scale;
        }
        siteWidth *= scale;
        siteHeight *= scale;
        if (format == "dac2012") {
            gridOriginX *= scale;
            gridOriginY *= scale;
            tileW *= scale;
            tileH *= scale;
            for (auto& x : capV) {
                x *= scale;
            }
            for (auto& x : capH) {
                x *= scale;
            }
            for (auto& x : wireWidth) {
                x *= scale;
            }
            for (auto& x : wireSpace) {
                x *= scale;
            }
            for (auto& x : viaSpace) {
                x *= scale;
            }
        }
    }

    void estimateSiteSize() {
        set<int> sizeSet;
        set<int> heightSet;

        for (int i = 0; i < nCells; i++) {
            if (cellFixed[i] == (char)1) {
                continue;
            }
            int type = cellType[i];
            int typeW = typeWidth[type];
            int typeH = typeHeight[type];
            if (sizeSet.count(typeW) == 0) {
                sizeSet.insert(typeW);
            }
            if (sizeSet.count(typeH) == 0) {
                sizeSet.insert(typeH);
            }
            if (heightSet.count(typeH) == 0) {
                heightSet.insert(typeH);
            }
        }

        vector<int> sizes;
        vector<int> heights;
        set<int>::iterator ii = sizeSet.begin();
        set<int>::iterator ie = sizeSet.end();
        for (; ii != ie; ++ii) {
            sizes.push_back(*ii);
        }
        ii = heightSet.begin();
        ie = heightSet.end();
        for (; ii != ie; ++ii) {
            heights.push_back(*ii);
        }
        siteWidth = gcd(sizes);
        siteHeight = gcd(heights);
        logger.info("estimate site size = %d x %d", siteWidth, siteHeight);
        ii = heightSet.begin();
        ie = heightSet.end();
        for (; ii != ie; ++ii) {
            logger.info("standard cell heights: %d rows", (*ii) / siteHeight);
        }
    }
    int gcd(vector<int>& nums) {
        if (nums.size() == 0) {
            return 0;
        }
        if (nums.size() == 1) {
            return nums[0];
        }
        int primes[20] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71};
        int greatestFactor = 1;
        for (bool factorFound = true; factorFound;) {
            factorFound = false;
            for (int f = 0; f < 10; f++) {
                int factor = primes[f];
                bool factorValid = true;
                for (int i = 0; i < (int)nums.size(); i++) {
                    int num = nums[i];
                    // logger.info("%d : %d", i, num);
                    if (num % factor != 0) {
                        factorValid = false;
                        break;
                    }
                }
                if (!factorValid) {
                    continue;
                }
                greatestFactor *= factor;
                for (int i = 0; i < (int)nums.size(); i++) {
                    nums[i] /= factor;
                }
                factorFound = true;
                break;
            }
        }
        return greatestFactor;
    }
};

BookshelfData bsData;

bool isBookshelfSymbol(unsigned char c) {
    static char symbols[256] = {0};
    static bool inited = false;
    if (!inited) {
        symbols[(int)'('] = 1;
        symbols[(int)')'] = 1;
        // symbols[(int)'['] = 1;
        // symbols[(int)']'] = 1;
        symbols[(int)','] = 1;
        // symbols[(int)'.'] = 1;
        symbols[(int)':'] = 1;
        symbols[(int)';'] = 1;
        // symbols[(int)'/'] = 1;
        symbols[(int)'#'] = 1;
        symbols[(int)'{'] = 1;
        symbols[(int)'}'] = 1;
        symbols[(int)'*'] = 1;
        symbols[(int)'\"'] = 1;
        symbols[(int)'\\'] = 1;

        symbols[(int)' '] = 2;
        symbols[(int)'\t'] = 2;
        symbols[(int)'\n'] = 2;
        symbols[(int)'\r'] = 2;
        inited = true;
    }
    return symbols[(int)c] != 0;
}

bool readBSLine(istream& is, vector<string>& tokens) {
    tokens.clear();
    string line;
    while (is && tokens.empty()) {
        // read next line in
        getline(is, line);

        char token[1024] = {0};
        int lineLen = (int)line.size();
        int tokenLen = 0;
        for (int i = 0; i < lineLen; i++) {
            char c = line[i];
            if (c == '#') {
                break;
            }
            if (isBookshelfSymbol(c)) {
                if (tokenLen > 0) {
                    token[tokenLen] = (char)0;
                    tokens.push_back(string(token));
                    token[0] = (char)0;
                    tokenLen = 0;
                }
            } else {
                token[tokenLen++] = c;
                if (tokenLen > 1024) {
                    // TODO: unhandled error
                    tokens.clear();
                    return false;
                }
            }
        }
        // line finished, something else in token
        if (tokenLen > 0) {
            token[tokenLen] = (char)0;
            tokens.push_back(string(token));
            tokenLen = 0;
        }
    }
    return !tokens.empty();
}

void printTokens(vector<string>& tokens) {
    for (auto const& token : tokens) {
        std::cout << token << " : ";
    }
    std::cout << std::endl;
}

bool Database::readBSAux(const std::string& auxFile, const std::string& plFile) {
    std::string directory;
    unsigned found = auxFile.find_last_of("/\\");
    if (found == auxFile.npos) {
        directory = "./";
    } else {
        directory = auxFile.substr(0, found);
        directory += "/";
    }
    logger.info("dir = %s", directory.c_str());

    ifstream fs(auxFile.c_str());
    if (!fs.good()) {
        logger.error("cannot open file: %s", auxFile.c_str());
        return false;
    }

    vector<string> tokens;
    readBSLine(fs, tokens);
    fs.close();

    std::string fileNodes;
    std::string fileNets;
    std::string fileScl;
    std::string fileRoute;
    std::string fileShapes;
    std::string fileWts;
    std::string filePl;

    for (int i = 1; i < (int)tokens.size(); i++) {
        const string& file = tokens[i];
        const size_t dotPos = file.find_last_of(".");
        if (dotPos == string::npos) {
            continue;
        }

        string ext = file.substr(dotPos);
        if (ext == ".nodes") {
            fileNodes = directory + file;
        } else if (ext == ".nets") {
            fileNets = directory + file;
        } else if (ext == ".scl") {
            fileScl = directory + file;
        } else if (ext == ".route") {
            fileRoute = directory + file;
        } else if (ext == ".shapes") {
            fileShapes = directory + file;
        } else if (ext == ".wts") {
            fileWts = directory + file;
        } else if (ext == ".pl") {
            filePl = directory + file;
        } else {
            logger.error("unrecognized file extension: %s", ext.c_str());
        }
    }
    // step 1: read floorplan, rows from:
    //  scl - rows
    // step 2: read cell types from:
    //  nets - pin name, pin offset
    //  nodes - with, height, cell type name
    // step 3: read cell from:
    //  nodes - cell name, type
    // step 4: read placement:
    //  pl - cell fix/movable, position

    bsData.format = setting.BookshelfVariety;
    bsData.nCells = 0;
    bsData.nTypes = 0;
    bsData.nNets = 0;
    bsData.nRows = 0;
    // bsData.siteWidth = 240;
    // bsData.siteHeight = 2880;

    readBSNodes(fileNodes);

    bsData.estimateSiteSize();

    if (bsData.format == "dac2012") {
        readBSNets(fileNets);
        readBSRoute(fileRoute);
        readBSShapes(fileShapes);
        readBSWts(fileWts);
        // readBSPl    ( filePl    );
        readBSPl(plFile);
        readBSScl(fileScl);
    } else if (bsData.format == "ispd2005") {
        readBSNets(fileNets);
        // readBSRoute ( fileRoute );
        // readBSShapes( fileShapes);
        readBSWts(fileWts);
        // readBSPl    ( filePl    );
        readBSPl(plFile);
        readBSScl(fileScl);
    }

    if (bsData.siteWidth < 10) {
        bsData.scaleData(100);
        // this->scale = 100;
    } else if (bsData.siteWidth < 100) {
        bsData.scaleData(10);
        // this->scale = 10;
    } else {
        // this->scale = 1;
    }

    logger.info("parsing rows");
    this->LefConvertFactor = 1;  // suppose 1 in bookshelf
    this->dieLX = INT_MAX;
    this->dieLY = INT_MAX;
    this->dieHX = INT_MIN;
    this->dieHY = INT_MIN;
    for (int i = 0; i < bsData.nRows; i++) {
        Row* row = this->addRow("core_SITE_ROW_" + to_string(i), "core", bsData.rowX[i], bsData.rowY[i]);
        row->xStep(bsData.siteWidth);
        row->yStep(bsData.siteHeight);
        row->xNum(bsData.rowSites[i]);
        row->yNum(1);
        row->flip((i % 2) == 1);
        this->dieLX = std::min(this->dieLX, row->x());
        this->dieLY = std::min(this->dieLY, row->y());
        this->dieHX = std::max(this->dieHX, row->x() + (int)row->width());
        this->dieHY = std::max(this->dieHY, row->y() + bsData.siteHeight);
        // make sure the parsed results are the same as the estimated results
        assert_msg(bsData.rowXStep[i] == bsData.siteWidth,
                   "Row %s rowXStep (%d) is not equal to siteWidth (%d)",
                   row->name().c_str(),
                   bsData.rowXStep[i],
                   bsData.siteWidth);
        assert_msg(bsData.rowHeight[i] == bsData.siteHeight,
                   "Row %s rowYStep (%d) is not equal to siteHeight (%d)",
                   row->name().c_str(),
                   bsData.rowHeight[i],
                   bsData.siteHeight);
    }

    // parsing gcellgrid
    unsigned nLayers = 9;
    bool isDac2012 = false;
    if (bsData.format == "dac2012") {
        isDac2012 = true;
        nLayers = bsData.gridNZ;
        this->gcellgrid.numX = {bsData.gridNX + 1};
        this->gcellgrid.numY = {bsData.gridNY + 1};
        this->gcellgrid.startX = {bsData.gridOriginX};
        this->gcellgrid.startY = {bsData.gridOriginY};
        this->gcellgrid.stepX = {bsData.tileW};
        this->gcellgrid.stepY = {bsData.tileH};
        this->bsRouteInfo.hasInfo = true;
        this->bsRouteInfo.capV = std::move(bsData.capV);
        this->bsRouteInfo.capH = std::move(bsData.capH);
        this->bsRouteInfo.viaSpace = std::move(bsData.viaSpace);
        this->bsRouteInfo.blockagePorosity = bsData.blockagePorosity;
        // this->grGrid.gcellNX = bsData.gridNX;
        // this->grGrid.gcellNY = bsData.gridNY;
        // this->grGrid.gcellL = bsData.gridOriginX;
        // this->grGrid.gcellB = bsData.gridOriginY;
        // this->grGrid.gcellStepX = bsData.tileW;
        // this->grGrid.gcellStepY = bsData.tileH;
        // this->grGrid.capV = std::move(bsData.capV);
        // this->grGrid.capH = std::move(bsData.capH);
        // this->grGrid.minWireWidth = std::move(bsData.wireWidth);
        // this->grGrid.minWireSpacing = std::move(bsData.wireSpace);
        // this->grGrid.viaSpacing = std::move(bsData.viaSpace);
        // this->grGrid.minPitch.resize(nLayers);
        // for (int i = 0; i < nLayers; i++) {
        //     this->grGrid.minPitch[i] = this->grGrid.minWireWidth[i] + this->grGrid.minWireSpacing[i];
        // }
    }

    // NOTE: ICCAD/DAC 2012 does not define trackPitch and the capacity of each GCell
    // cannot be directly computed by tracks. We restore the bookshelf capacity value
    // in Database::Others::capV(H) and will handle them in GRDatabase.
    logger.info("parsing layers");
    int defaultPitch = bsData.siteWidth;
    int defaultWidth = bsData.siteWidth / 2;
    int defaultSpace = bsData.siteWidth - defaultWidth;
    char m1direction = 'h';  // M1 Route layer, rIndex == 0
    char m2direction = 'v';  // M2 Route layer, rIndex == 1
    if ((nLayers > 1) && (bsData.format == "dac2012")) {
        m1direction = (this->bsRouteInfo.capV[1] > 0) ? 'h' : 'v';
        m2direction = (this->bsRouteInfo.capV[1] > 0) ? 'v' : 'h';
    }
    for (unsigned i = 0; i != nLayers; ++i) {
        Layer& layer = this->addLayer(string("M").append(to_string(i + 1)), 'r');
        // if (!i) {
        //     layer.direction = 'x';
        //     layer.track.direction = 'x';
        // } else if (i % 2) {
        if (i % 2) {
            layer.direction = m2direction;
            layer.track.direction = m2direction;
        } else {
            layer.direction = m1direction;
            layer.track.direction = m1direction;
        }
        if (isDac2012) {
            layer.width = bsData.wireWidth[i];
            layer.spacing = bsData.wireSpace[i];
            layer.pitch = layer.width + layer.spacing;
            layer.offset = layer.spacing;
        } else {
            layer.pitch = defaultPitch;
            layer.offset = defaultSpace;
            layer.width = defaultWidth;
            layer.spacing = defaultSpace;
        }
        // if (i % 2) {
        //     layer.track.start = this->dieLX + (layer.pitch / 2);
        //     layer.track.num = (this->dieHX - this->dieLX) / layer.pitch;
        // } else {
        //     layer.track.start = this->dieLY + (layer.pitch / 2);
        //     layer.track.num = (this->dieHY - this->dieLY) / layer.pitch;
        // }
        // layer.track.step = layer.pitch;

        if (i + 1 == nLayers) {
            break;
        } else {
            this->addLayer(string("N").append(to_string(i + 1)), 'c');
        }
    }

    logger.info("parsing celltype");
    const Layer& layer = this->layers[0];
    for (int i = 0; i < bsData.nTypes; i++) {
        CellType* celltype = this->addCellType(bsData.typeName[i], this->celltypes.size());
        celltype->width = bsData.typeWidth[i];
        celltype->height = bsData.typeHeight[i];
        celltype->stdcell = (!bsData.typeFixed[i]);
        for (int j = 0; j < bsData.typeNPins[i]; ++j) {
            char direction = 'x';
            switch (bsData.typePinDir[i][j]) {
                case 'I':
                    direction = 'i';
                    break;
                case 'O':
                    direction = 'o';
                    break;
                default:
                    logger.error("unknown pin direction: %c", bsData.typePinDir[i][j]);
                    break;
            }
            PinType* pintype = celltype->addPin(bsData.typePinName[i][j], direction, 's');
            pintype->addShape(layer, bsData.typePinX[i][j], bsData.typePinY[i][j]);
        }
        for (auto& curShape : bsData.typeShapes[i]) {
            int xl = curShape[0];
            int yl = curShape[1];
            int w = curShape[2];
            int h = curShape[3];
            celltype->addNonRegularRects(xl, yl, xl + w, yl + h);
            this->placeBlockages.emplace_back(xl, yl, xl + w, yl + h);
        }
    }

    logger.info("parsing cells");
    for (int i = 0; i < bsData.nCells; i++) {
        int typeID = bsData.cellType[i];
        if (typeID < 0) {
            Layer& layer = *this->getRLayer(bsData.IOPinRouteLayer[i]);
            IOPin* iopin = this->addIOPin(bsData.cellName[i]);
            switch (typeID) {
                case -1:
                    iopin->type->direction('o');
                    break;
                case -2:
                    iopin->type->direction('i');
                    break;
                default:
                    iopin->type->direction('x');
                    break;
            }
            iopin->type->addShape(layer, bsData.cellX[i], bsData.cellY[i]);
        } else {
            string celltypename(bsData.typeName[typeID]);
            Cell* cell = this->addCell(bsData.cellName[i], this->getCellType(celltypename));
            cell->place(bsData.cellX[i], bsData.cellY[i], false, false);
            // cout<<cell->x<<","<<cell->y<<endl;
            cell->fixed((bsData.cellFixed[i] == (char)1));
        }
    }

    logger.info("parsing nets");
    for (unsigned i = 0; i != bsData.nNets; ++i) {
        Net* net = this->addNet(bsData.netName[i]);
        for (unsigned j = 0; j != bsData.netCells[i].size(); ++j) {
            Pin* pin = nullptr;
            int cellID = bsData.netCells[i][j];
            if (bsData.cellType[cellID] < 0) {
                IOPin* iopin = this->getIOPin(bsData.cellName[cellID]);
                pin = iopin->pin;
                if (pin->is_connected) {
                    string netName(net->name);
                    logger.warning("IO Pin is re-connected: %s %s", netName.c_str(), bsData.cellName[cellID].c_str());
                }
                iopin->is_connected = true;
            } else {
                Cell* cell = this->getCell(bsData.cellName[cellID]);
                pin = cell->pin(bsData.netPins[i][j]);
                if (pin->is_connected) {
                    string netName(net->name);
                    logger.warning("Pin is re-connected: %s %s %d",
                                   netName.c_str(),
                                   bsData.cellName[cellID].c_str(),
                                   bsData.netPins[i][j]);
                }
                cell->is_connected = true;
            }
            pin->net = net;
            pin->is_connected = true;
            net->addPin(pin);
        }
    }

    // parsing routing blockages
    for (auto& [i, blkg] : bsData.routeBlkgs) {
        int x = bsData.cellX[i];
        int y = bsData.cellY[i];
        int typeID = bsData.cellType[i];
        int w = bsData.typeWidth[typeID];
        int h = bsData.typeHeight[typeID];
        auto& shapes = bsData.typeShapes[typeID];
        for (int layerId : blkg) {
            Layer& layer = *this->getRLayer(layerId);
            if (shapes.size() == 0) {
                this->routeBlockages.emplace_back(layer, x, y, x + w, y + h);
            } else {
                for (auto& curShape : shapes) {
                    int sxl = curShape[0];
                    int syl = curShape[1];
                    int sw = curShape[2];
                    int sh = curShape[3];
                    this->routeBlockages.emplace_back(layer, sxl, syl, sxl + sw, syl + sh);
                }
            }
        }
    }

    bsData.clearData();
    logger.info("finish reading bookshelf.");

    return true;
}
bool Database::readBSNodes(const std::string& file) {
    logger.info("reading nodes");
    ifstream fs(file.c_str());
    if (!fs.good()) {
        logger.error("cannot open file: %s", file.c_str());
        return false;
    }
    // int nNodes = 0;
    // int nTerminals = 0;
    vector<string> tokens;
    while (readBSLine(fs, tokens)) {
        // logger.info("%d : %s", i++, tokens[0].c_str());
        if (tokens[0] == "UCLA") {
            continue;
        } else if (tokens[0] == "NumNodes") {
            // nNodes = atoi(tokens[1].c_str());
        } else if (tokens[0] == "NumTerminals") {
            // nTerminals = atoi(tokens[1].c_str());
        } else if (tokens.size() >= 3) {
            string cName = tokens[0];
            int cWidth = atoi(tokens[1].c_str());
            int cHeight = atoi(tokens[2].c_str());
            bool cFixed = false;
            string cType = cName;
            if (tokens.size() >= 4) {
                cType = tokens[3];
                //  cout << cName << '\t' << cType << endl;
            }
            if (cType == "terminal" && cWidth > 1 && cHeight > 1) {
                // logger.info("read terminal");
                cType = cName;
                cFixed = true;
            }
            if (cType == "terminal_NI") {
                cType = cName;
                cFixed = true;
            }
            int typeID = -1;
            if (cType == "terminal") {
                // logger.info("read terminal");
                typeID = -1;
                cFixed = true;
            } else if (bsData.typeMap.find(cType) == bsData.typeMap.end()) {
                typeID = bsData.nTypes++;
                bsData.typeMap[cType] = typeID;
                bsData.typeName.push_back(cType);
                bsData.typeWidth.push_back(cWidth);
                bsData.typeHeight.push_back(cHeight);
                bsData.typeFixed.push_back((char)(cFixed ? 1 : 0));
                bsData.typeNPins.push_back(0);
                bsData.typePinName.push_back(vector<string>());
                bsData.typePinDir.push_back(vector<char>());
                bsData.typePinX.push_back(vector<double>());
                bsData.typePinY.push_back(vector<double>());
                bsData.typeShapes.push_back(vector<vector<int>>());
                //  cout << cType << '\t' << tokens.size() << endl;
            } else {
                typeID = bsData.typeMap[cType];
                assert(cWidth == bsData.typeWidth[typeID]);
                assert(cHeight == bsData.typeHeight[typeID]);
            }
            int cellID = bsData.nCells++;
            bsData.cellMap[cName] = cellID;
            bsData.cellType.push_back(typeID);
            bsData.cellFixed.push_back((char)(cFixed ? 1 : 0));
            bsData.cellName.push_back(cName);
            bsData.cellX.push_back(0);
            bsData.cellY.push_back(0);
            bsData.IOPinRouteLayer.push_back(0);
        }
    }

    fs.close();
    return true;
}

bool Database::readBSNets(const std::string& file) {
    logger.info("reading net");
    ifstream fs(file.c_str());
    if (!fs.good()) {
        logger.error("cannot open file: %s", file.c_str());
        return false;
    }
    vector<string> tokens;
    while (readBSLine(fs, tokens)) {
        if (tokens[0] == "UCLA") {
            continue;
        } else if (tokens[0] == "NumNets") {
            // logger.info("#nets : %d", atoi(tokens[1].c_str()));
            int numNets = atoi(tokens[1].c_str());
            bsData.netName.resize(numNets);
            bsData.netCells.resize(numNets);
            bsData.netPins.resize(numNets);
        } else if (tokens[0] == "NumPins") {
            // logger.info("#pins : %d", atoi(tokens[1].c_str()));
        } else if (tokens[0] == "NetDegree") {
            int degree = atoi(tokens[1].c_str());
            string nName = tokens[2];
            int netID = bsData.nNets++;
            bsData.netName[netID] = nName;
            for (int i = 0; i < degree; i++) {
                readBSLine(fs, tokens);
                string cName = tokens[0];
                if (bsData.cellMap.find(cName) == bsData.cellMap.end()) {
                    assert(false);
                    logger.error("cell not found : %s", cName.c_str());
                    return false;
                }

                int cellID = bsData.cellMap[cName];
                int typeID = bsData.cellType[cellID];
                int typePinID = -1;

                char dir = tokens[1].c_str()[0];
                double pinX = 0;
                double pinY = 0;
                pinX = bsData.typeWidth[typeID] * 0.5 + (double)atof(tokens[2].c_str());
                pinY = bsData.typeHeight[typeID] * 0.5 + (double)atof(tokens[3].c_str());
                /*
                if(tokens.size() >= 6){
                    //pinX = (int)round(atof(tokens[4].c_str())*bsData.siteWidth);
                    //pinY = (int)round(atof(tokens[5].c_str())*bsData.siteWidth);
                }else{
                    pinX = bsData.typeWidth[typeID]  * 0.5 + (int)round(atof(tokens[2].c_str()));
                    pinY = bsData.typeHeight[typeID] * 0.5 + (int)round(atof(tokens[3].c_str()));
                }
                */
                string pinName = (tokens.size() < 7) ? "" : tokens[6];

                string tpName;
                if (pinName == "" && typeID >= 0) {
                    stringstream ss;
                    ss << bsData.typeNPins[typeID];
                    pinName = ss.str();
                    // logger.info("pinname = %s", pinName.c_str());
                }
                if (typeID >= 0) {
                    tpName.append(bsData.typeName[typeID]);
                    tpName.append(":");
                    tpName.append(pinName);
                }
                if (typeID == -1) {
                    // IOPin
                    if (dir == 'I') {
                        typeID = -1;
                    } else if (dir == 'O') {
                        typeID = -2;
                    } else {
                        typeID = -3;
                    }
                } else if (bsData.typePinMap.find(tpName) == bsData.typePinMap.end()) {
                    typePinID = bsData.typeNPins[typeID]++;
                    bsData.typePinMap[tpName] = typePinID;
                    bsData.typePinName[typeID].push_back(pinName);
                    bsData.typePinDir[typeID].push_back(dir);
                    bsData.typePinX[typeID].push_back(pinX);
                    bsData.typePinY[typeID].push_back(pinY);
                } else {
                    typePinID = bsData.typePinMap[tpName];
                    assert(bsData.typePinX[typeID][typePinID] == pinX);
                    assert(bsData.typePinY[typeID][typePinID] == pinY);
                    assert(bsData.typePinDir[typeID][typePinID] == dir);
                }
                bsData.netCells[netID].push_back(cellID);
                bsData.netPins[netID].push_back(typePinID);
            }
        }
    }

    fs.close();
    return true;
}

bool Database::readBSScl(const std::string& file) {
    logger.info("reading scl");
    ifstream fs(file.c_str());
    if (!fs.good()) {
        logger.error("cannot open file: %s", file.c_str());
        return false;
    }
    vector<string> tokens;
    // char direction = 'x';
    while (readBSLine(fs, tokens)) {
        if (tokens[0] == "UCLA") {
            continue;
        } else if (tokens[0] == "NumRows") {
            int numRows = atoi(tokens[1].c_str());
            bsData.rowX.resize(numRows);
            bsData.rowY.resize(numRows);
            bsData.rowSites.resize(numRows);
            bsData.rowXStep.resize(numRows);
            bsData.rowHeight.resize(numRows);
        } else if (tokens[0] == "CoreRow") {
            // if (tokens[1] == "Horizontal") {
            //     direction = 'h';
            // } else if (tokens[1] == "Vertical") {
            //     direction = 'v';
            // } else {
            //     direction = 'x';
            // }
        } else if (tokens[0] == "Coordinate") {
            bsData.rowY[bsData.nRows] = atoi(tokens[1].c_str());
        } else if (tokens[0] == "SubrowOrigin" && tokens[2] == "NumSites") {
            bsData.rowX[bsData.nRows] = atoi(tokens[1].c_str());
            bsData.rowSites[bsData.nRows] = atoi(tokens[3].c_str());
        } else if (tokens[0] == "Sitewidth") {
            bsData.rowXStep[bsData.nRows] = atoi(tokens[1].c_str());
        } else if (tokens[0] == "Height") {
            bsData.rowHeight[bsData.nRows] = atoi(tokens[1].c_str());
        } else if (tokens[0] == "End") {
            bsData.nRows++;
        }
    }
    fs.close();
    return true;
}

bool Database::readBSRoute(const std::string& file) {
    logger.info("reading route");
    ifstream fs(file.c_str());
    if (!fs.good()) {
        logger.error("cannot open file: %s", file.c_str());
        return false;
    }
    vector<string> tokens;
    const unsigned ReadingHeader = 0;
    const unsigned ReadingPinLayer = 1;
    const unsigned ReadingBlockages = 2;
    const unsigned ReadingCapAdjust = 3;
    unsigned status = ReadingHeader;
    while (readBSLine(fs, tokens)) {
        if (tokens[0] == "Grid") {
            bsData.gridNX = atoi(tokens[1].c_str());
            bsData.gridNY = atoi(tokens[2].c_str());
            bsData.gridNZ = atoi(tokens[3].c_str());
        } else if (tokens[0] == "VerticalCapacity") {
            for (unsigned i = 1; i < tokens.size(); i++) {
                bsData.capV.push_back(atoi(tokens[i].c_str()));
            }
        } else if (tokens[0] == "HorizontalCapacity") {
            for (unsigned i = 1; i < tokens.size(); i++) {
                bsData.capH.push_back(atoi(tokens[i].c_str()));
            }
        } else if (tokens[0] == "MinWireWidth") {
            for (unsigned i = 1; i < tokens.size(); i++) {
                bsData.wireWidth.push_back(atoi(tokens[i].c_str()));
            }
        } else if (tokens[0] == "MinWireSpacing") {
            for (unsigned i = 1; i < tokens.size(); i++) {
                bsData.wireSpace.push_back(atoi(tokens[i].c_str()));
            }
        } else if (tokens[0] == "ViaSpacing") {
            for (unsigned i = 1; i < tokens.size(); i++) {
                bsData.viaSpace.push_back(atoi(tokens[i].c_str()));
            }
        } else if (tokens[0] == "GridOrigin") {
            bsData.gridOriginX = atoi(tokens[1].c_str());
            bsData.gridOriginY = atoi(tokens[2].c_str());
        } else if (tokens[0] == "TileSize") {
            bsData.tileW = atoi(tokens[1].c_str());
            bsData.tileH = atoi(tokens[2].c_str());
        } else if (tokens[0] == "BlockagePorosity") {
            bsData.blockagePorosity = atof(tokens[1].c_str());
            // assert_msg(bsData.blockagePorosity == 0, "We haven't yet supported non-zero blockagePorosity");
        } else if (tokens[0] == "NumNiTerminals") {
            status = ReadingPinLayer;
        } else if (tokens[0] == "NumBlockageNodes") {
            status = ReadingBlockages;
        } else if (tokens[0] == "NumEdgeCapacityAdjustments") {
            status = ReadingCapAdjust;
        } else if (status == ReadingPinLayer) {
            std::string cName = tokens[0];
            if (bsData.cellMap.find(cName) == bsData.cellMap.end()) {
                logger.error("pin not found : %s", cName.c_str());
                getchar();
            }
            int cellID = bsData.cellMap[cName];
            bsData.IOPinRouteLayer[cellID] = atoi(tokens[1].c_str()) - 1;
            //  cout<<pin<<" "<<layer<<endl;
        } else if (status == ReadingBlockages) {
            std::string cName = tokens[0];
            if (bsData.cellMap.find(cName) == bsData.cellMap.end()) {
                logger.error("cell not found : %s", cName.c_str());
                getchar();
            }
            int cellID = bsData.cellMap[cName];
            // int numLayers = atoi(tokens[1].c_str());
            vector<int> blockedLayers;
            for (unsigned i = 2; i < tokens.size(); i++) {
                int layerId = atoi(tokens[i].c_str()) - 1;
                blockedLayers.emplace_back(layerId);
            }
            auto curBlkgPair = std::make_pair(cellID, std::move(blockedLayers));
            bsData.routeBlkgs.emplace_back(curBlkgPair);
            //  cout<<node<<" "<<maxLayer<<endl;
        } else if (status == ReadingCapAdjust) {
            /*
            int fx = atoi(tokens[0].c_str());
            int fy = atoi(tokens[1].c_str());
            int fz = atoi(tokens[2].c_str());
            int tx = atoi(tokens[3].c_str());
            int ty = atoi(tokens[4].c_str());
            int tz = atoi(tokens[5].c_str());
            int cap = atoi(tokens[6].c_str());
            cout<<fx<<" "<<fy<<" "<<fz<<" "<<tx<<" "<<ty<<" "<<tz<<" "<<cap<<endl;
            */
        }
    }
    fs.close();
    return true;
}

bool Database::readBSShapes(const std::string& file) {
    logger.info("reading shapes");
    ifstream fs(file.c_str());
    if (!fs.good()) {
        logger.error("cannot open file: %s", file.c_str());
        return false;
    }
    vector<string> tokens;
    // char direction = 'x';
    while (readBSLine(fs, tokens)) {
        if (tokens[0] == "shapes") {
            continue;
        } else if (tokens[0] == "NumNonRectangularNodes") {
            int numNonRectangularNodes = atoi(tokens[1].c_str());
        } else if (tokens.size() == 2) {
            string cName = tokens[0];
            int cellID = bsData.cellMap[cName];
            int typeID = bsData.cellType[cellID];
            int numShapes = atoi(tokens[1].c_str());
            for (int i = 0; i < numShapes; i++) {
                readBSLine(fs, tokens);
                string shapeId = tokens[0];
                int xl = atoi(tokens[1].c_str());
                int yl = atoi(tokens[2].c_str());
                int w = atoi(tokens[3].c_str());
                int h = atoi(tokens[4].c_str());
                bsData.typeShapes[typeID].push_back({xl, yl, w, h});
            }
        }
    }

    fs.close();
    return true;
}

bool Database::readBSWts(const std::string& file) {
    logger.info("reading weights");
    ifstream fs(file.c_str());
    if (!fs.good()) {
        logger.error("cannot open file: %s", file.c_str());
        return false;
    }
    vector<string> tokens;
    fs.close();
    return true;
}

bool Database::readBSPl(const std::string& file) {
    logger.info("reading placement");
    ifstream fs(file.c_str());
    if (!fs.good()) {
        logger.error("cannot open file: %s", file.c_str());
        return false;
    }
    vector<string> tokens;
    while (readBSLine(fs, tokens)) {
        if (tokens[0] == "UCLA") {
            continue;
        } else if (tokens.size() >= 4) {
            robin_hood::unordered_map<string, int>::iterator itr = bsData.cellMap.find(tokens[0]);
            if (itr == bsData.cellMap.end()) {
                assert(false);
                logger.error("cell not found: %s", tokens[0].c_str());
                return false;
            }
            int cell = itr->second;
            double x = atof(tokens[1].c_str());
            double y = atof(tokens[2].c_str());
            bsData.cellX[cell] = (int)round(x);
            bsData.cellY[cell] = (int)round(y);

            if (tokens.size() >= 5) {
                if (tokens[4] == "/FIXED" || tokens[4] == "/FIXED_NI") {
                    bsData.cellFixed[cell] = (char)1;
                }
            }
        }
    }
    fs.close();
    return true;
}

bool Database::writeBSPl(const std::string& file) {
    ofstream fs(file.c_str());
    if (!fs.good()) {
        logger.error("cannot open file: %s", file.c_str());
        return false;
    }
    fs << "UCLA pl 1.0\n";
    fs << "# User   : Chinese University of Hong Kong\n\n";
    for (auto cell : this->cells) {
        fs << '\t' << left << setw(60) << cell->name() << right << setw(8) << cell->lx() / siteW << setw(8)
           << cell->ly() / siteW << " : N";
        if (cell->fixed()) {
            if (cell->width() / siteW == 1 && cell->height() / siteW == 1) {
                fs << " /FIXED_NI";
            } else {
                fs << " /FIXED";
            }
        }
        fs << endl;
    }
    fs.close();
    return true;
}
