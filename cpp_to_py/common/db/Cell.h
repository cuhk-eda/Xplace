#pragma once

namespace db {
class CellType {
    friend class Database;

private:
    int _originX = 0;
    int _originY = 0;
    //  X=1  Y=2  R90=4  (can be combined)
    char _symmetry = 0;
    string _siteName = "";
    char _botPower = 'x';
    char _topPower = 'x';
    vector<Geometry> _obs;

    // _nonRegularRects.size() > 0 implies that this cell is a fixed cell and 
    // its shape is a polygon. Each rectangle is also appended into 
    // Databased.placeBlockages during parsing the polygon-shape cell.
    // NOTE: We only support this feature for ICCAD/DAC 2012 benchmarks.
    //     When processing GPDatabase, we will set this kind of cells' width and 
    //     height as 0 and use placeement blockages to represent their shapes.
    vector<Rectangle> _nonRegularRects; 

    int _libcell = -1;

public:
    std::string name = "";
    char cls = 'x';
    bool stdcell = false;
    int width = 0;
    int height = 0;
    vector<PinType*> pins;
    int edgetypeL = 0;
    int edgetypeR = 0;
    int usedCount = 0;

    CellType(const string& name, int libcell) : _libcell(libcell), name(name) {}
    ~CellType();

    PinType* addPin(const string& name, const char direction, const char type);
    void addPin(PinType& pintype);

    template <class... Args>
    void addObs(Args&&... args) {
        _obs.emplace_back(args...);
    }
    template <class... Args>
    void addNonRegularRects(Args&&... args) {
        _nonRegularRects.emplace_back(args...);
    }

    PinType* getPin(string& name);

    int originX() const { return _originX; }
    int originY() const { return _originY; }
    char symmetry() const { return _symmetry; }
    char botPower() const { return _botPower; }
    char topPower() const { return _topPower; }
    const std::vector<Geometry>& obs() const { return _obs; }
    const std::vector<Rectangle>& nonRegularRects() const { return _nonRegularRects; }
    int libcell() const { return _libcell; }

    void setOrigin(int x, int y);
    void setXSymmetry() { _symmetry &= 1; }
    void setYSymmetry() { _symmetry &= 2; }
    void set90Symmetry() { _symmetry &= 4; }
    void siteName(const std::string& name) { _siteName = name; }

    bool operator==(const CellType& r) const;
    bool operator!=(const CellType& r) const { return !(*this == r); }
};

class Cell {
private:
    string _name = "";
    int _spaceL = 0;
    int _spaceR = 0;
    int _spaceB = 0;
    int _spaceT = 0;
    bool _fixed = false;
    CellType* _type = nullptr;
    std::vector<Pin*> _pins;

    int _lx = INT_MIN;
    int _ly = INT_MIN;
    bool _flipX = false;
    bool _flipY = false;

public:
    bool highlighted = false;
    Region* region = nullptr;
    int gpdb_id = -1;
    bool is_connected = false;

    Cell(const string& name = "", CellType* t = nullptr) : _name(name) { ctype(t); }
    ~Cell();

    const std::string& name() const { return _name; }
    Pin* pin(const std::string& name) const;
    Pin* pin(unsigned i) const { return _pins[i]; }
    CellType* ctype() const { return _type; }
    void ctype(CellType* t);
    int lx() const;
    int ly() const;
    int hx() const { return lx() + width(); }
    int hy() const { return ly() + height(); }
    int cx() const { return lx() + width() / 2; }
    int cy() const { return ly() + height() / 2; }
    bool flipX() const;
    bool flipY() const;
    int orient() const;
    int width() const { return _type->width + _spaceL + _spaceR; }
    int height() const { return _type->height + _spaceB + _spaceT; }
    // int siteWidth() const;
    // int siteHeight() const;
    bool fixed() const { return _fixed; }
    void fixed(bool fix) { _fixed = fix; }
    bool placed() const;
    void place(int x, int y);
    void place(int x, int y, int orient);
    void place(int x, int y, bool flipX, bool flipY);
    void unplace();
    unsigned numPins() const { return _pins.size(); }

    friend ostream& operator<<(ostream& os, const Cell& c) {
        return os << c._name << "\t(" << c.lx() << ", " << c.ly() << ')';
    }
};
}  // namespace db