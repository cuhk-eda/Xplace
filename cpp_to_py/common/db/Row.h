#pragma once

namespace db {
class Row {
    friend class Database;

private:
    string _name = "";
    string _macro = "";
    int _x = 0;
    int _y = 0;
    unsigned _xNum = 0;
    unsigned _yNum = 0;
    bool _flip = false;
    unsigned _xStep = 0;
    unsigned _yStep = 0;
    char _topPower = 'x';
    char _botPower = 'x';

public:
    std::vector<RowSegment> segments;

    Row(const string& name, const string& macro, const int x, const int y, const unsigned xNum = 0, const unsigned yNum = 0, const bool flip = false, const unsigned xStep = 0, const unsigned yStep = 0)
        : _name(name)
        , _macro(macro)
        , _x(x)
        , _y(y)
        , _xNum(xNum)
        , _yNum(yNum)
        , _flip(flip)
        , _xStep(xStep)
        , _yStep(yStep) {}

    //  return the left-most site x-index
    int getSiteL(int dbLX, int siteW) const { return (_x - dbLX) / siteW; }
    //  return the right-most site x-index + 1
    int getSiteR(int dbLX, int siteW) const { return getSiteL(dbLX, siteW) + _xNum; }
    //  return the bottom-most site y-index
    int getSiteB(int dbLY, int siteH) const { return (_y - dbLY) / siteH; }
    //  return the top-most site y-index + 1
    int getSiteT(int dbLY, int siteH) const { return getSiteB(dbLY, siteH) + _yNum; }

    const string& name() const { return _name; }
    const string& macro() const { return _macro; }
    int x() const { return _x; }
    int y() const { return _y; }
    unsigned xNum() const { return _xNum; }
    unsigned yNum() const { return _yNum; }
    bool flip() const { return _flip; }
    unsigned xStep() const { return _xStep; }
    unsigned yStep() const { return _yStep; }
    char topPower() const { return _topPower; }
    char botPower() const { return _botPower; }

    void x(const int value) { _x = value; }
    void y(const int value) { _y = value; }
    void xNum(const unsigned value) { _xNum = value; }
    void yNum(const unsigned value) { _yNum = value; }
    void flip(const bool value) { _flip = value; }
    void xStep(const unsigned value) { _xStep = value; }
    void yStep(const unsigned value) { _yStep = value; }

    unsigned width() const { return _xStep * _xNum; }
    bool isPowerValid() const { return (_topPower != _botPower) && (_topPower != 'x') && (_botPower != 'x'); }

    void shiftX(const int value) { _x += value; }
    void shiftY(const int value) { _y += value; }
    void shrinkXNum(const int value) { _xNum -= value; }
    void shrinkYNum(const int value) { _yNum -= value; }

};

class RowSegment {
public:
    int x = 0;
    int w = 0;
    Region* region = nullptr;
};
} // namespace db

