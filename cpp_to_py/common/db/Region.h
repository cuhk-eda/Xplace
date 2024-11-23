#pragma once
#include <string>
#include <vector>

#include "Geometry.h"

namespace db {

using std::string;
using std::vector;

class Region : public Rectangle {
private:
    string _name = "";
    //  'f' for fence, 'g' for guide
    char _type = 'x';

public:
    static const unsigned char InvalidRegion = 0xff;

    unsigned char id = InvalidRegion;
    double density = 0;

    vector<string> members;
    vector<Rectangle> rects;

    Region(const string& name = "", const char type = 'x')
        : Rectangle(INT_MAX, INT_MAX, INT_MIN, INT_MIN), _name(name), _type(type) {}

    const string& name() const { return _name; }
    char type() const { return _type; }

    void addRect(const int xl, const int yl, const int xh, const int yh);
    void resetRects() { sliceH(rects); }
};
}  //   namespace db
