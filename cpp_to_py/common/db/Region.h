#pragma once

namespace db {
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

    inline const string& name() const { return _name; }
    inline char type() const { return _type; }

    void addRect(const int xl, const int yl, const int xh, const int yh);
    void resetRects() { sliceH(rects); }

};
}  //   namespace db

