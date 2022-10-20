#pragma once

namespace db {
class Rectangle {
public:
    int lx = INT_MAX;
    int ly = INT_MAX;
    int hx = INT_MIN;
    int hy = INT_MIN;

    Rectangle(int lx = INT_MAX, int ly = INT_MAX, int hx = INT_MIN, int hy = INT_MIN)
        : lx(lx), ly(ly), hx(hx), hy(hy) {}

    int w() const { return hx - lx; }
    int h() const { return hy - ly; }
    int cx() const { return (hx + lx) / 2; }
    int cy() const { return (hy + ly) / 2; }
    bool operator<(const Rectangle& geo) const { return (ly == geo.ly) ? (lx < geo.lx) : (ly < geo.ly); }
    bool operator==(const Rectangle& geo) const { return lx == geo.lx && ly == geo.ly && hx == geo.hx && hy == geo.hy; }
    Rectangle& operator+=(const Rectangle& geo);
    static bool CompareXInc(const Rectangle& a, const Rectangle& b) {
        return (a.lx == b.lx) ? (a.hx < b.hx) : (a.lx < b.lx);
    }
    static bool CompareXDec(const Rectangle& a, const Rectangle& b) {
        return (a.hx == b.hx) ? (a.lx > b.lx) : (a.hx > b.hx);
    }
    static bool CompareYInc(const Rectangle& a, const Rectangle& b) {
        return (a.ly == b.ly) ? (a.hy < b.hy) : (a.ly < b.ly);
    }
    static bool CompareYDec(const Rectangle& a, const Rectangle& b) {
        return (a.hy == b.hy) ? (a.ly > b.ly) : (a.hy > b.hy);
    }
    static bool IsInvalid(const Rectangle& geo) { return (geo.lx >= geo.hx) || (geo.ly >= geo.hy); }
    static void sliceH(vector<Rectangle>& rects);
    static void sliceV(vector<Rectangle>& rects);

    friend ostream& operator<<(ostream& os, const Rectangle& r) {
        return os << '(' << r.lx << ", " << r.ly << ")\t(" << r.hx << ", " << r.hy << ')';
    }
};

class Geometry : public Rectangle {
public:
    const Layer& layer;

    Geometry(const Layer& layer, const int lx, const int ly, const int hx, const int hy)
        : Rectangle(lx, ly, hx, hy), layer(layer) {}
    Geometry(const Geometry& geom) : Rectangle(geom), layer(geom.layer) {}

    inline bool operator==(const Geometry& rhs) const;
};

class GeoMap : public Rectangle {
private:
    map<int, Geometry> _map;

public:
    bool empty() const noexcept { return _map.empty(); }
    size_t size() const noexcept { return _map.size(); }
    const Geometry& front() const { return _map.begin()->second; }
    const Geometry& front2() const { return (++_map.begin())->second; }
    const Geometry& back() const { return _map.rbegin()->second; }
    map<int, Geometry>::const_iterator begin() const noexcept { return _map.begin(); }
    map<int, Geometry>::const_iterator end() const noexcept { return _map.end(); }
    map<int, Geometry>::const_iterator find(const int k) const { return _map.find(k); }

    const Geometry& at(const int k) const { return _map.at(k); }

    void emplace(const int k, const Geometry& shape);
};

}  // namespace db
