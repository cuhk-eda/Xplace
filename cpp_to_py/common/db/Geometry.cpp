#include "Database.h"
using namespace db;

/***** Rectangle *****/

Rectangle& Rectangle::operator+=(const Rectangle& geo) {
    lx = min(lx, geo.lx);
    ly = min(ly, geo.ly);
    hx = max(hx, geo.hx);
    hy = max(hy, geo.hy);
    return *this;
}

void Rectangle::sliceH(vector<Rectangle>& rects) {
    // sort all bottom and top bounds
    vector<int> ys;
    for (const Rectangle& rect : rects) {
        ys.push_back(rect.ly);
        ys.push_back(rect.hy);
    }
    sort(ys.begin(), ys.end());

    // remove duplicated values
    ys.erase(unique(ys.begin(), ys.end()), ys.end());

    // cut each rect with the y values
    vector<Rectangle> slices;
    for (const Rectangle& rect : rects) {
        vector<int>::iterator yi = lower_bound(ys.begin(), ys.end(), rect.ly);
        vector<int>::iterator ye = upper_bound(yi, ys.end(), rect.hy);
        int lasty = *(yi++);
        for (; yi != ye; yi++) {
            slices.push_back(Rectangle(rect.lx, lasty, rect.hx, *yi));
            lasty = *yi;
        }
    }
    // merge overlapping rect
    sort(slices.begin(), slices.end(), [](const Rectangle& a, const Rectangle& b) -> bool {
        return a.ly == b.ly ? a.lx < b.lx : a.ly < b.ly;
    });
    for (int i = 1; i < (int)slices.size(); ++i) {
        Rectangle& L = slices[i - 1];
        Rectangle& R = slices[i];
        if (L.ly != R.ly || L.hy != R.hy) {
            continue;
        }
        if (L.hx >= R.lx) {
            R.lx = min(L.lx, R.lx);
            R.hx = max(L.hx, R.hx);
            L.hx = L.lx;
        }
    }
    // remove empty rects
    vector<Rectangle>::iterator sEnd = remove_if(slices.begin(), slices.end(), Rectangle::IsInvalid);
    if (sEnd != slices.end()) {
        slices.resize(distance(slices.begin(), sEnd));
    }
    rects.swap(slices);
}

void Rectangle::sliceV(vector<Rectangle>& rects) {
    // sort all left and right bounds
    vector<int> xs;
    for (const Rectangle& rect : rects) {
        xs.push_back(rect.lx);
        xs.push_back(rect.hx);
    }
    sort(xs.begin(), xs.end());

    // remove duplicated values
    xs.erase(unique(xs.begin(), xs.end()), xs.end());

    // cut each rect with the y values
    vector<Rectangle> slices;
    for (const Rectangle& rect : rects) {
        vector<int>::iterator xi = lower_bound(xs.begin(), xs.end(), rect.lx);
        vector<int>::iterator xe = upper_bound(xi, xs.end(), rect.hx);
        int lastx = *(xi++);
        for (; xi != xe; xi++) {
            slices.push_back(Rectangle(lastx, rect.ly, *xi, rect.hy));
            lastx = *xi;
        }
    }
    // merge overlapping rect
    sort(slices.begin(), slices.end(), [](const Rectangle& a, const Rectangle& b) -> bool {
        return a.lx == b.lx ? a.ly < b.ly : a.lx < b.lx;
    });
    for (int i = 1; i < (int)slices.size(); ++i) {
        Rectangle& L = slices[i - 1];
        Rectangle& R = slices[i];
        if (L.lx != R.lx || L.hx != R.hx) {
            continue;
        }
        if (L.hy >= R.ly) {
            R.ly = min(L.ly, R.ly);
            R.hy = max(L.hy, R.hy);
            L.hy = L.ly;
        }
    }
    // remove empty rects
    vector<Rectangle>::iterator sEnd = remove_if(slices.begin(), slices.end(), Rectangle::IsInvalid);
    if (sEnd != slices.end()) {
        slices.resize(distance(slices.begin(), sEnd));
    }
    rects.swap(slices);
}

/***** Geometry *****/

bool Geometry::operator==(const Geometry& rhs) const { return layer == rhs.layer && Rectangle::operator==(rhs); };

/***** GeoMap *****/

void GeoMap::emplace(const int k, const Geometry& shape) {
    if (_map.find(k) == _map.end()) {
        _map.emplace(k, shape);
    } else {
        _map.at(k) += shape;
    }
    Rectangle::operator+=(shape);
}

