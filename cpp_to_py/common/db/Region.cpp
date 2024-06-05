#include "Database.h"
using namespace db;

/***** Region *****/

void Region::addRect(const int xl, const int yl, const int xh, const int yh) {
    rects.emplace_back(xl, yl, xh, yh);
    lx = std::min(lx, xl);
    ly = std::min(ly, yl);
    hx = std::max(hx, xh);
    hy = std::max(hy, yh);
}
