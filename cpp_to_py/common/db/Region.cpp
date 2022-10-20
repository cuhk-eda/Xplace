#include "Database.h"
using namespace db;

/***** Region *****/

void Region::addRect(const int xl, const int yl, const int xh, const int yh) {
    rects.emplace_back(xl, yl, xh, yh);
    lx = min(lx, xl);
    ly = min(ly, yl);
    hx = max(hx, xh);
    hy = max(hy, yh);
}
