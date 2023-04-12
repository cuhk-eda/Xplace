#include "Database.h"
using namespace db;

/***** PinSTA *****/
PinSTA::PinSTA() {
    type = 'x';
    capacitance = -1.0;
    for (int i = 0; i < 4; i++) {
        aat[i] = 0.0;
        rat[i] = 0.0;
        slack[i] = 0.0;
        nCriticalPaths[i] = 0;
    }
}
/***** IOPin *****/
IOPin::IOPin(const string& name, const string& netName, const char direction) : _netName(netName), name(name) {
    type = new PinType(name, direction, 's');
    pin = new Pin(this);
}

IOPin::~IOPin() {
    delete pin;
    delete type;
}

void IOPin::getBounds(int& lx, int& ly, int& hx, int& hy, int& rIndex) const {
    type->getBounds(lx, ly, hx, hy);
    lx += x;
    ly += y;
    hx += x;
    hy += y;
    if (type->shapes.size()) {
        rIndex = type->shapes[0].layer.rIndex;
    } else {
        rIndex = -1;
    }
}

/***** Pin *****/

void Pin::getPinCenter(int& x, int& y) {
    int lx, ly, hx, hy;
    type->getBounds(lx, ly, hx, hy);
    if (cell) {
        x = cell->lx() + (lx + hx) / 2;
        y = cell->ly() + (ly + hy) / 2;
    } else if (iopin) {
        x = iopin->x + (lx + hx) / 2;
        y = iopin->y + (ly + hy) / 2;
    } else {
        logger.error("invalid pin %s:%d", __FILE__, __LINE__);
        x = INT_MIN;
        y = INT_MIN;
    }
}

void Pin::getPinBounds(int& lx, int& ly, int& hx, int& hy, int& rIndex) {
    if (cell) {
        type->getBounds(lx, ly, hx, hy);
        lx += cell->lx();
        ly += cell->ly();
        hx += cell->lx();
        hy += cell->ly();
        rIndex = type->shapes[0].layer.rIndex;
    } else if (iopin) {
        iopin->getBounds(lx, ly, hx, hy, rIndex);
    } else {
        lx = INT_MAX;
        ly = INT_MAX;
        hx = INT_MIN;
        hy = INT_MIN;
        rIndex = -1;
    }
}

utils::BoxT<double> Pin::getPinParentBBox() const {
    double x, y, w, h;
    if (this->cell != nullptr) {  // cell pin
        x = this->cell->lx();
        y = this->cell->ly();
        w = this->cell->width();
        h = this->cell->height();
    } else {  // iopin
        x = this->iopin->lx();
        y = this->iopin->ly();
        w = this->iopin->width();
        h = this->iopin->height();
    }
    return utils::BoxT<double>(x, y, x + w, y + h);
}

utils::PointT<int> Pin::getPinParentCenter() const {
    int cx, cy;
    if (this->cell != nullptr) {  // cell pin
        cx = this->cell->cx();
        cy = this->cell->cy();
    } else {  // iopin
        cx = this->iopin->cx();
        cy = this->iopin->cy();
    }
    return utils::PointT<int>(cx, cy);
}

/***** Pin Type *****/

void PinType::addShape(const Layer& layer, const int lx, const int ly, const int hx, const int hy) {
    boundLX = min(boundLX, lx);
    boundLY = min(boundLY, ly);
    boundHX = max(boundHX, hx);
    boundHY = max(boundHY, hy);
    shapes.emplace_back(layer, lx, ly, hx, hy);
}

void PinType::getBounds(int& lx, int& ly, int& hx, int& hy) const {
    lx = boundLX;
    ly = boundLY;
    hx = boundHX;
    hy = boundHY;
}

bool PinType::operator<(const PinType& r) const {
    if (boundLX < r.boundLX) {
        return true;
    }
    if (boundLX > r.boundLX) {
        return false;
    }
    if (boundLY < r.boundLY) {
        return true;
    }
    if (boundLY > r.boundLY) {
        return false;
    }
    if (boundHX < r.boundHX) {
        return true;
    }
    if (boundHX > r.boundHX) {
        return false;
    }
    if (boundHY < r.boundHY) {
        return true;
    } else if (boundHY == r.boundHY) {
        if (shapes[0].layer.rIndex < r.shapes[0].layer.rIndex) {
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

bool PinType::comparePin(vector<PinType*> pins1, vector<PinType*> pins2) {
    std::sort(pins1.begin(), pins1.end(), [](PinType* lhs, PinType* rhs) { return (*lhs < *rhs); });
    std::sort(pins2.begin(), pins2.end(), [](PinType* lhs, PinType* rhs) { return (*lhs < *rhs); });
    for (unsigned j = 0; j != pins1.size(); ++j) {
        if (pins1[j]->_type != 'p' && pins1[j]->_type != 'g' && *pins1[j] != *pins2[j]) {
            return false;
        }
    }
    return true;
}
