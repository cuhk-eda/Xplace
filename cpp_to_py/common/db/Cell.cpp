#include "Database.h"
using namespace db;

/***** Cell *****/

Cell::~Cell() {
    for (Pin* pin : _pins) {
        delete pin;
    }
    _pins.clear();
}

Pin* Cell::pin(const string& name) const {
    for (Pin* pin : _pins) {
        if (pin->type->name() == name) {
            return pin;
        }
    }
    return nullptr;
}

void Cell::ctype(CellType* t) {
    if (!t) {
        return;
    }
    if (_type) {
        logger.error("type of cell %s already set", _name.c_str());
        return;
    }
    _type = t;
    ++(_type->usedCount);
    _pins.resize(_type->pins.size(), nullptr);
    for (unsigned i = 0; i != _pins.size(); ++i) {
        _pins[i] = new Pin(this, i);
    }
}

int Cell::lx() const { return _lx; }
int Cell::ly() const { return _ly; }
bool Cell::flipX() const { return _flipX; }
bool Cell::flipY() const { return _flipY; }
int Cell::orient() const {
    if (!flipX() && !flipY()) {
        return 0;  // N
    } else if (flipX() && flipY()) {
        return 2;  // S
    } else if (flipX() && !flipY()) {
        return 4;  // FN
    } else if (!flipX() && flipY()) {
        return 6;  // FS
    }
    return 0;
}

bool Cell::placed() const { return (lx() != INT_MIN) && (ly() != INT_MIN); }
// int Cell::siteWidth() const { return width() / database.siteW; }
// int Cell::siteHeight() const { return height() / database.siteH; }

void Cell::place(int x, int y) {
    if (_fixed) {
        logger.warning("moving fixed cell %s to (%d,%d)", _name.c_str(), x, y);
    }
    _lx = x;
    _ly = y;
}

void Cell::place(int x, int y, int orient) {
    if (_fixed) {
        logger.warning("moving fixed cell %s to (%d,%d)", _name.c_str(), x, y);
    }
    _lx = x;
    _ly = y;
    switch (orient) {
        case 0:
            _flipX = false;
            _flipY = false;
            break;
        case 2:
            _flipX = true;
            _flipY = true;
            break;
        case 4:
            _flipX = true;
            _flipY = false;
            break;
        case 6:
            _flipX = false;
            _flipY = true;
            break;
        default:
            _flipX = false;
            _flipY = false;
            break;
    }
}

void Cell::place(int x, int y, bool flipX, bool flipY) {
    if (_fixed) {
        logger.warning("moving fixed cell %s to (%d,%d)", _name.c_str(), x, y);
    }
    _lx = x;
    _ly = y;
    _flipX = flipX;
    _flipY = flipY;
}

void Cell::unplace() {
    if (_fixed) {
        logger.warning("unplace fixed cell %s", _name.c_str());
    }
    _lx = _ly = INT_MIN;
    _flipX = _flipY = false;
}

/***** Cell Type *****/

CellType::~CellType() {
    for (PinType* pin : pins) {
        delete pin;
    }
}

PinType* CellType::addPin(const string& name, const char direction, const char type) {
    PinType* newpintype = new PinType(name, direction, type);
    pins.push_back(newpintype);
    return newpintype;
}

PinType* CellType::getPin(string& name) {
    for (int i = 0; i < (int)pins.size(); i++) {
        if (pins[i]->name() == name) {
            return pins[i];
        }
    }
    return nullptr;
}

void CellType::setOrigin(int x, int y) {
    _originX = x;
    _originY = y;
}

bool CellType::operator==(const CellType& r) const {
    if (width != r.width || height != r.height) {
        return false;
    } else if (_originX != r.originX() || _originY != r.originY() || _symmetry != r.symmetry() ||
               pins.size() != r.pins.size()) {
        return false;
    } else if (edgetypeL != r.edgetypeL || edgetypeR != r.edgetypeR) {
        return false;
    } else {
        //  return PinType::comparePin(pins, r.pins);
        for (unsigned i = 0; i != pins.size(); ++i) {
            if (*pins[i] != *r.pins[i]) {
                return false;
            }
        }
    }
    return true;
}
