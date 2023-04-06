#include "Database.h"
using namespace db;

/***** NetRouteNode *****/

NetRouteNode::NetRouteNode(const Layer* layer, const int x, const int y, const int z)
    : _layer(layer), x(x), y(y), z(z) {}

/***** NetRouteSegment *****/

NetRouteSegment::NetRouteSegment(const unsigned fromi, const unsigned toi, const char dir, const unsigned len)
    : fromNode(fromi), toNode(toi) {
    if (len) {
        path.emplace_back(dir, len);
    }
}

long long NetRouteSegment::length() const {
    long long len = 0;
    for (const auto& [dir, plen] : path) {
        if (dir != 'U' && dir != 'D') {
            len += plen;
        }
    }
    return len;
}

/***** NetRouting *****/

void NetRouting::addWire(const Layer* layer,
                         const int fromx,
                         const int fromy,
                         const int fromz,
                         const int tox,
                         const int toy,
                         const int toz) {
    NetRouteNode fromNode(layer, fromx, fromy, fromz);
    NetRouteNode toNode(layer, tox, toy, toz);

    int fromi = -1;
    int toi = -1;
    for (unsigned i = 0; i != nodes.size(); ++i) {
        if (nodes[i] == fromNode) {
            fromi = i;
        }
        if (nodes[i] == toNode) {
            toi = i;
        }
    }
    if (fromi < 0) {
        fromi = nodes.size();
        nodes.push_back(fromNode);
    }
    if (toi < 0) {
        toi = nodes.size();
        nodes.push_back(toNode);
    }

    if (fromi == toi) {
        return;
    }

    char dir = '\0';
    unsigned len = 0;
    if (fromx < tox) {
        dir = 'E';
        len = tox - fromx;
    } else if (fromx > tox) {
        dir = 'W';
        len = fromx - tox;
    } else if (fromy < toy) {
        dir = 'N';
        len = toy - fromy;
    } else {
        dir = 'S';
        len = fromy - toy;
    }

    segments.emplace_back(fromi, toi, dir, len);
}

void NetRouting::clear() {
    segments.clear();
    nodes.clear();
}

long long NetRouting::length() const {
    long long len = 0;
    for (const NetRouteSegment& seg : segments) {
        len += seg.length();
    }
    return len;
}

/***** Net *****/

void Net::addPin(Pin* pin) {
    // if (pin->type->direction() == 'o' && pins.size()) {
    //     Pin* firstPin = pins[0];
    //     pins[0] = pin;
    //     pins.push_back(firstPin);
    // } else {
    //     pins.push_back(pin);
    // }
    // NOTE: we don't swap the output pin to the vector head now
    pins.push_back(pin);
}

/***** PowerNet *****/

void PowerNet::addRail(SNet* snet, int lx, int hx, int y) {
    map<int, SNet*>::iterator rail = rails.find(y);
    if (rail != rails.end() && rail->second != snet) {
        logger.error("rail %s already exists at y=%d , new rail %s is from %d to %d", rail->second->name.c_str(),
                 y, snet->name.c_str(),
                 lx,
                 hx);
        return;

    }
    rails.emplace(y, snet);
}

bool PowerNet::getRowPower(int ly, int hy, char& topPower, char& botPower) {
    bool valid = true;
    topPower = 'x';
    botPower = 'x';
    map<int, SNet*>::iterator topRail = rails.find(hy);
    if (topRail != rails.end()) {
        topPower = topRail->second->type;
    } else {
        valid = false;
    }
    map<int, SNet*>::iterator botRail = rails.find(ly);
    if (botRail != rails.end()) {
        botPower = botRail->second->type;
    } else {
        valid = false;
    }
    return valid;
}

