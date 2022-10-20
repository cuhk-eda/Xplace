#pragma once

namespace db {

class NetRouteNode {
private:
    const Layer* _layer = nullptr;

public:
    int x = 0;
    int y = 0;
    int z = 0;
    Pin* pin;

    NetRouteNode(const Layer* layer = nullptr, const int x = 0, const int y = 0, const int z = 0);

    inline bool operator==(const NetRouteNode& r) const {
        return _layer == r._layer && x == r.x && y == r.y && z == r.z;
    }
};

class NetRouteSegment {
public:
    int z;
    unsigned fromNode;
    unsigned toNode;
    //  path = [<direction,len>]
    //  direction : N,S,E,W,U,D
    vector<pair<char, int>> path;

    NetRouteSegment(const unsigned fromi = 0, const unsigned toi = 0, const char dir = '\0', const unsigned len = 0);

    long long length() const;
};

class NetRouting {
public:
    vector<NetRouteNode> nodes;
    vector<NetRouteSegment> segments;

    void addWire(const Layer* layer,
                 const int fromx,
                 const int fromy,
                 const int fromz,
                 const int tox,
                 const int toy,
                 const int toz);
    void clear();
    long long length() const;
};

class Net {
private:
    NetRouting _routing;

    bool gRouted = false;
    bool dRouted = false;

public:
    const string name = "";
    std::vector<Pin*> pins;
    const NDR* ndr = nullptr;
    int gpdb_id = -1;

    Net(const string& name, const NDR* ndr = nullptr) : name(name), ndr(ndr) {}
    Net(const Net& net) : _routing(net._routing), name(net.name), pins(net.pins), ndr(net.ndr) {}

    bool globalRouted() const { return gRouted; }
    bool detailedRouted() const { return dRouted; }
    unsigned numPins() const { return pins.size(); }

    void resetRouting() { _routing.clear(); }
    void addPin(Pin* pin);
    void addWire(const Layer* layer,
                 const int fromx,
                 const int fromy,
                 const int fromz,
                 const int tox,
                 const int toy,
                 const int toz) {
        _routing.addWire(layer, fromx, fromy, fromz, tox, toy, toz);
    }
};

/*only support horizontal power rails*/
class PowerRail {
public:
    SNet* snet;
    int lx;
    int hx;
    PowerRail(SNet* sn, int l, int h) {
        snet = sn;
        lx = l;
        hx = h;
    }
};

class PowerNet {
private:
    std::map<int, SNet*> rails;

public:
    void addRail(SNet* snet, int lx, int hx, int y);
    bool getRowPower(int ly, int hy, char& topPower, char& botPower);
};
}  // namespace db
