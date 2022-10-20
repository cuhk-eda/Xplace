#pragma once

namespace db {
class WireRule {
private:
    const Layer* _layer;

public:
    const unsigned width;
    const unsigned space;

    WireRule(const Layer* layer, const unsigned width, const unsigned space)
        : _layer(layer), width(width), space(space) {}

    const Layer* layer() const { return _layer; }
};

class NDR {
private:
    const string _name;
    bool _hardSpacing = false;

public:
    vector<WireRule> rules;
    vector<ViaType*> vias;

    NDR(const string& name, const bool hardSpacing) : _name(name), _hardSpacing(hardSpacing) {}

    const string& name() const { return _name; }
    bool hardSpacing() const { return _hardSpacing; }
};

class EdgeTypes {
public:
    vector<string> types = {"default"};
    vector<vector<int>> distTable = {{0}};

    int getEdgeType(const string& name) const;
    int getEdgeSpace(const int edge1, const int edge2) const;
};
}  // namespace db
