#pragma once
#include <vector>

#include "Via.h"

namespace db {

using std::vector;

class Geometry;

class SNet {
public:
    string name;
    vector<Geometry> shapes;
    vector<Via> vias;
    char type = 'x';

    SNet(const string& name) : name(name) {}

    template <class... Args>
    void addShape(Args&&... args) {
        shapes.emplace_back(args...);
    }

    template <class... Args>
    void addVia(Args&&... args) {
        vias.emplace_back(args...);
    }
};
}  // namespace db
