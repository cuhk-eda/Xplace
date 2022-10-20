#pragma once

#include <set>

namespace db {
class Via {
public:
    int x;
    int y;
    ViaType* type;
    Via(ViaType* type, int x, int y) : x(x), y(y), type(type) {}
};

class ViaRule {
public:
    ~ViaRule() {
        botLayer = nullptr;
        cutLayer = nullptr;
        topLayer = nullptr;
    }
    bool hasViaRule = false;
    string name = "";

    std::pair<int, int> cutSize = {-1, -1};       // (X, Y)
    std::pair<int, int> cutSpacing = {-1, -1};    // (X, Y)
    std::pair<int, int> botEnclosure = {-1, -1};  // (X, Y)
    std::pair<int, int> topEnclosure = {-1, -1};  // (X, Y)

    int numCutRows = -1;
    int numCutCols = -1;

    std::pair<int, int> originOffset = {-1, -1};  // (X, Y)
    std::pair<int, int> botOffset = {-1, -1};     // (X, Y)
    std::pair<int, int> topOffset = {-1, -1};     // (X, Y)

    const Layer* botLayer = nullptr;  // Route Layer
    const Layer* cutLayer = nullptr;  // Cut Layer
    const Layer* topLayer = nullptr;  // Route Layer
};

class ViaType {
private:
    bool isDef_ = false;

public:
    string name = "";
    set<Geometry> rects;
    ViaRule rule;

    ViaType(const string& name = "", const bool isDef = false) : isDef_(isDef), name(name) {}

    template <class... Args>
    void addRect(Args&&... args) {
        rects.emplace(args...);
    }
    void isDef(bool b) { isDef_ = b; }
    bool isDef() const { return isDef_; }
};
}  // namespace db
