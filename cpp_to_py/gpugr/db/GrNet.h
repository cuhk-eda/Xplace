#pragma once
#include <vector>
#include <unordered_map>
namespace gr {

class GrNet {
public:
    void setPins(const std::vector<std::vector<int>>& p) { pins = p; }
    void setBoundingBox(int lx, int ly, int ux, int uy) {
        lowerx = lx;
        lowery = ly;
        upperx = ux;
        uppery = uy;
    }
    void setWires(const std::vector<int>& w) { wires = w; }
    void setVias(const std::vector<int>& v) { vias = v; }
    void resetRoute() { wires.clear(), vias.clear(); }
    void addVias();
    bool needToRoute();
    void setNoRoute() { noroute = 1; }
    int area() { return (upperx - lowerx + 1) * (uppery - lowery + 1); }
    int hpwl() { return upperx + uppery - lowerx - lowery; }
    const std::vector<int>& getWires() { return wires; }
    const std::vector<int>& getVias() { return vias; }
    const std::vector<std::vector<int>>& getPins() { return pins; }
    int lowerx, lowery, upperx, uppery, noroute = 0;
    std::vector<int> points;
    std::vector<int> pin2gbpinId;
    std::vector<std::vector<int>> pin2gpdbPinIds;

private:
    std::vector<std::vector<int>> pins;
    std::vector<int> wires, vias;
};

}  // namespace gr
