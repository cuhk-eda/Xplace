#pragma once

namespace db {
class Track {
    // contain DEF track info
    // support multi layer track DEF definition
    // For example:
    //     TRACKS Y 0 DO 2752 STEP 200 LAYER metal1 metal3 metal5 ;
    // Variable will be:
    // layers_ = {metal1, metal3, metal5}
    // direction = 'h', start = 0, num = 2752, step = 200
protected:
    vector<string> layers_;

public:
    char direction = 'x';  // 'x' for None, 'v' for "X", 'h' for "Y"
    int start = INT_MAX;
    unsigned num = 0;
    unsigned step = 0;

    Track(const char direction = 'x', const int start = INT_MAX, const unsigned num = 0, const unsigned step = 0)
        : direction(direction), start(start), num(num), step(step) {}

    void addLayer(const string& layer) { layers_.push_back(layer); }

    const vector<string>& getLayer() const { return layers_; }
    int getFirstTrackLoc() const { return start; }
    int getLastTrackLoc() const { return start + (num - 1) * step; }
    int getTrackLoc(unsigned trackIndex) const { return start + trackIndex * step; }
    int getPitch() const { return step; }

    char macro() const;
    unsigned numLayers() const { return layers_.size(); }
    const string& layer(unsigned index) const { return layers_[index]; }
};

class Layer {
    friend class Database;

private:
    string _name = "";
    //'r' for route or 'c' for cut
    char _type = 'x';

    Layer* _below = nullptr;
    Layer* _above = nullptr;

public:
    char direction = 'x';
    //  int index;
    //  index at route layers 'M1' = 0
    int rIndex = -1;
    // index at cut layers 'M12' = 0
    int cIndex = -1;
    // for route layer
    int pitch = -1;
    int offset = -1;
    int width = -1;
    int area = -1;
    int minWidth = -1;
    int maxWidth = -1;

    int spacing = -1;                                      // minSpacing
    std::tuple<int, int, int> maxEOLSpace = {-1, -1, -1};  // spacing, width, within
    std::tuple<int, int, int, int, int> maxEOLSpaceParallelEdge = {
        -1, -1, -1, -1, -1};  // spacing, width, within, parSpace, parWithin

    // ParallelRunLength Spacing
    vector<int> parLength;
    vector<int> parWidth;
    vector<vector<int>> parWidthSpace; // 2D table: (parWidth, parLength) -> widthSpacing

    Track track;
    Track nonPreferDirTrack;

    Layer(const string& name = "", const char type = 'x') : _name(name), _type(type) {}

    const string& name() const { return _name; }

    bool isRouteLayer() const { return _type == 'r'; }
    bool isCutLayer() const { return _type == 'c'; }
    Layer* getLayerBelow() const { return _below; }
    Layer* getLayerAbove() const { return _above; }

    bool operator==(const Layer& rhs) const { return rIndex == rhs.rIndex && cIndex == rhs.cIndex; }
    bool operator!=(const Layer& rhs) const { return !(*this == rhs); }
};
}  // namespace db
