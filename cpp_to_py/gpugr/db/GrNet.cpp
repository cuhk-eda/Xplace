#include "GrNet.h"

namespace gr {

bool GrNet::needToRoute() {
    std::unordered_map<int, int> cnt;
    for (auto e : pins) {
        for (auto f : e) {
            cnt[f]++;
        }
    }
    for (auto e : pins) {
        for (auto f : e) {
            if (cnt[f] == pins.size()) return false;
        }
    }
    return true;
}

}