#include "Database.h"
using namespace db;

/***** EdgeTypes *****/

int EdgeTypes::getEdgeType(const string& name) const {
    unsigned numTypes = types.size();
    for (unsigned i = 0; i != numTypes; ++i) {
        if (types[i] == name) {
            return i;
        }
    }
    return -1;
}

int EdgeTypes::getEdgeSpace(const int edge1, const int edge2) const {
    if (!distTable.size()) {
        return 0;
    }

#ifdef DEBUG
    if (edge1 < 0 || edge1 >= (int)types.size()) {
        logger.error("invalid edge ID: %d", edge1);
    }
    if (edge2 < 0 || edge2 >= (int)types.size()) {
        logger.error("invalid edge ID: %d", edge2);
    }
#endif
    return distTable[edge1][edge2];
}
