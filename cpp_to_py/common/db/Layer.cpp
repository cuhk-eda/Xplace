#include "Database.h"
using namespace db;

/***** Track *****/

char Track::macro() const {
    switch (direction) {
        case 'h':
            return 'Y';
        case 'v':
            return 'X';
        default:
            logger.error("track direction not recognized: %c", direction);
            return '\0';
    }
}
