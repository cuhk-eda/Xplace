#include "Database.h"
using namespace db;

/***** SiteMap *****/

void SiteMap::initSiteMap(unsigned nx, unsigned ny) {
    this->nx = nx;
    this->ny = ny;
    sites.resize(nx, vector<unsigned char>(ny));
    regions.resize(nx, vector<unsigned char>(ny));
}

void SiteMap::getSiteBound(int x, int y, int &lx, int &ly, int &hx, int &hy) const {
    lx = siteL + x * siteStepX;
    ly = siteB + y * siteStepY;
    hx = lx + siteStepX;
    hy = ly + siteStepY;
}

void SiteMap::blockRegion(const unsigned x, const unsigned y) { regions[x][y] = Region::InvalidRegion; }

void SiteMap::setSites(int lx, int ly, int hx, int hy, unsigned char property, bool isContained) {
    int slx = 0;
    int sly = 0;
    int shx = 0;
    int shy = 0;
    if (isContained) {
        slx = binContainedL(lx, siteL, siteR, siteStepX);
        sly = binContainedL(ly, siteB, siteT, siteStepY);
        shx = binContainedR(hx, siteL, siteR, siteStepX);
        shy = binContainedR(hy, siteB, siteT, siteStepY);
    } else {
        slx = binOverlappedL(lx, siteL, siteR, siteStepX);
        sly = binOverlappedL(ly, siteB, siteT, siteStepY);
        shx = binOverlappedR(hx, siteL, siteR, siteStepX);
        shy = binOverlappedR(hy, siteB, siteT, siteStepY);
    }
    for (int x = slx; x <= shx; ++x) {
        for (int y = sly; y <= shy; ++y) {
            setSiteMap(x, y, property);
        }
    }
}

void SiteMap::unsetSites(int lx, int ly, int hx, int hy, unsigned char property) {
    const int slx = binOverlappedL(lx, siteL, siteR, siteStepX);
    const int sly = binOverlappedL(ly, siteB, siteT, siteStepY);
    const int shx = binOverlappedR(hx, siteL, siteR, siteStepX);
    const int shy = binOverlappedR(hy, siteB, siteT, siteStepY);
    for (int x = slx; x <= shx; ++x) {
        for (int y = sly; y <= shy; ++y) {
            unsetSiteMap(x, y, property);
        }
    }
}

void SiteMap::blockRegion(int lx, int ly, int hx, int hy) {
    int slx = binOverlappedL(lx, siteL, siteR, siteStepX);
    int sly = binOverlappedL(ly, siteB, siteT, siteStepY);
    int shx = binOverlappedR(hx, siteL, siteR, siteStepX);
    int shy = binOverlappedR(hy, siteB, siteT, siteStepY);
    for (int x = slx; x <= shx; x++) {
        for (int y = sly; y <= shy; y++) {
            setRegion(x, y, Region::InvalidRegion);
        }
    }
}
void SiteMap::setRegion(int lx, int ly, int hx, int hy, unsigned char region) {
    int slx = binContainedL(lx, siteL, siteR, siteStepX);
    int sly = binContainedL(ly, siteB, siteT, siteStepY);
    int shx = binContainedR(hx, siteL, siteR, siteStepX);
    int shy = binContainedR(hy, siteB, siteT, siteStepY);
    for (int x = slx; x <= shx; x++) {
        for (int y = sly; y <= shy; y++) {
            setRegion(x, y, region);
        }
    }
}
