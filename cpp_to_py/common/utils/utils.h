#pragma once

#include "geo.h"
#include "log.h"
#include "robin_hood.h"

#include <algorithm>
#include <climits>
#include <cstdarg>
#include <string>

//#define isNaN(x) ((x)!=(x))

namespace utils {
template <typename T>
inline void minmax(const T v1, const T v2, T &minv, T &maxv) {
    if (v1 < v2) {
        minv = v1;
        maxv = v2;
    } else {
        minv = v2;
        maxv = v1;
    }
}
template <typename T>
inline void bounds(const T v, T &minv, T &maxv) {
    if (v < minv) {
        minv = v;
    } else if (v > maxv) {
        maxv = v;
    }
}
}  // namespace utils

// function for contain / overlap

inline int binContainedL(int lx, int blx, int bhx, int binw) { return (std::max(blx, lx) - blx + binw - 1) / binw; }
inline int binContainedR(int hx, int blx, int bhx, int binw) { return (std::min(bhx, hx) - blx) / binw - 1; }
inline int binOverlappedL(int lx, int blx, int bhx, int binw) { return (std::max(blx, lx) - blx) / binw; }
inline int binOverlappedR(int hx, int blx, int bhx, int binw) {
    return (std::min(bhx, hx) - blx + binw - 1) / binw - 1;
}

////////////INTEGER PACKING/////////////////
inline long long packInt(const int x, const int y) { return ((long)x) << 32 | ((long)y); }
inline void unpackInt(int &x, int &y, long long i) {
    x = (int)(i >> 32);
    y = (int)(i & 0xffffffff);
}
inline long long packCoor(int x, int y) {
    int ix = x + (INT_MAX >> 2);
    int iy = y + (INT_MAX >> 2);
    return packInt(ix, iy);
}
inline void unpackCoor(int &x, int &y, long long i) {
    unpackInt(x, y, i);
    x -= (INT_MAX >> 2);
    y -= (INT_MAX >> 2);
}

////////////FLOATING-POINT RANDOM NUMBER//////////////
inline int getrand(int lo, int hi) { return (rand() % (hi - lo + 1)) + lo; }
inline double getrand(double lo, double hi) { return (((double)rand() / (double)RAND_MAX) * (hi - lo) + lo); }

//////////////BIT MANIPULATION///////////////////////

template <typename T>
inline void setBit(T &val, T bit) {
    val |= bit;
}
template <typename T>
inline void unsetBit(T &val, T bit) {
    val &= (~bit);
}
template <typename T>
inline void toggleBit(T &val, T bit) {
    val ^= bit;
}
template <typename T>
inline bool isSetBit(T val, T bit) {
    return (val & bit) > 0;
}
template <typename T>
inline T getBit(T val, T bit) {
    return val & bit;
}

template <typename T>
inline T rect_overlap_area(T alx, T aly, T ahx, T ahy, T blx, T bly, T bhx, T bhy){
    if(alx>=bhx || ahx<=blx || aly>=bhy || ahy<=bly){
        return 0.0;
    }
    return (std::min(ahx,bhx) - std::max(alx,blx)) * (std::min(ahy,bhy) - std::max(aly,bly));
}
