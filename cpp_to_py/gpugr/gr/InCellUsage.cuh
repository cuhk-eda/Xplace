#pragma once
#include "common/common.h"

namespace gr {

__device__ __forceinline__ float myExp(float x) { return (1 << min(30, static_cast<int>(x))); }

__device__ __forceinline__ int inCellViaUsage(int idx, int *vias, int N, int LAYER) {
    int layer = idx / N / N - 1, y = idx / N % N, x = idx % N;
    int ans = 0;
    if (layer + 2 < LAYER) ans += vias[idx];                 // a via from botLayer to currlayer
    if (layer >= 0) ans += vias[layer * N * N + x * N + y];  // a via from currlayer to topLayer
    return ans;
}

__device__ __forceinline__ float twoCellsViaUsage(int idx, int *vias, int N, int LAYER) {
    return sqrt(0.5 * (inCellViaUsage(idx, vias, N, LAYER) + inCellViaUsage(idx + 1, vias, N, LAYER))) * 1.5;
}

__device__ __forceinline__ float inCellUsedArea(int idx, int *wires, float *fixed, int N) {
    float ans = 0;
    if (idx % N > 0) ans += fixed[idx - 1] + wires[idx - 1];
    if (idx % N + 1 < N) ans += fixed[idx] + wires[idx];
    return ans / 2;
}

__device__ __forceinline__ float inCellViaCost(
    int idx, int *wires, float *fixed, const float *capacity, float logisticSlope, int N) {
    return 1.0 / (1.0 + myExp(logisticSlope * (capacity[idx] - inCellUsedArea(idx, wires, fixed, N))));
}

__device__ __forceinline__ float cellResource(
    int idx, int *wires, float *fixed, int *vias, const float *capacity, int N, int LAYER) {
    float ans = wires[idx] + fixed[idx];
    if (idx % N) ans += wires[idx - 1] + fixed[idx - 1];
    ans /= 2;
    ans += sqrt(1.0 * inCellViaUsage(idx, vias, N, LAYER)) * 1.5;
    return capacity[idx] - ans;
}

}  // namespace gr