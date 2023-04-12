#pragma once
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>

namespace gr {

extern const int MAX_LAYER_NUM, MAX_PIN_NUM;

class GPUMazeRouter {
public:
    void startGPU(int device_id, int layer, int x, int y);
    void endGPU();
    void query();
    void run(int DIRECTION, int iterleft);

    void getResults(double &t,
                    int64_t *costSum,
                    int *pins,
                    int *dist,
                    int *fgprev,
                    int *wireCost,
                    int *viaCost,
                    int *wires,
                    int *vias,
                    int *routes,
                    int N,
                    int SCALE,
                    int DIRECTION,
                    int routesOffset,
                    int netId);

    const static int MAX_TOT_PIN_NUM = 1000000, MAX_NUM_NET = 100;

    int MAX_TURN_NUM = 10;
    int LAYER, X, Y, NX, NY; // 0: x-1,x,x+1...  1: y-1,y,y+1...
    int *costMap, *viaMap, *markMap, *cudaRoutedPin;

    int *cost, *via, *costL, *costR, *cudaMap, *reset;
    int *cudaPrev;
    const int MAX_PIN_SIZE_PER_NET = 500000, MAX_PIN_NUM = 10000;

    int firstTime = 0; // varible to determine whether the first run of TF
};
extern int counter1, counter2, counter3, counter4;
extern double time1;

}  // namespace gr