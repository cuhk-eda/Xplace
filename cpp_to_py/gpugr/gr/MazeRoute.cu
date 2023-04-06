#include <gpugr/taskflow/cudaflow.hpp>

#include "MazeRoute.h"

namespace gr {

#define tracing 0
#define INF 1000000000
__managed__ int sumCnt = 0, sumLen = 0, minLen = 100000, maxLen = 0, DDEBUG;

double GPUTime1 = 0, GPUTime2 = 0, GPUTime3 = 0;
__global__ void Init(int *map, int *prev) {
    map[blockIdx.x * blockDim.x + threadIdx.x] = INF;
    prev[blockIdx.x * blockDim.x + threadIdx.x] = blockIdx.x * blockDim.x + threadIdx.x;
}
__global__ void MinAll(int *x, int *y, int *prev, int LAYER, int X, int Y) {
    int delta = X * Y, idx = blockIdx.x * Y + threadIdx.x, p[10];
    for (int i = 0; i < LAYER; i++) p[i] = i;
    for (int i = 1; i < LAYER; i++) {
        if (x[idx + delta * i] > x[idx + delta * (i - 1)] + y[idx + delta * (i - 1)])
            x[idx + delta * i] = x[idx + delta * (i - 1)] + y[idx + delta * (i - 1)], p[i] = p[i - 1];
        if (x[idx + (LAYER - 1 - i) * delta] > x[idx + (LAYER - i) * delta] + y[idx + (LAYER - 1 - i) * delta])
            x[idx + (LAYER - 1 - i) * delta] = x[idx + (LAYER - i) * delta] + y[idx + (LAYER - 1 - i) * delta],
                                      p[LAYER - 1 - i] = p[LAYER - i];
    }
    for (int i = 0; i < LAYER; i++)
        if (p[i] != i) prev[idx + i * delta] = idx + p[i] * delta;
}

__global__ void SumL(int *cost, int *costSum, int X, int Y) {
    extern __shared__ int sum[];
    sum[threadIdx.x] = threadIdx.x == 0 ? 0 : cost[blockIdx.x * Y + threadIdx.x - 1];
    __syncthreads();
    for (int d = 0; (1 << d) < Y; d++) {
        if ((threadIdx.x >> d & 1)) sum[threadIdx.x] += sum[(threadIdx.x >> d << d) - 1];
        __syncthreads();
    }
    costSum[blockIdx.x * Y + threadIdx.x] = sum[threadIdx.x];
}
__global__ void SumR(int *cost, int *costSum, int X, int Y) {
    extern __shared__ int sum[];
    sum[threadIdx.x] = threadIdx.x == 0 ? 0 : cost[blockIdx.x * Y + Y - 1 - threadIdx.x];
    __syncthreads();
    for (int d = 0; (1 << d) < Y; d++) {
        if ((threadIdx.x >> d & 1)) sum[threadIdx.x] += sum[(threadIdx.x >> d << d) - 1];
        __syncthreads();
    }
    // printf("%d %d: %d\n", blockIdx.x, threadIdx.x, costSum[blockIdx.x * Y + Y - 1 - threadIdx.x])
    costSum[blockIdx.x * Y + Y - 1 - threadIdx.x] = sum[threadIdx.x];
}
__global__ void SumU(int *cost, int *costSum, int X, int Y) {
    extern __shared__ int sum[];
    sum[threadIdx.x] = threadIdx.x == 0 ? 0 : cost[(threadIdx.x - 1) * Y + blockIdx.x];
    __syncthreads();
    for (int d = 0; (1 << d) < X; d++) {
        if ((threadIdx.x >> d & 1)) sum[threadIdx.x] += sum[(threadIdx.x >> d << d) - 1];
        __syncthreads();
    }
    costSum[threadIdx.x * Y + blockIdx.x] = sum[threadIdx.x];
}

__global__ void SumD(int *cost, int *costSum, int X, int Y) {
    extern __shared__ int sum[];
    sum[threadIdx.x] = threadIdx.x == 0 ? 0 : cost[(X - 1 - threadIdx.x) * Y + blockIdx.x];
    __syncthreads();
    for (int d = 0; (1 << d) < X; d++) {
        if ((threadIdx.x >> d & 1)) sum[threadIdx.x] += sum[(threadIdx.x >> d << d) - 1];
        __syncthreads();
    }
    costSum[(X - 1 - threadIdx.x) * Y + blockIdx.x] = sum[threadIdx.x];
}
/*
#define POS(n) ((n) + ((n) >> 5))
__global__ void MinLR(int *costL, int *costR, int *map, int *prev, int startLayer, int LAYER, int X, int Y, int N) {
    extern __shared__ int shared[];
    int *cL = shared, *cR = cL + Y, *minL = cR + Y, *minR = minL + POS(N), *pL = minR + POS(N), *pR = pL + POS(N);
    //#pragma unroll
    for(int id = 2 * threadIdx.x; id <= 2 * threadIdx.x + 1; id++) if(id < Y) {
        minL[POS(id)] = minR[POS(Y - 1 - id)] = map[(blockIdx.y + (startLayer + 2 * blockIdx.x) * X) * Y + id];
        cL[id] = costL[(blockIdx.y + (startLayer + 2 * blockIdx.x) * X) * Y + id];
        cR[id] = costR[(blockIdx.y + (startLayer + 2 * blockIdx.x) * X) * Y + id];
        minL[POS(id)] -= cL[id];
        minR[POS(Y - 1 - id)] -= cR[id];
        pL[POS(id)] = pR[POS(id)] = id;
    }
    __syncthreads();
    /*if(startLayer + 2 * blockIdx.x == 1 && blockIdx.y == 26 && threadIdx.x == 0) {
        for(int i = 0; i < Y; i++)
            printf("cL[%d]=%d\n", i, cL[i]);
        for(int i = 0; i < Y; i++)
            printf("map[%d]=%d\n", i, cL[i] + minL[POS(i)]);
        for(int i = 0; i < Y; i++)
            printf("minL[%d]=%d\n ", i, minL[POS(i)]);
        for(int i = 0; i < Y; i++)
            printf("minR[%d]=%d\n ", i, minR[POS(i)]);
    *printf("minL Above\n");
        for(int i = 0; i < Y; i++)
            printf("%d ", minR[POS(i)]);
        printf("cR Above\n");
    }
    __syncthreads();
    int offset = 1;
    for(int d = N >> 1; d > 0; d >>= 1) {
        if(threadIdx.x < d) {
            int ai = offset * (2 * threadIdx.x + 1) - 1, bi = ai + offset;
            ai = POS(ai);
            bi = POS(bi);
            if(minL[ai] < minL[bi])
                minL[bi] = minL[ai], pL[bi] = pL[ai];
            if(minR[ai] < minR[bi])
                minR[bi] = minR[ai], pR[bi] = pR[ai];
        }
        offset <<= 1;
        __syncthreads();
    }
    if(threadIdx.x == 0)
        minL[POS(N - 1)] = minR[POS(N - 1)] = INF;
    __syncthreads();
    for(int d = 1; d < N; d <<= 1) {
        offset >>= 1;
        if(threadIdx.x < d) {
            int ai = offset * (2 * threadIdx.x + 1) - 1, bi = ai + offset;
            ai = POS(ai);
            bi = POS(bi);
            int t = minL[ai], pt = pL[ai];
            minL[ai] = minL[bi];
            pL[ai] = pL[bi];
            if(t < minL[bi])
                minL[bi] = t, pL[bi] = pt;
            t = minR[ai], pt = pR[ai];
            minR[ai] = minR[bi];
            pR[ai] = pR[bi];
            if(t < minR[bi])
                minR[bi] = t, pR[bi] = pt;
        }
        __syncthreads();
    }
    /*if(threadIdx.x == 0) {
        if(startLayer + 2 * blockIdx.x == 1 && blockIdx.y == 26) {
            printf("after sweeing\n");
            for(int i = 0; i < Y; i++)
                printf("minL[%d] = %d\n", i, minL[POS(i + 1)]);
            for(int i = 0; i < Y; i++)
                printf("map[%d]=%d\n", i, min(cL[i] + minL[POS(i + 1)], minR[POS(Y - 1 - i + 1)] + cR[i]));
        }
    }
    __syncthreads();
    //#pragma unroll
    for(int id = 2 * threadIdx.x; id <= 2 * threadIdx.x + 1; id++) if(id < Y) {
        int pos = (blockIdx.y + (startLayer + 2 * blockIdx.x) * X) * Y + id;
        //if(startLayer + 2 * blockIdx.x == 1 && blockIdx.y == 20)
        //    printf(" %d :%d pL %d\n", id, minL[POS(id + 1)], pL[POS(id + 1)]);
        if(min(minL[POS(id + 1)] + cL[id], minR[POS(Y - 1 - id + 1)] + cR[id]) >= map[(blockIdx.y + (startLayer + 2 *
blockIdx.x) * X) * Y + id]) continue; if(minL[POS(id + 1)] + cL[id] < minR[POS(Y - 1 - id + 1)] + cR[id]) { map[pos] =
minL[POS(id + 1)] + cL[id], prev[pos] = pos - id + pL[POS(id + 1)];
            //if(startLayer + 2 * blockIdx.x == 1 && blockIdx.y == 26)
            //    printf("prev[%d] = %d\n", id, pL[POS(id + 1)]);
        } else {
            map[pos] = minR[POS(Y - 1 - id + 1)] + cR[id], prev[pos] = pos - id + Y - 1 - pR[POS(Y - 1 - id + 1)];
            //if(startLayer + 2 * blockIdx.x == 1 && blockIdx.y == 26)
            //    printf("prev[%d] = %d\n", id, Y - 1 - pR[POS(Y - 1 - id + 1)]);
        //    if(startLayer + 2 * blockIdx.x == 1 && blockIdx.y == 20)
        //        printf(" %d :%d pR %d\n", id, minR[POS(Y - 1 - id + 1)], Y - 1 - pR[POS(Y - 1 - id + 1)]);

        }
    }
}*/
/*
__global__ void MinUD(int *costL, int *costR, int *map, int *prev, int startLayer, int LAYER, int X, int Y, int N) {
    extern __shared__ int shared[];
    int *cL = shared, *cR = cL + X, *minL = cR + X, *minR = minL + POS(N), *pL = minR + POS(N), *pR = pL + POS(N);
    #pragma unroll
    for(int id = 2 * threadIdx.x; id <= 2 * threadIdx.x + 1; id++) if(id < X) {
        minL[POS(id)] = map[(id + (startLayer + 2 * blockIdx.x) * X) * Y + blockIdx.y];
        minR[POS(X - 1 - id)] = minL[POS(id)];
        cL[id] = costL[(blockIdx.y + (startLayer + 2 * blockIdx.x) * Y) * X + id];
        cR[id] = costR[(blockIdx.y + (startLayer + 2 * blockIdx.x) * Y) * X + id];
        minL[POS(id)] -= cL[id];
        minR[POS(X - 1 - id)] -= cR[id];
    }
    __syncthreads();
    int offset = 1;
    for(int d = N >> 1; d > 0; d >>= 1) {
        if(threadIdx.x < d) {
            int ai = offset * (2 * threadIdx.x + 1) - 1, bi = ai + offset;
            ai = POS(ai);
            bi = POS(bi);
            if(minL[ai] < minL[bi])
                minL[bi] = minL[ai], pL[bi] = pL[ai];
            if(minR[ai] < minR[bi])
                minR[bi] = minR[ai], pR[bi] = pR[ai];
        }
        offset <<= 1;
        __syncthreads();
    }
    if(threadIdx.x == 0)
        minL[POS(N - 1)] = minR[POS(N - 1)] = INF;
    __syncthreads();
    for(int d = 1; d < N; d <<= 1) {
        offset >>= 1;
        if(threadIdx.x < d) {
            int ai = offset * (2 * threadIdx.x + 1) - 1, bi = ai + offset;
            ai = POS(ai);
            bi = POS(bi);
            int t = minL[ai], pt = pL[ai];
            minL[ai] = minL[bi];
            pL[ai] = pL[bi];
            if(t < minL[bi])
                minL[bi] = t, pL[bi] = pt;
            t = minR[ai], pt = pR[ai];
            minR[ai] = minR[bi];
            pR[ai] = pR[bi];
            if(t < minR[bi])
                minR[bi] = t, pR[bi] = pt;
        }
        __syncthreads();
    }
    #pragma unroll
    for(int id = 2 * threadIdx.x; id <= 2 * threadIdx.x + 1; id++) if(id < X) {
        int pos  = (id + (startLayer + 2 * blockIdx.x) * X) * Y + blockIdx.y;
        if(minL[POS(id + 1)] + cL[id] < minR[POS(X - 1 - id + 1)] + cR[id])
            map[pos] = minL[POS(id + 1)] + cL[id], prev[pos] = (pL[POS(id + 1)] + (startLayer + 2 * blockIdx.x) * X) * Y
+ blockIdx.y; else map[pos] = minR[POS(X - 1 - id + 1)] + cR[id], prev[pos] = (X - 1 - pR[POS(X - 1 - id + 1)] +
(startLayer + 2 * blockIdx.x) * X) * Y + blockIdx.y;
    }
}*/
/*
__global__ void sweepLR(int *costL, int *costR, int *map, int *prev, int layer, int X, int Y) {
    extern __shared__ int shared[];
    int index = blockIdx.x * Y + threadIdx.x, *minL = shared, *minR = minL + Y, *pL = minR + Y, *pR = pL + Y;
    layer += blockIdx.y * 2;
    costL += layer * X * Y;
    costR += layer * X * Y;
    map += layer * X * Y;
    prev += layer * X * Y;
    minL[threadIdx.x] = map[index] - costL[index];
    minR[Y - 1 - threadIdx.x] = map[index] - costR[index];
    pL[threadIdx.x] = pR[threadIdx.x] = threadIdx.x;
    __syncthreads();
    for(int d = 1; d < Y; d <<= 1) {
        if(threadIdx.x >= d) {
            int idx = threadIdx.x - d;
            if(minL[threadIdx.x] > minL[idx])
                minL[threadIdx.x] = minL[idx], pL[threadIdx.x] = pL[idx];
            if(minR[threadIdx.x] > minR[idx])
                minR[threadIdx.x] = minR[idx], pR[threadIdx.x] = pR[idx];
        }
        __syncthreads();
    }
    int val = min(minL[threadIdx.x] + costL[index], minR[Y - 1 - threadIdx.x] + costR[index]);
  if(val < map[index])
        map[index] = val, prev[index] = layer * X * Y + blockIdx.x * Y + (minL[threadIdx.x] + costL[index] == val ?
pL[threadIdx.x] : Y - 1 - pR[Y - 1 - threadIdx.x]);
}*/
/*
#define POS(n) ((n) +((n) >> 5))
__global__ void sweepLR(int *costL, int *costR, int *map, int *prev, int layer, int X, int Y) {
    extern __shared__ int shared[];
    int index = blockIdx.x * Y + threadIdx.x, *minL = shared, *minR = minL + POS(Y), *pL = minR + POS(Y), *pR = pL +
POS(Y), d; layer += blockIdx.y * 2; costL += layer * X * Y; costR += layer * X * Y; map += layer * X * Y; prev += layer
* X * Y; minL[POS(threadIdx.x)] = map[index] - costL[index]; minR[POS(Y - 1 - threadIdx.x)] = map[index] - costR[index];
    pL[POS(threadIdx.x)] = pR[POS(threadIdx.x)] = threadIdx.x;
    __syncthreads();
    for(d = 0; (1 << d) < Y; d++) {
        int to = ((threadIdx.x + 1) << (d + 1)) - 1;
        if(to < Y && to >= (1 << d)) {
            int from = POS(to - (1 << d));
            to = POS(to);
            if(minL[to] > minL[from])
                minL[to] = minL[from], pL[to] = pL[from];
            if(minR[to] > minR[from])
                minR[to] = minR[from], pR[to] = pR[from];
        }
        __syncthreads();
    }
    for(d -= 2; d >= 0; d--) {
        int from = ((threadIdx.x + 1) << (d + 1)) - 1;
        if(from + (1 << d) < Y) {
            int to = POS(from + (1 << d));
            from = POS(from);
            if(minL[to] > minL[from])
                minL[to] = minL[from], pL[to] = pL[from];
            if(minR[to] > minR[from])
                minR[to] = minR[from], pR[to] = pR[from];
        }
        __syncthreads();
    }
    int val = min(minL[POS(threadIdx.x)] + costL[index], minR[POS(Y - 1 - threadIdx.x)] + costR[index]);
  if(val < map[index])
        map[index] = val, prev[index] = layer * X * Y + blockIdx.x * Y + (minL[POS(threadIdx.x)] + costL[index] == val ?
pL[POS(threadIdx.x)] : Y - 1 - pR[POS(Y - 1 - threadIdx.x)]);
}
__global__ void sweepUD(int *costL, int *costR, int *map, int *prev, int layer, int X, int Y) {
    extern __shared__ int shared[];
    int index = threadIdx.x * Y + blockIdx.x, *minL = shared, *minR = minL + POS(X), *pL = minR + POS(X), *pR = pL +
POS(X), d; layer += blockIdx.y * 2; costL += layer * X * Y; costR += layer * X * Y; map += layer * X * Y; prev += layer
* X * Y; minL[POS(threadIdx.x)] = map[index] - costL[index]; minR[POS(X - 1 - threadIdx.x)] = map[index] - costR[index];
    pL[POS(threadIdx.x)] = pR[POS(threadIdx.x)] = threadIdx.x;
    __syncthreads();
    for(d = 0; (1 << d) < X; d++) {
        int to = ((threadIdx.x + 1) << (d + 1)) - 1;
        if(to < X && to >= (1 << d)) {
            int from = POS(to - (1 << d));
            to = POS(to);
            if(minL[to] > minL[from])
                minL[to] = minL[from], pL[to] = pL[from];
            if(minR[to] > minR[from])
                minR[to] = minR[from], pR[to] = pR[from];
        }
        __syncthreads();
    }
    for(d -= 2; d >= 0; d--) {
        int from = ((threadIdx.x + 1) << (d + 1)) - 1;
        if(from + (1 << d) < X) {
            int to = POS(from + (1 << d));
            from = POS(from);
            if(minL[to] > minL[from])
                minL[to] = minL[from], pL[to] = pL[from];
            if(minR[to] > minR[from])
                minR[to] = minR[from], pR[to] = pR[from];
        }
        __syncthreads();
    }
    int val = min(minL[POS(threadIdx.x)] + costL[index], minR[POS(X - 1 - threadIdx.x)] + costR[index]);
    if(val < map[index])
        map[index] = val, prev[index] = layer * X * Y + (minL[POS(threadIdx.x)] + costL[index] == val ?
pL[POS(threadIdx.x)] : X - 1 - pR[POS(X - 1 - threadIdx.x)]) * Y + blockIdx.x;
}
*/

/*
__global__ void Sweep(int *costL, int *costR, int *map, int LAYER, int X, int Y, int DIRECTION) {
    extern __shared__ int shared[];
    int *sharedMap = shared, sharedCostL = shared + LAYER * X * Y, sharedCostR = sharedCostL + LAYER * X * Y;
}*/

__global__ void resetLR(int *reset, int *map, int *prev, int layer, int X, int Y) {
    extern __shared__ int sum[];
    layer += blockIdx.y * 2;
    reset += layer * X * Y;
    map += layer * X * Y;
    prev += layer * X * Y;
    int index = blockIdx.x * Y + threadIdx.x;
    sum[threadIdx.x] = reset[index];
    __syncthreads();
    reset[index] = 0;
    for (int d = 0; (1 << d) < Y; d++) {
        if (threadIdx.x >> d & 1) sum[threadIdx.x] += sum[(threadIdx.x >> d << d) - 1];
        __syncthreads();
    }
    if (sum[threadIdx.x] > 0) map[index] = 0;  //, printf("resetLR (%d, %d, %d)\n", layer, blockIdx.x, threadIdx.x);
    // if(sum[threadIdx.x] > 1)
    //     printf("%d\n", sum[threadIdx.x]);
}
__global__ void sweepLR(int *costL, int *costR, int *map, int *prev, int layer, int X, int Y) {
    extern __shared__ int shared[];
    int *minL = shared, *minR = minL + Y, *pL = minR + Y, *pR = pL + Y;
    layer += blockIdx.y * 2;
    costL += layer * X * Y;
    costR += layer * X * Y;
    map += layer * X * Y;
    prev += layer * X * Y;
#pragma unroll
    for (int cur = threadIdx.x * 2; cur <= threadIdx.x * 2 + 1; cur++)
        if (cur < Y) {
            int index = blockIdx.x * Y + cur;
            minL[cur] = map[index] - costL[index];
            minR[Y - 1 - cur] = map[index] - costR[index];
            pL[cur] = pR[cur] = cur;
        }
    __syncthreads();
    for (int d = 0; (1 << d) < Y; d++) {
        int cur = (threadIdx.x >> d << (d + 1) | (1 << d)) | (threadIdx.x & ((1 << d) - 1));
        if (cur < Y) {
            int idx = (cur >> d << d) - 1;
            if (minL[cur] > minL[idx]) minL[cur] = minL[idx], pL[cur] = pL[idx];
            if (minR[cur] > minR[idx]) minR[cur] = minR[idx], pR[cur] = pR[idx];
        }
        __syncthreads();
    }
#pragma unroll
    for (int cur = threadIdx.x * 2; cur <= threadIdx.x * 2 + 1; cur++)
        if (cur < Y) {
            int index = blockIdx.x * Y + cur, val = min(minL[cur] + costL[index], minR[Y - 1 - cur] + costR[index]);
            if (val < map[index])
                map[index] = val, prev[index] = layer * X * Y + blockIdx.x * Y +
                                                (minL[cur] + costL[index] == val ? pL[cur] : Y - 1 - pR[Y - 1 - cur]);
        }
}
__global__ void resetUD(int *reset, int *map, int *prev, int layer, int X, int Y) {
    extern __shared__ int sum[];
    layer += blockIdx.y * 2;
    reset += layer * X * Y;
    map += layer * X * Y;
    prev += layer * X * Y;
    int index = threadIdx.x * Y + blockIdx.x;
    sum[threadIdx.x] = reset[index];
    __syncthreads();
    reset[index] = 0;
    for (int d = 0; (1 << d) < X; d++) {
        if (threadIdx.x >> d & 1) sum[threadIdx.x] += sum[(threadIdx.x >> d << d) - 1];
        __syncthreads();
    }
    if (sum[threadIdx.x] > 0) map[index] = 0;  //, printf("resetLR (%d, %d, %d)\n", layer, threadIdx.x, blockIdx.x);
    // if(sum[threadIdx.x] < 0)
    //     printf("ERROR\n");
}
__global__ void sweepUD(int *costL, int *costR, int *map, int *prev, int layer, int X, int Y) {
    extern __shared__ int shared[];
    int *minL = shared, *minR = minL + X, *pL = minR + X, *pR = pL + X;
    layer += blockIdx.y * 2;
    costL += layer * X * Y;
    costR += layer * X * Y;
    map += layer * X * Y;
    prev += layer * X * Y;
#pragma unroll
    for (int cur = threadIdx.x * 2; cur <= threadIdx.x * 2 + 1; cur++)
        if (cur < X) {
            int index = cur * Y + blockIdx.x;
            minL[cur] = map[index] - costL[index];
            minR[X - 1 - cur] = map[index] - costR[index];
            pL[cur] = pR[cur] = cur;
        }
    __syncthreads();
    for (int d = 0; (1 << d) < X; d++) {
        int cur = (threadIdx.x >> d << (d + 1) | (1 << d)) | (threadIdx.x & ((1 << d) - 1));
        if (cur < X) {
            int idx = (cur >> d << d) - 1;
            if (minL[cur] > minL[idx]) minL[cur] = minL[idx], pL[cur] = pL[idx];
            if (minR[cur] > minR[idx]) minR[cur] = minR[idx], pR[cur] = pR[idx];
        }
        __syncthreads();
    }
#pragma unroll
    for (int cur = threadIdx.x * 2; cur <= threadIdx.x * 2 + 1; cur++)
        if (cur < X) {
            int index = cur * Y + blockIdx.x;
            int val = min(minL[cur] + costL[index], minR[X - 1 - cur] + costR[index]);
            if (val < map[index])
                map[index] = val,
                prev[index] = layer * X * Y +
                              (minL[cur] + costL[index] == val ? pL[cur] : X - 1 - pR[X - 1 - cur]) * Y + blockIdx.x;
        }
}
__global__ void Path(int *map,
                     int *pins,
                     int *routedPin,
                     int *prev,
                     int *mark,
                     int LAYER,
                     int X,
                     int Y,
                     int N,
                     int DIRECTION,
                     int SCALE) {
    int minDist = INF, p = -1, lef = -1, rig = -1, pinId = -1, fgPin = -1;
    for (int i = 0, cur = 1; i < pins[0]; i++) {
        if (i && routedPin[i] == 0)
            for (int j = 1; j <= pins[cur]; j++) {
                int layer = pins[cur + j] / N / N, x = pins[cur + j] / N % N, y = pins[cur + j] % N;
                x /= SCALE;
                y /= SCALE;
                int pos = layer * X * Y + x * Y + y;
                if (!(layer & 1) ^ DIRECTION) pos = layer * X * Y + y * Y + x;
                if (map[pos] < minDist) {
                    minDist = map[pos];
                    fgPin = pins[cur + j];
                    p = pos;
                    lef = cur + 1;
                    rig = cur + pins[cur];
                    pinId = i;
                }
            }
        cur += pins[cur] + 1;
    }
    if (minDist == INF) {
        for (int i = 1; i < pins[0]; i++) printf("routed pin %d = %d\n", i, routedPin[i]);
        printf("Path finding failed\n");
        return;
    }
    if (p == -1) {
        printf("Path finding failed 2\n");
        return;
    }
    if (routedPin[pinId] == 0 && minDist == map[p]) {
        routedPin[pinId] = 1;
        mark[0] = 0;
        mark[++mark[0]] = fgPin;
        mark[++mark[0]] = lef;
        mark[++mark[0]] = rig;
        mark[++mark[0]] = p;
        // printf("FG pin: %d %d %d\n", fgPin / N / N, fgPin / N % N, fgPin % N);
        // printf("%d ---->  %d\n", p, prev[p]);
        while (map[p] > 0) {
            int layer = p / X / Y, x = p / Y % X, y = p % Y, _layer = prev[p] / X / Y, _x = prev[p] / Y % X,
                _y = prev[p] % Y, temp = prev[p];
            /*for(int i = min(_layer, layer); i <= max(_layer, layer); i++)
                for(int j = min(_x, x); j <= max(_x, x); j++)
                    for(int k = min(_y, y); k <= max(_y, y); k++) if(i * X * Y + j * Y + k != temp)
                        map[i * X * Y + j * Y + k] = 0, prev[i * X * Y + j * Y + k] = i * X * Y + j * Y + k;*/
            if (layer != _layer) {
                for (int i = min(_layer, layer); i <= max(_layer, layer); i++)
                    if (i * X * Y + x * Y + y != temp) {
                        map[i * X * Y + x * Y + y] = 0, prev[i * X * Y + x * Y + y] = i * X * Y + x * Y + y;
                    }
            } else {
                // map[p] = 0, prev[p] = p;
                for (int j = min(_x, x); j <= max(_x, x); j++)
                    for (int k = min(_y, y); k <= max(_y, y); k++)
                        if (layer * X * Y + j * Y + k != temp) {
                            map[layer * X * Y + j * Y + k] = 0,
                                                        prev[layer * X * Y + j * Y + k] = layer * X * Y + j * Y + k;
                        }
                /*reset[lower]++;
                if(y != _y && upper % Y + 1 < Y)
                    reset[upper + 1]--;
                if(x != _x && upper / Y % X + 1 < X)
                    reset[upper + Y]--;*/
            }
            mark[++mark[0]] = p = temp;
        }
        /*if(DDEBUG == 181060) {
            for(int i = mark[0]; i >= 4; i--)
                printf("(%d %d %d) -> ", mark[i] / X / Y, mark[i] / Y % X, mark[i] % Y);
            printf("\n");
        }*/
    } else
        printf("ERROR: Trace back\n");
    for (int i = lef; i <= rig; i++) {
        int layer = pins[i] / N / N, x = pins[i] / N % N, y = pins[i] % N;
        x /= SCALE;
        y /= SCALE;
        int pos = layer * X * Y + x * Y + y;
        if (!(layer & 1) ^ DIRECTION) pos = layer * X * Y + y * Y + x;
        map[pos] = 0;
        prev[pos] = pos;
    }
}

__global__ void setStart(int *d, int *pins, int X, int Y, int DIRECTION, int SCALE, int N) {
    // printf("starts: ");
    for (int i = 1; i <= pins[0]; i++) {
        int layer = pins[i] / N / N, x = pins[i] / N % N, y = pins[i] % N;
        x = x / SCALE;
        y = y / SCALE;
        if ((layer & 1) ^ DIRECTION) {
            d[layer * X * Y + x * Y + y] = 0;
            // printf("(%d %d %d)  ", layer, x, y);
        } else {
            d[layer * X * Y + y * Y + x] = 0;
            // printf("(%d %d %d)  ", layer, y, x);
        }
    }
    // printf("\n");
}

__device__ void viaSweep(int lower, int upper, int x, int y, int N, int DIRECTION, int *dist, int *prev, int *viaCost) {
    for (int l = lower; l < upper; l++) {
        int posLower = l * N * N + x * N + y, posUpper = (l + 1) * N * N + y * N + x;
        if (!(l & 1) ^ DIRECTION) posLower = l * N * N + y * N + x, posUpper = (l + 1) * N * N + x * N + y;
        if (dist[posUpper] > dist[posLower] + viaCost[posLower])
            dist[posUpper] = dist[posLower] + viaCost[posLower], prev[posUpper] = posLower;
    }
    for (int l = upper - 1; l >= lower; l--) {
        int posLower = l * N * N + x * N + y, posUpper = (l + 1) * N * N + y * N + x;
        if (!(l & 1) ^ DIRECTION) posLower = l * N * N + y * N + x, posUpper = (l + 1) * N * N + x * N + y;
        if (dist[posLower] > dist[posUpper] + viaCost[posLower])
            dist[posLower] = dist[posUpper] + viaCost[posLower], prev[posLower] = posUpper;
    }
}
__device__ void wireSweep(int layer, int x, int miny, int maxy, int N, int *dist, int *prev, int *wireCost) {
    for (int i = miny; i < maxy; i++) {
        int pos = layer * N * N + x * N + i;
        if (dist[pos + 1] > dist[pos] + wireCost[pos]) dist[pos + 1] = dist[pos] + wireCost[pos], prev[pos + 1] = pos;
    }
    for (int i = maxy; i > miny; i--) {
        int pos = layer * N * N + x * N + i;
        if (dist[pos - 1] > dist[pos] + wireCost[pos - 1])
            dist[pos - 1] = dist[pos] + wireCost[pos - 1], prev[pos - 1] = pos;
    }
}
__global__ void test(int offset, int p, int *wires, int *prev, int *dist) {
    if (threadIdx.x < blockDim.x) wires[offset + threadIdx.x]++;
    if (offset != p) prev[offset + threadIdx.x] = offset + threadIdx.x, dist[offset + threadIdx.x] = 0;
}
template <typename T>
__device__ void inline cudaSwap(T &a, T &b) {
    T c(a);
    a = b;
    b = c;
}
__global__ void routeFG(int64_t *costSum,
                        int *pins,
                        int *mark,
                        int *dist,
                        int *prev,
                        int *wireCost,
                        int *viaCost,
                        int *wires,
                        int *vias,
                        int *routes,
                        int LAYER,
                        int SCALE,
                        int X,
                        int Y,
                        int N,
                        int DIRECTION) {
    extern __shared__ int shared[];
    int *localDist = shared, *localPrev = localDist + LAYER * SCALE * SCALE,
        *localWireCost = localPrev + LAYER * SCALE * SCALE, *localViaCost = localWireCost + LAYER * SCALE * SCALE;
    // if(threadIdx.x == 0)
    //     sumCnt++, sumLen += mark[0] - 4 + 1, minLen = min(minLen, mark[0] - 4 + 1), maxLen = max(maxLen, mark[0] - 4
    //     + 1);
    for (int i = mark[0]; i >= 4; i--) {
        int layer = mark[i] / X / Y, x = mark[i] / Y % X, y = mark[i] % Y;
        if (i == mark[0] || i == 4) {
            int lower = max(layer - 9, 0), upper = min(LAYER - 1, layer + 9), minx = x * SCALE, miny = y * SCALE;
            int local_x = threadIdx.x / SCALE, local_y = threadIdx.x % SCALE, global_x = minx + local_x,
                global_y = miny + local_y;
            for (int l = lower; l <= upper; l++)
                if (global_x < N && global_y < N) {
                    if (!(l & 1) ^ DIRECTION) {
                        cudaSwap(local_x, local_y);
                        cudaSwap(global_x, global_y);
                    }
                    localDist[l * SCALE * SCALE + local_x * SCALE + local_y] =
                        dist[l * N * N + global_x * N + global_y];
                    localPrev[l * SCALE * SCALE + local_x * SCALE + local_y] = -1;
                    localWireCost[l * SCALE * SCALE + local_x * SCALE + local_y] =
                        wireCost[l * N * N + global_x * N + global_y];
                    localViaCost[l * SCALE * SCALE + local_x * SCALE + local_y] =
                        viaCost[l * N * N + global_x * N + global_y];
                    if (!(l & 1) ^ DIRECTION) {
                        cudaSwap(local_x, local_y);
                        cudaSwap(global_x, global_y);
                    }
                }
            __syncthreads();
            int numLayers = upper - max(lower, 1) + 1;
            for (int j = 0; j < 4; j++) {
                if (global_x < N && global_y < N)
                    viaSweep(lower, upper, local_x, local_y, SCALE, DIRECTION, localDist, localPrev, localViaCost);
                __syncthreads();
                for (int cur = threadIdx.x; cur < numLayers * SCALE; cur += blockDim.x) {
                    int l = upper - cur / SCALE;
                    if ((l & 1) ^ DIRECTION) {
                        if (minx + cur % SCALE < N)
                            wireSweep(l,
                                      cur % SCALE,
                                      0,
                                      min(SCALE, N - miny) - 1,
                                      SCALE,
                                      localDist,
                                      localPrev,
                                      localWireCost);
                    } else {
                        if (miny + cur % SCALE < N)
                            wireSweep(l,
                                      cur % SCALE,
                                      0,
                                      min(SCALE, N - minx) - 1,
                                      SCALE,
                                      localDist,
                                      localPrev,
                                      localWireCost);
                    }
                }
                __syncthreads();
            }
            for (int l = lower; l <= upper; l++)
                if (global_x < N && global_y < N) {
                    if (!(l & 1) ^ DIRECTION) {
                        cudaSwap(local_x, local_y);
                        cudaSwap(global_x, global_y);
                    }
                    if (dist[l * N * N + global_x * N + global_y] >
                        localDist[l * SCALE * SCALE + local_x * SCALE + local_y]) {
                        dist[l * N * N + global_x * N + global_y] =
                            localDist[l * SCALE * SCALE + local_x * SCALE + local_y];
                        int local_prev = localPrev[l * SCALE * SCALE + local_x * SCALE + local_y];
                        int local_layer = local_prev / SCALE / SCALE, local_prev_x = local_prev / SCALE % SCALE,
                            local_prev_y = local_prev % SCALE;
                        if ((local_layer & 1) ^ DIRECTION)
                            prev[l * N * N + global_x * N + global_y] =
                                local_layer * N * N + (local_prev_x + minx) * N + local_prev_y + miny;
                        else
                            prev[l * N * N + global_x * N + global_y] =
                                local_layer * N * N + (local_prev_x + miny) * N + local_prev_y + minx;
                    }
                    if (!(l & 1) ^ DIRECTION) {
                        cudaSwap(local_x, local_y);
                        cudaSwap(global_x, global_y);
                    }
                }
        }
        __syncthreads();
        if (i > 4) {
            int _layer = mark[i - 1] / X / Y, _x = mark[i - 1] / Y % X, _y = mark[i - 1] % Y;
            if (_layer != layer) {
                int lower = max(min(layer, _layer) - 9, 0), upper = min(LAYER - 1, max(layer, _layer) + 9),
                    minx = x * SCALE, miny = y * SCALE;
                int local_x = threadIdx.x / SCALE, local_y = threadIdx.x % SCALE, global_x = minx + local_x,
                    global_y = miny + local_y;
                for (int l = lower; l <= upper; l++)
                    if (global_x < N && global_y < N) {
                        if (!(l & 1) ^ DIRECTION) {
                            cudaSwap(local_x, local_y);
                            cudaSwap(global_x, global_y);
                        }
                        localDist[l * SCALE * SCALE + local_x * SCALE + local_y] =
                            dist[l * N * N + global_x * N + global_y];
                        localPrev[l * SCALE * SCALE + local_x * SCALE + local_y] = -1;
                        localWireCost[l * SCALE * SCALE + local_x * SCALE + local_y] =
                            wireCost[l * N * N + global_x * N + global_y];
                        localViaCost[l * SCALE * SCALE + local_x * SCALE + local_y] =
                            viaCost[l * N * N + global_x * N + global_y];
                        if (!(l & 1) ^ DIRECTION) {
                            cudaSwap(local_x, local_y);
                            cudaSwap(global_x, global_y);
                        }
                    }
                __syncthreads();
                int numLayers = upper - max(lower, 1) + 1;
                for (int j = 0; j < 4; j++) {
                    if (global_x < N && global_y < N)
                        viaSweep(lower, upper, local_x, local_y, SCALE, DIRECTION, localDist, localPrev, localViaCost);
                    __syncthreads();
                    for (int cur = threadIdx.x; cur < numLayers * SCALE; cur += blockDim.x) {
                        int l = upper - cur / SCALE;
                        if ((l & 1) ^ DIRECTION) {
                            if (minx + cur % SCALE < N)
                                wireSweep(l,
                                          cur % SCALE,
                                          0,
                                          min(SCALE, N - miny) - 1,
                                          SCALE,
                                          localDist,
                                          localPrev,
                                          localWireCost);
                        } else {
                            if (miny + cur % SCALE < N)
                                wireSweep(l,
                                          cur % SCALE,
                                          0,
                                          min(SCALE, N - minx) - 1,
                                          SCALE,
                                          localDist,
                                          localPrev,
                                          localWireCost);
                        }
                    }
                    __syncthreads();
                }
                for (int l = lower; l <= upper; l++)
                    if (global_x < N && global_y < N) {
                        if (!(l & 1) ^ DIRECTION) {
                            cudaSwap(local_x, local_y);
                            cudaSwap(global_x, global_y);
                        }
                        if (dist[l * N * N + global_x * N + global_y] >
                            localDist[l * SCALE * SCALE + local_x * SCALE + local_y]) {
                            dist[l * N * N + global_x * N + global_y] =
                                localDist[l * SCALE * SCALE + local_x * SCALE + local_y];
                            int local_prev = localPrev[l * SCALE * SCALE + local_x * SCALE + local_y];
                            int local_layer = local_prev / SCALE / SCALE, local_prev_x = local_prev / SCALE % SCALE,
                                local_prev_y = local_prev % SCALE;
                            if ((local_layer & 1) ^ DIRECTION)
                                prev[l * N * N + global_x * N + global_y] =
                                    local_layer * N * N + (local_prev_x + minx) * N + local_prev_y + miny;
                            else
                                prev[l * N * N + global_x * N + global_y] =
                                    local_layer * N * N + (local_prev_x + miny) * N + local_prev_y + minx;
                        }
                        if (!(l & 1) ^ DIRECTION) {
                            cudaSwap(local_x, local_y);
                            cudaSwap(global_x, global_y);
                        }
                    }
                __syncthreads();
            } else if (threadIdx.x < SCALE) {
                // int t = (layer & 1);
                // for(int layer = 1; layer < LAYER; layer++) if((layer & 1) == t) {
                int a, from, to;
                if ((layer & 1) ^ DIRECTION) {
                    a = x * SCALE + threadIdx.x;
                    from = y * SCALE;
                    to = _y * SCALE;
                } else {
                    a = y * SCALE + threadIdx.x;
                    from = x * SCALE;
                    to = _x * SCALE;
                }
                if (a < N) {
                    if (from < to) {
                        from = layer * N * N + a * N + from + SCALE - 1;
                        to = layer * N * N + a * N + to;
                        if (costSum[from] - costSum[to] < 1000000000) {
                            int tot = costSum[from] - costSum[to];
                            if (dist[from] + tot < dist[to]) dist[to] = dist[from] + tot, prev[to] = from;
                            // else
                            //    printf("%d %d [%d %d] error!  %d + %d >=  %d\n", (layer & 1) ^ DIRECTION, a, from % N,
                            //    to % N, dist[from], tot, dist[to]);
                        }
                    } else {
                        from = layer * N * N + a * N + from;
                        to = layer * N * N + a * N + to + SCALE - 1;
                        if (costSum[to] - costSum[from] < 1000000000) {
                            int tot = costSum[to] - costSum[from];
                            if (dist[from] + tot < dist[to]) dist[to] = dist[from] + tot, prev[to] = from;
                            // else
                            //    printf("%d %d [%d %d] error!  %d + %d >=  %d\n", (layer & 1) ^ DIRECTION, a, from % N,
                            //    to % N, dist[from], tot, dist[to]);
                        }
                    }
                }
                //}
            }
            __syncthreads();
        }
        __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        routes += pins[-1];
        int p = mark[1];
        while (dist[p] > 0) {
            int pp = prev[p];
            prev[p] = p;
            dist[p] = 0;
            int minp = min(p, pp), maxp = max(p, pp);
            // if(routes[0] >= 999)
            //     printf("ERROR!!!\n");
            if (p / N / N != pp / N / N)
                routes[routes[0]++] = minp, routes[routes[0]++] = -1, atomicAdd(vias + minp, 1);
            else {
                // test<<<1, maxp - minp>>> (minp, pp - minp, wires, prev, dist);
                // cudaDeviceSynchronize();
                for (int i = min(p, pp); i < max(p, pp); i++) atomicAdd(wires + i, 1);
                // prev[p] = p;
                // dist[p] = 0;
                for (int i = min(p, pp); i <= max(p, pp); i++)
                    if (i != pp) prev[i] = i, dist[i] = 0;

                // if(routes[0] >= 3 && routes[routes[0] - 1] != -1 && routes[routes[0] - 2] + routes[routes[0] - 1] ==
                // min(p, pp))
                //     routes[routes[0] - 1]++;
                // else if(routes[0] >= 3 && routes[routes[0] - 1] != -1 && routes[routes[0] - 2] - 1 == min(p, pp))
                //     routes[routes[0] - 2]--, routes[routes[0] - 1]++;
                // else
                routes[routes[0]++] = min(p, pp), routes[routes[0]++] = maxp - minp;
            }
            p = pp;
        }
        // for(int i = 1; i < routes[0]; i += 2)
        //     printf("(%d %d %d) >> ", routes[i] / N / N, routes[i] / N % N, routes[i] % N);
        // printf(" END\n");
        for (int i = mark[2]; i <= mark[3]; i++) dist[pins[i]] = 0, prev[pins[i]] = pins[i];
    }
    __syncthreads();
    /*for(int t = 1; t < routes[0]; t += 2) if(routes[t + 1] != -1)
        for(int i = routes[t] + threadIdx.x; i < routes[t] + routes[t + 1]; i += blockDim.x)
            atomicAdd(wires + i, 1);*/
}

__global__ void setInf(int *map, int *prev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (map[idx] > 0) map[idx] = INF, prev[idx] = idx;
}

void GPUMazeRouter::query() { printf("avg min max: %d %d %d\n", sumLen / sumCnt, minLen, maxLen); }

void GPUMazeRouter::run(int DIRECTION, int iterleft) {
    auto startTime = clock();
    MAX_TURN_NUM = 5;
    for (int i = 1; i < LAYER; i++)
        if ((i & 1) ^ DIRECTION) {
            SumL<<<X, Y, Y * sizeof(int)>>>(cost + i * X * Y, costL + i * X * Y, X, Y);
            SumR<<<X, Y, Y * sizeof(int)>>>(cost + i * X * Y, costR + i * X * Y, X, Y);
        } else {
            SumD<<<Y, X, X * sizeof(int)>>>(cost + i * X * Y, costR + i * X * Y, X, Y);
            SumU<<<Y, X, X * sizeof(int)>>>(cost + i * X * Y, costL + i * X * Y, X, Y);
        }
    cudaDeviceSynchronize();
    {
        cudaDeviceSynchronize();
        int errorType = cudaGetLastError();
        if (errorType) {
            std::cerr << "run.cu0 CUDA ERROR: " << errorType << std::endl;
            exit(0);
        }
    }
}

void GPUMazeRouter::getResults(double &time_cnt, int64_t *costSum, int *pins, int *dist, int *fgprev, int *wireCost, int *viaCost, int *wires, int *vias, int *routes, int N, int SCALE, int DIRECTION, int batchSize, int netId) {  
    DDEBUG = netId;
    /*printf("Pin Number: %d\n", pins[1]);
    printf("Coarse-Grained:\n");
    for(int i = 0, cur = 2; i < pins[1]; i++) {
        for(int j = 1; j <= pins[cur]; j++) {
            int layer = pins[cur + j] / N / N, x = pins[cur + j] / N % N, y = pins[cur + j] % N;
            if(((layer & 1) ^ DIRECTION) == 0) 
                std::swap(x, y);
            x /= SCALE;
            y /= SCALE;
            printf("(%d %d %d) ", layer, x, y);
        }
        puts("");
        cur += pins[cur] + 1;
    }
    printf("Fine-Grained:\n");
    for(int i = 0, cur = 2; i < pins[1]; i++) {
        for(int j = 1; j <= pins[cur]; j++) {
            int layer = pins[cur + j] / N / N, x = pins[cur + j] / N % N, y = pins[cur + j] % N;
            printf("(%d %d %d) ", layer, x, y);
        }
        puts("");
        cur += pins[cur] + 1;
    }
    puts("--END--");*/

    // FIXME: use taskflow for MazeRoute would be faster while it would cause unknown bug 
    //        during terminating the Python program
    constexpr bool use_tf = false;
    int *map = cudaMap, *prev = cudaPrev, *mark = markMap, *routedPin = cudaRoutedPin;
    dim3 blockLR(X, LAYER / 2), blockUD(Y, LAYER / 2);
    if (use_tf) {
        std::vector<tf::cudaScopedPerThreadStream> streams(batchSize);
        const int LOG = 1;
        static std::vector<tf::cudaFlow> flow(MAX_NUM_NET * LOG);
        cudaDeviceSynchronize();
        double t = clock();
        if(firstTime != MAX_TURN_NUM) {
            firstTime = MAX_TURN_NUM;
            for(int d = 0; d < LOG; d++) {
                for(int i = 0; i < MAX_NUM_NET; i++) {
                    std::vector<tf::cudaTask> minall(MAX_TURN_NUM << d), sweeplr(MAX_TURN_NUM << d), sweepud(MAX_TURN_NUM << d), path(1 << d), fgroute(1 << d);
                    for(int iter = 0; iter < (1 << d); iter++) {              
                        tf::cudaTask setInfinity = flow[i * LOG + d].kernel(LAYER * X, Y, 0, setInf, map + i * LAYER * X * Y, prev + i * LAYER * X * Y).name("setInfinity");
                        if(iter)
                            setInfinity.succeed(fgroute[iter - 1]);
                        for(int turn = 0; turn < MAX_TURN_NUM; turn++) {
                            minall[iter * MAX_TURN_NUM + turn] = flow[i * LOG + d].kernel(X, Y, 0, MinAll, map + i * LAYER * X * Y, via, prev + i * LAYER * X * Y, LAYER, X, Y).name("minall");
                            if(turn)
                                minall[iter * MAX_TURN_NUM + turn].succeed(sweeplr[iter * MAX_TURN_NUM + turn - 1]).succeed(sweepud[iter * MAX_TURN_NUM + turn - 1]);
                            //else if(iter)                            
                            //    minall[iter * MAX_TURN_NUM + turn].succeed(fgroute[iter - 1]);
                            else
                                minall[iter * MAX_TURN_NUM + turn].succeed(setInfinity);
                            sweeplr[iter * MAX_TURN_NUM + turn] = flow[i * LOG + d].kernel(blockLR, (Y + 1) / 2, 4 * Y * sizeof(int), sweepLR, costL, costR, map + i * LAYER * X * Y, prev + i * LAYER * X * Y, 1 + (DIRECTION == 1), X, Y).succeed(minall[iter * MAX_TURN_NUM + turn]);
                            sweepud[iter * MAX_TURN_NUM + turn] = flow[i * LOG + d].kernel(blockUD, (X + 1) / 2, 4 * X * sizeof(int), sweepUD, costL, costR, map + i * LAYER * X * Y, prev + i * LAYER * X * Y, 1 + (DIRECTION == 0), X, Y).succeed(minall[iter * MAX_TURN_NUM + turn]);
                        }        
                        path[iter] = flow[i * LOG + d].kernel(1, 1, 0, Path, map + i * LAYER * X * Y, pins + i * MAX_PIN_SIZE_PER_NET + 1, routedPin + i * MAX_PIN_NUM, prev + i * LAYER * X * Y, mark + i * MAX_PIN_NUM * 10, LAYER, X, Y, N, DIRECTION, SCALE).name("path" + std::to_string(d)).succeed(sweeplr[(iter + 1) * MAX_TURN_NUM - 1]).succeed(sweepud[(iter + 1) * MAX_TURN_NUM - 1]);
                        fgroute[iter] = flow[i * LOG + d].kernel(1, SCALE * SCALE, 4 * LAYER * SCALE * SCALE * sizeof(int), routeFG, costSum, pins + i * MAX_PIN_SIZE_PER_NET + 1, mark + i * MAX_PIN_NUM * 10, dist + i * LAYER * N * N, fgprev + i * LAYER * N * N, wireCost, viaCost, wires, vias, routes, LAYER, SCALE, X, Y, N, DIRECTION).name("fgroute" + std::to_string(d)).succeed(path[iter]);
                    }
                    flow[i * LOG + d].instantiate();
                }
            }
        // for(int i = 0; i + 1 < LAYER; i++)
            //    path.succeed(sweep[offset + i]);
            //fgroute.succeed(path);
        }
        cudaDeviceSynchronize();
        time_cnt += clock() - t;

        //double t = clock();
        //flow.offload_n(pins[0] - 1);
        cudaMemset(reset, 0, batchSize * LAYER * X * Y * sizeof(int));
        for(int net = 0; net < batchSize; net++) {
            Init<<<LAYER * X, Y, 0, streams[net]>>> (map, prev);
            setStart<<<1, 1, 0, streams[net]>>>(map, pins + 2, X, Y, DIRECTION, SCALE, N);
            cudaMemsetAsync(routedPin, 0, pins[1] * sizeof(int), streams[net]);
            //dim3 blockLR(LAYER / 2, X);
            int len = pins[1] - 1, cur = LOG - 1;
            while(len) {
                if(len < (1 << cur)) cur--;
                else
                    flow[net * LOG + cur].offload_to(streams[net]), len -= 1 << cur;
            }
            //cudaDeviceSynchronize();
            //exit(0);
            map += LAYER * X * Y;
            prev += LAYER * X * Y;
            mark += MAX_PIN_NUM * 10;
            routedPin += MAX_PIN_NUM;
            pins += MAX_PIN_SIZE_PER_NET;
            dist += LAYER * N * N;
            fgprev += LAYER * N * N;
        }
    } else {
        std::vector<cudaStream_t> streams(batchSize);
        for(int net = 0; net < batchSize; net++) {
            // -------------------------
            Init<<<LAYER * X, Y, 0, streams[net]>>> (map, prev);
            setStart<<<1, 1, 0, streams[net]>>>(map, pins + 2, X, Y, DIRECTION, SCALE, N);
            cudaMemsetAsync(routedPin, 0, pins[1] * sizeof(int), streams[net]);
            double t = clock();
            //dim3 blockLR(LAYER / 2, X);
            // int len = pins[1] - 1, cur = LOG - 1;
            for(int iteration = pins[1]; iteration >= 2; iteration--) {
                //flow[net].offload_to(streams[net]);
                //cudaStreamSynchronize(streams[net]);
                setInf<<<LAYER * X, Y, 0, streams[net]>>> (map, prev);

                for(int turn = MAX_TURN_NUM; turn >= 1; turn--) {
                    MinAll<<<X, Y, 0, streams[net]>>> (map, via, prev, LAYER, X, Y);
                    sweepLR<<<blockLR, Y, 4 * Y * sizeof(int), streams[net]>>> (costL, costR, map, prev, 1 + (DIRECTION == 1), X, Y);
                    sweepUD<<<blockUD, X, 4 * X * sizeof(int), streams[net]>>> (costL, costR, map, prev, 1 + (DIRECTION == 0), X, Y);
                }
                Path<<<1, 1, 0, streams[net]>>> (map, pins + 1, routedPin, prev, mark, LAYER, X, Y, N, DIRECTION, SCALE);
                routeFG<<<1, SCALE * SCALE, 4 * LAYER * SCALE * SCALE * sizeof(int), streams[net]>>> (costSum, pins + 1, mark, dist, fgprev, wireCost, viaCost, wires, vias, routes, LAYER, SCALE, X, Y, N, DIRECTION);
            }
            time_cnt += clock() - t;
            //cudaDeviceSynchronize();
            //exit(0);
            map += LAYER * X * Y;
            prev += LAYER * X * Y;
            mark += MAX_PIN_NUM * 10;
            routedPin += MAX_PIN_NUM;
            pins += MAX_PIN_SIZE_PER_NET;
            dist += LAYER * N * N;
            fgprev += LAYER * N * N;
        }
    }
    //time_cnt += clock() - t;
    //}
    { cudaDeviceSynchronize(); int errorType = cudaGetLastError(); if(errorType) { std::cerr << "run.2 CUDA ERROR: " << errorType  << std::endl; exit(0); }}
}

void GPUMazeRouter::startGPU(int device_id, int layer, int x, int y) {
    cudaSetDevice(device_id);
    LAYER = layer;
    X = x;
    Y = y;
    NX = NY = 1;
    while (NX <= X) NX <<= 1;
    while (NY <= Y) NY <<= 1;
    cudaMalloc(&markMap, MAX_PIN_NUM * MAX_NUM_NET * 10 * sizeof(int));
    cudaMalloc(&cost, LAYER * X * Y * sizeof(int));
    cudaMalloc(&via, LAYER * X * Y * sizeof(int));
    cudaMalloc(&costL, LAYER * X * Y * sizeof(int));
    cudaMalloc(&costR, LAYER * X * Y * sizeof(int));
    cudaMalloc(&reset, MAX_NUM_NET * LAYER * X * Y * sizeof(int));
    cudaMemset(reset, 0, MAX_NUM_NET * LAYER * X * Y * sizeof(int));
    cudaMalloc(&cudaMap, MAX_NUM_NET * LAYER * X * Y * sizeof(int));
    cudaMalloc(&cudaRoutedPin, MAX_PIN_NUM * MAX_NUM_NET * sizeof(int));
    cudaMalloc(&cudaPrev, MAX_NUM_NET * LAYER * X * Y * sizeof(int));
}

void GPUMazeRouter::endGPU() {
    cudaFree(markMap);
    cudaFree(cost);
    cudaFree(via);
    cudaFree(costL);
    cudaFree(costR);
    cudaFree(reset);
    cudaFree(cudaMap);
    cudaFree(cudaRoutedPin);
    cudaFree(cudaPrev);
}

}  // namespace gr