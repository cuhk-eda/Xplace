#include "GPURouter.h"
#include "InCellUsage.cuh"
#include <cstdio>

namespace gr {

constexpr int MAX_ROUTE_LEN_PER_PIN = 130;   // too large may exceed the maximum GPU memory

constexpr int INF = 10000000;
constexpr int MAX_COST = 10000000;

#define BLOCK_SIZE 512
#define BLOCK_NUMBER(n) (((n) + (BLOCK_SIZE) - 1) / BLOCK_SIZE)

__managed__ int STAMP = 0, wireLen, viaLen;

void GPURouter::initialize(int device_id, int layer, int x, int y, int N_, int cgxsize_, int cgysize_, int direction, int csrn_scale) {
    gpuMR.startGPU(device_id, layer, cgxsize_, cgysize_);
    DEVICE_ID = device_id;
    DIRECTION = direction;
    COARSENING_SCALE = csrn_scale;
    cgxsize = cgxsize_;
    cgysize = cgysize_;

    LAYER = layer;
    N = N_;
    X = x;
    Y = y;
    int gridGraphSize = LAYER * N * N;
    cudaMalloc(&dist, (MAX_BATCH_SIZE + 6) * gridGraphSize * sizeof(int));
    cudaMalloc(&prev, (MAX_BATCH_SIZE + 6) * gridGraphSize * sizeof(int));
    cudaMalloc(&capacity, gridGraphSize * sizeof(float));
    cudaMalloc(&wireDist, gridGraphSize * sizeof(float));
    cudaMalloc(&fixedLength, gridGraphSize * sizeof(float));
    cudaMalloc(&fixed, gridGraphSize * sizeof(float));
    cudaMalloc(&wires, gridGraphSize * sizeof(int));
    cudaMalloc(&vias, gridGraphSize * sizeof(int));    
    cudaMemset(wires, 0, sizeof(int) * gridGraphSize);
    cudaMemset(vias, 0, sizeof(int) * gridGraphSize);
    cudaMalloc(&modifiedWire, gridGraphSize * sizeof(int));
    cudaMalloc(&modifiedVia, gridGraphSize * sizeof(int));
    cudaMalloc(&viaCost, gridGraphSize * sizeof(dtype));
    cudaMalloc(&cost, gridGraphSize * sizeof(dtype));
    cudaMalloc(&costSum, gridGraphSize * sizeof(int64_t));
    cudaMalloc(&cell_resource, gridGraphSize * sizeof(float));
    cudaMalloc(&isOverflowWire, gridGraphSize * sizeof(int));
    cudaMalloc(&isOverflowVia, gridGraphSize * sizeof(int));
    cudaMalloc(&unitShortCostDiscounted, LAYER * sizeof(float));
    cudaMallocManaged(&allpins, MAX_BATCH_SIZE * MAX_PIN_SIZE_PER_NET * sizeof(int));
}

GPURouter::~GPURouter() {
    gpuMR.endGPU();
    cudaFree(dist);
    cudaFree(prev);
    cudaFree(capacity);
    cudaFree(wireDist);
    cudaFree(fixedLength);
    cudaFree(fixed);
    cudaFree(wires);
    cudaFree(vias);
    cudaFree(modifiedWire);
    cudaFree(modifiedVia);
    cudaFree(viaCost);
    cudaFree(cost);
    cudaFree(costSum);
    cudaFree(cell_resource);
    cudaFree(isOverflowWire);
    cudaFree(isOverflowVia);
    cudaFree(unitShortCostDiscounted);
    cudaFree(allpins);

    if(pins != nullptr) cudaFree(pins);
    if(pinNum != nullptr) cudaFree(pinNum);
    if(pinNumOffset != nullptr) cudaFree(pinNumOffset);
    if(routes != nullptr) cudaFree(routes);
    if(routesOffset != nullptr) cudaFree(routesOffset);
    if(isOverflowNet != nullptr) cudaFree(isOverflowNet);
    if(points != nullptr) cudaFree(points);
    if(gbpoints != nullptr) cudaFree(gbpoints);
    if(gbpinRoutes != nullptr) cudaFree(gbpinRoutes);
    if(gbpin2netId != nullptr) cudaFree(gbpin2netId);
    if(plPinId2gbPinId != nullptr) cudaFree(plPinId2gbPinId);

    if(routesOffsetCPU != nullptr) { delete[] routesOffsetCPU; }
    if(pinNumCPU != nullptr) { delete[] pinNumCPU; }
}

void GPURouter::setUnitViaMultiplier(float value) {
    unitViaMultiplier = value;
}

void GPURouter::setUnitVioCost(vector<float>& values, float discount) {
    //printf("??? setUnitVioCost %.2f\n", discount);
    float temp[100];
    for(int i = 0; i < LAYER; i++)
        temp[i] = values[i] * discount;
    cudaMemcpy(unitShortCostDiscounted, temp, LAYER * sizeof(float), cudaMemcpyHostToDevice);
}

void GPURouter::setLogisticSlope(float value) {
    logisticSlope = value;
}

void GPURouter::setUnitViaCost(float value) {
    unitViaCost = value;
}

void GPURouter::setMap(const vector<float> &cap, const vector<float> &wir, const vector<float> &fixedL, const vector<float> &fix) {
    int gridGraphSize = LAYER * N * N;
    auto copy = [&] (const vector<float> &vec, float *target) {
        cudaMemcpy(target, vec.data(), gridGraphSize * sizeof(float), cudaMemcpyHostToDevice);
    };
    copy(cap, capacity);
    copy(wir, wireDist);
    copy(fixedL, fixedLength);
    copy(fix, fixed);
}

__global__ void calculateCellResource(float *cell_resource, int *wires, float *fixed, int *vias, const float *capacity, int N, int LAYER, int tot) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= tot) return;
    cell_resource[idx] = cellResource(idx, wires, fixed, vias, capacity, N, LAYER);
}

__global__ void calculateCoarseCost(float *cell_resource, int *cost, int *wires, float *fixed, int *vias, float *capacity, int N, int xsize, int ysize, int X, int Y, int LAYER, int DIRECTION, int COARSENING_SCALE) {
    int layer = blockIdx.x / xsize, x = blockIdx.x % xsize, y = threadIdx.x;
    if(layer == 0) {
        cost[blockIdx.x * blockDim.x + threadIdx.x] = 10000000;
        return;
    }
    int minx = x * COARSENING_SCALE, maxx = min(X - 1, x * COARSENING_SCALE + COARSENING_SCALE - 1);
    int miny = y * COARSENING_SCALE, maxy = min(Y - 1, y * COARSENING_SCALE + COARSENING_SCALE - 1);
    float ans = 0;
    if(DIRECTION ^ (layer & 1)) {
        if(y + 1 < ysize) {
            float sum = 0;
            for(int i = minx; i <= maxx; i++)
                for(int j = miny; j <= maxy; j++) {
                    //if(layer == 1 && x == 3 && y == 18)
                    //    printf("GPU %d %d: %.2lf\n", i, j, cellResource(layer * N * N + i * N + j, wires, fixed, vias, capacity, N, LAYER));
                    sum += cell_resource[layer * N * N + i * N + j];
                }
            ans += 1.0 * COARSENING_SCALE / max(0.1, sum / (maxx - minx + 1) / (maxy - miny + 1));
            //if(layer == 1 && x == 3 && y == 18)
            //    printf("sum: %.2lf\n", sum);
            sum = 0;
            for(int i = minx; i <= maxx; i++)
                for(int j = miny + COARSENING_SCALE; j <= min(Y - 1, maxy + COARSENING_SCALE); j++) {
                    //if(layer == 1 && x == 3 && y == 18)
                    //    printf("GPU %d %d: %.2lf\n", i, j, cellResource(layer * N * N + i * N + j, wires, fixed, vias, capacity, N, LAYER));
                    sum += cell_resource[layer * N * N + i * N + j];
                }
            ans += 1.0 * COARSENING_SCALE / max(0.1, sum / (maxx - minx + 1) / (min(Y - 1, maxy + COARSENING_SCALE) - miny - COARSENING_SCALE + 1));
            //if(layer == 1 && x == 3 && y == 18)
            //    printf("sum: %.2lf\n", sum);
            cost[layer * xsize * ysize + x * ysize + y] = 100 * ans;
        }
    } else {
        if(x + 1 < xsize) {
            float sum = 0;
            for(int i = minx; i <= maxx; i++)
                for(int j = miny; j <= maxy; j++)
                    sum += cell_resource[layer * N * N + j * N + i];
            ans += 1.0 * COARSENING_SCALE / max(0.1, sum / (maxx - minx + 1) / (maxy - miny + 1));
            sum = 0;
            for(int i = minx + COARSENING_SCALE; i <= min(X - 1, maxx + COARSENING_SCALE); i++)
                for(int j = miny; j <= maxy; j++)
                    sum += cell_resource[layer * N * N + j * N + i];
            ans += 1.0 * COARSENING_SCALE / max(0.1, sum / (min(X - 1, maxx + COARSENING_SCALE) - minx - COARSENING_SCALE + 1) / (maxy - miny + 1));
            cost[layer * xsize * ysize + x * ysize + y] = 100 * ans;

        }
    }
}

__global__ void calculateCoarseVia(float *cell_resource, int *coarseVia, int *wires, float *fixed, int *vias, float *capacity, int N, int LAYER, int xsize, int ysize, int X, int Y, int DIRECTION, int COARSENING_SCALE) {
    int layer = blockIdx.x / xsize, x = blockIdx.x % xsize, y = threadIdx.x;
    int minx = x * COARSENING_SCALE, maxx = min(X - 1, x * COARSENING_SCALE + COARSENING_SCALE - 1);
    int miny = y * COARSENING_SCALE, maxy = min(Y - 1, y * COARSENING_SCALE + COARSENING_SCALE - 1);
    if(layer + 1 < LAYER) {
        float sum = 0, ans = 0;
        for(int i = minx; i <= maxx; i++)
            for(int j = miny; j <= maxy; j++)
                if((layer & 1) ^ DIRECTION)
                    sum += cell_resource[layer * N * N + i * N + j];
                else
                    sum += cell_resource[layer * N * N + j * N + i];

        ans += 1.0 / max(0.1, sum / (maxx - minx + 1) / (maxy - miny + 1));    
        sum = 0;
        for(int i = minx; i <= maxx; i++)
            for(int j = miny; j <= maxy; j++)
                if(!(layer & 1) ^ DIRECTION)
                    sum += cell_resource[(layer + 1) * N * N + i * N + j];
                else
                    sum += cell_resource[(layer + 1) * N * N + j * N + i];
        ans += 1.0 / max(0.1, sum / (maxx - minx + 1) / (maxy - miny + 1));   
        coarseVia[layer * xsize * ysize + x * ysize + y] = 100 * ans;
    }
}

__global__ void initMap(dtype *dist, int *prev, int total, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        dist[idx] = INF;
        prev[idx] = idx % total;
        //for(int i = 0; i < N; i++)
        //    dist[idx + i * total] = INF, prev[idx + i * total] = idx;
    }
}

/*
__managed__ float minval, maxval = 0;
__managed__ unsigned long long allroutecost = 0;
#define debug -1
__global__ void traceBack(int *modifiedWire, int *modifiedVia, dtype *dist, int *prev, int *wires, int *vias, int *pins, int *routes, int *routesOffset, int *cudaPos, int netId, int N, int flag) {
    routes += routesOffset[netId];
    dtype minDist = INF;
    int p = -1, lef = -1, rig = -1, finished = 1;
    for(int i = 0, cur = 1; i < pins[0]; i++) {
        //if(netId == debug || debug == -2)
        //    printf("pin %d\n", i);
        for(int j = 1; j <= pins[cur]; j++) {
            int pos = pins[cur + j];
            //if(netId == debug || debug == -2)
            //    printf("access point %d=(%d,%d,%d) dist: %d\n", cudaPos[pos], cudaPos[pos] / N / N, cudaPos[pos] % (N * N) / N, cudaPos[pos] % N, dist[pos]);
            if(dist[pos] == INF) finished = 0;
            if(dist[pos] > 0 && dist[pos] < minDist) {
                minDist = dist[pos];
                p = pos;
                lef = cur + 1;
                rig = cur + pins[cur];
            }
        }
        cur += pins[cur] + 1;
    }
    if(flag && finished == 0)
        printf("REAL ERROR: DISCONNECTED NET%d\n", netId);
    if(p == -1) {
        //printf("WARNING: No pin is connected in this round. NET ID: %d\n", netId);
        return;
    }
    //if(netId == debug)
    //    printf("Start tracing result: %d %d %d\n", p / N / N, p % (N * N) / N, p % N);
    maxval = max(maxval, 1.0 * dist[p]);
    int expected = p;
    while(dist[expected] > 0)
        expected = prev[expected];
    while(dist[p] > 0) {
        //if(netId == debug || debug == -2)
        //    printf("%d %d (%d, %d, %d): %d; prev: %d %d (%d, %d, %d): %d\n", p, cudaPos[p], cudaPos[p] / N / N, cudaPos[p] % (N * N) / N, cudaPos[p] % N, dist[p], prev[p],  cudaPos[prev[p]], cudaPos[prev[p]] / N / N, cudaPos[prev[p]] % (N * N) / N, cudaPos[prev[p]] % N, dist[prev[p]]);
        int minp = cudaPos[p], maxp = cudaPos[prev[p]], pre = prev[p];
        if(minp > maxp) {
            int temp = minp;
            minp = maxp;
            maxp = temp;
        }
        int lmin = minp / N / N, lmax = maxp / N / N, x = minp % (N * N) / N, y = minp % N;
        if(lmin != lmax) {
            for(int i = lmin; i <= lmax; i++) {
                int idx = i * N * N + ((i - lmin) % 2 ? y * N + x : x * N + y);
                if(i < lmax) {
                    atomicAdd(vias + idx, 1);
                    routes[routes[0]++] = idx;
                    routes[routes[0]++] = -1;
                    modifiedVia[idx] = STAMP;
                }
                //if(idx != cudaPos[pre])
                //    dist[idx] = 0;
            }
        } else {
            routes[routes[0]++] = minp;
            routes[routes[0]++] = maxp - minp;
            for(int i = minp; i <= maxp; i++) {
                if(i < maxp) {
                    atomicAdd(wires + i, 1);
                    modifiedWire[i] = STAMP;
                }
                //if(i != cudaPos[pre])
                //    dist[i] = 0;
            }
        }
        //if(dist[pre] + sum != distp)
        //    printf("ERROR: INCONSISTENT COST; net %d sum %d\n", netId, sum);
        dist[p] = 0;
        p = pre;
    }
    if(expected != p)
        printf("netid %d: TRACEBACK ERROR\n", netId);
    //printf("Expected tracing result: %d %d %d\n", expected / N / N, expected % (N * N) / N, expected % N);
    //if(netId == debug)
    //    printf("Final tracing result: %d %d %d\n", p / N / N, p % (N * N) / N, p % N);
    for(int i = lef; i <= rig; i++) {
        dist[pins[i]] = 0;
        //if(debug == netId)
        //    printf("setting 0 distance: %d=%d,%d,%d\n", pins[i], pins[i] / N / N, pins[i] / N % N, pins[i] % N);
    }
    if(routes[0] > routesOffset[netId + 1] - routesOffset[netId])
        printf("%d **ERROR: ROUTE_LEN_PER_PIN INSUFFICENT; \n", routes[0]);
}
*/

__global__ void calculateWireCost(dtype *cost, float *wireDist, float *fixed, float *fixedLength, int *wires, int *vias, float *capacity, float *unitShortCost, float logisticSlope, int N, int LAYER) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(!(idx < LAYER * N * N && idx % N + 1 < N)) return;
    if(capacity[idx] < 0.01) {
        cost[idx] = INF;
        return;
    }
    int expectedLen = (fixed[idx] * fixedLength[idx] + wires[idx] * wireDist[idx]) / capacity[idx];
    float remain = capacity[idx] - (fixed[idx] + wires[idx] + 1 + twoCellsViaUsage(idx, vias, N, LAYER));
    //if(idx == 2 * N * N + 63)
    //    for(int i = 0; i < 9; i++)
    //    printf("%d unit %.2lf\n", i, unitShortCost[i]);
    float result = wireDist[idx] + expectedLen / (1.0 + exp(logisticSlope * remain)) * unitShortCost[idx / N / N];
    int result_r = static_cast<int>(result);
    cost[idx] = (result_r > MAX_COST ? MAX_COST : result_r);
}

__global__ void calculateViaCost(int *wires, float *fixed, float *capacity, dtype *viaCost, float unitViaMultiplier, float unitViaCost, float logisticSlope, int N, int LAYER) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //if(idx >= viaLen) return;
    //idx = ids[idx];
    //if(modified[idx] != STAMP) return;
    if(idx >= (LAYER - 1) * N * N) return;
    int layer = idx / N / N + 1, y = idx % (N * N) / N, x = idx % N;
    float result = unitViaCost * (unitViaMultiplier + inCellViaCost(idx, wires, fixed, capacity, logisticSlope, N) + inCellViaCost(layer * N * N + x * N + y, wires, fixed, capacity, logisticSlope, N));
    //result /= 100;
    int result_r = static_cast<int>(result);
    viaCost[idx] = (result_r > MAX_COST ? MAX_COST : result_r);
}

__global__ void setStartCells(dtype *dist, int *pins, int N, int T) {
    pins += blockIdx.x * N + 2;
    dist += blockIdx.x * T;
    for(int i = 1; i <= pins[0]; i++) {
        dist[pins[i]] = 0;
    }
}

__global__ void markUnrouteUsage(int *pins, int *vias, int cnt) {
    int cur = blockIdx.x * blockDim.x + threadIdx.x;
    if(cur < cnt) atomicAdd(vias + pins[cur], 1);
}

__global__ void markOverflowWires(const float *capacity, int *wires, int *vias, float *fixed, int *isOverflow, int N, int LAYER) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < LAYER * N * N && idx % N + 1 < N) 
        isOverflow[idx] = (wires[idx] + fixed[idx] + twoCellsViaUsage(idx, vias, N, LAYER) > capacity[idx]);
}

__global__ void markOverflowVias(const float *capacity, int *wires, int *vias, float *fixed, int *isOverflow, int N, int LAYER) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= (LAYER - 1) * N * N) return;
    int layer = idx / N / N + 1, y = idx % (N * N) / N, x = idx % N;
    int upper_idx = layer * N * N + x * N + y;
    isOverflow[idx] = (inCellUsedArea(idx, wires, fixed, N) > capacity[idx] || inCellUsedArea(upper_idx, wires, fixed, N) > capacity[upper_idx]);
}

__global__ void markOverflowNets(int *isOverflowVia, int *isOverflowWire, int *isOverflowNet, int *routes, int *routesOffset, int NET_NUM) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= NET_NUM) return;
    routes += routesOffset[idx];
    isOverflowNet[idx] = 0;
    if(routes[0] == -1) {
        // printf("resolving failed net from pattern routing. netId: %d\n", idx);
        isOverflowNet[idx] = 1;
        return;
    }
    //if(routes[0] == 0) return;
    for(int i = 1; i < routes[0]; i += 2) if(routes[i + 1] == -1) {
        if(isOverflowVia[routes[i]]) {
            isOverflowNet[idx] = 1;
            return;
        }
    } else {
        for(int j = 0; j < routes[i + 1]; j++)
            if(isOverflowWire[routes[i] + j]) {
                isOverflowNet[idx] = 1;
                return;
            }
    }
}

__global__ void commit(int *routes, int *routesOffset, int *wires, int *vias, int NET_NUM) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= NET_NUM) return;
    routes += routesOffset[idx];
    for(int i = 1; i < routes[0]; i += 2) if(routes[i + 1] > 0) {
        for(int j = 0; j < routes[i + 1]; j++)
            atomicAdd(wires + routes[i] + j, 1);
    } else
        atomicAdd(vias + routes[i], 1);
}

__global__ void ripupOverflowNets(int *isOverflowNet, int *routes, int *routesOffset, int *wires, int *vias, int NET_NUM) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= NET_NUM || isOverflowNet[idx] == 0) return;
    routes += routesOffset[idx];
    for(int i = 1; i < routes[0]; i += 2) if(routes[i + 1] > 0) {
        for(int j = 0; j < routes[i + 1]; j++)
            atomicAdd(wires + routes[i] + j, -1);
    } else
        atomicAdd(vias + routes[i], -1);
    routes[0] = 1;
}

__global__ void getWires(int N, int LAYER, int *wires, int *ids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int layer = idx / (N * N), x = idx / N % N, y = idx % N;
    // bool needsUpdate = false;
    for(int dl = -1; dl <= 1; dl++) if(0 <= layer + dl && layer + dl < LAYER)
        for(int dx = -1; dx <= 1; dx++) if(0 <= x + dx && x + dx < N)
            for(int dy = -1; dy <= 1; dy++) if(0 <= y + dy && y + dy < N)
                if(wires[(layer + dl) * N * N + (x + dx) * N + (y + dy)] == STAMP) {
                    ids[atomicAdd(&wireLen, 1)] = idx;
                    return;
                }
}

__global__ void getVias(int N, int LAYER, int *vias, int *ids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int layer = idx / (N * N), x = idx / N % N, y = idx % N;
    // bool needsUpdate = false;
    for(int dl = -1; dl <= 1; dl++) if(0 <= layer + dl && layer + dl < LAYER)
        for(int dx = -1; dx <= 1; dx++) if(0 <= x + dx && x + dx < N)
            for(int dy = -1; dy <= 1; dy++) if(0 <= y + dy && y + dy < N)
                if(vias[(layer + dl) * N * N + (x + dx) * N + (y + dy)] == STAMP) {
                    ids[atomicAdd(&viaLen, 1)] = idx;
                    return;
                }
}

__global__ void calculateCostSum(int LAYER, int N, int *cost, int64_t *costSum) {
    extern __shared__ int64_t sum[];
    int a = threadIdx.x << 1, b = threadIdx.x << 1 | 1;
    for(int i = 1; i < LAYER; i++) {
        sum[a] = cost[i * N * N + blockIdx.x * N + N - 1 - a];
        sum[b] = cost[i * N * N + blockIdx.x * N + N - 1 - b];
        __syncthreads();
        for(int d = 0; (1 << d) < N; d++) {
            if(a >> d & 1)
                sum[a] += sum[(a >> d << d) - 1];
            if(b >> d & 1)
                sum[b] += sum[(b >> d << d) - 1];
            __syncthreads();
        }
        costSum[i * N * N + blockIdx.x * N + N - 1 - a] = sum[a];
        costSum[i * N * N + blockIdx.x * N + N - 1 - b] = sum[b];
        __syncthreads();
    }
}

__global__ void printWires(int *wires, int LAYER, int N) {
    for(int i = 0; i < LAYER; i++)
        for(int j = 0; j < N; j++)
            for(int k = 0; k < N; k++)
                if(wires[i * N * N + j * N + k]) printf("%d %d  ", i * N * N + j * N + k, wires[i * N * N + j * N + k]);
}

__global__ void output(int *wires, int *vias, int LAYER, int X, int Y, int N, int DIRECTION) {
    for(int i = 2; i < 3; i++)
        for(int j = 0; j < X; j++)
            for(int k = 0; k < Y; k++)
                if((i & 1) ^ DIRECTION) {
                    if(k + 1 < Y) printf("%d %d %d: %d\n", i, j, k, wires[i * N * N + j * N + k]);
                } else {
                    if(j + 1 < X) printf("%d %d %d: %d\n", i, j, k, wires[i * N * N + j * N + k]);
                }
        /*if((i & 1) ^ DIRECTION)
            for(int j = 0; j < X; j++)
                for(int k = 0; k < Y - 1; k++)
                    printf("%d %d %d: %d\n", i, j, k, wires[i * N * N + j * N + k]);
        else
            for(int j = 0; j < Y; j++)
                for(int k = 0; k < X - 1; k++)
                    printf("%d %d %d: %d\n", i, j, k, wires[i * N * N + j * N + k]);*/
}
/*
__global__ void generateBatch(int n, int *minx, int *maxx, int *miny, int *maxy, int *res, int *siz, int *used, int *flag) {
    //printf("thid %d %d\n", threadIdx.x, n);
    extern __shared__ int selected[];
    for(int i = threadIdx.x; i < n; i += blockDim.x) used[i] = 0;
    if(threadIdx.x == 0) selected[1] = 0;
    __syncthreads();
    while(selected[1] < n) {
        int cur = selected[1];
        if(threadIdx.x == 0) selected[0] = n;
        __syncthreads();
        for(int i = threadIdx.x; i < n; i += blockDim.x) {
            if(used[i] == 0) atomicMin(selected, i);
            flag[i] = 0;
        }
        __syncthreads();
        #define allowed_overlap 2
        while(selected[0] < n) {
            int t = selected[0];
            __syncthreads();
            if(threadIdx.x == 0) res[cur++] = t, used[t] = 1, selected[0] = n;//, printf("%d ", t);
            __syncthreads();
            for(int offset = 0; offset < n; offset += blockDim.x) {
                int i = threadIdx.x + offset;
                if(i < n) {
                    if(used[i] == 0 && flag[i] == 0) {
                        if(maxx[i] + allowed_overlap < minx[t] || 
                        maxx[t] + allowed_overlap < minx[i] ||
                        maxy[i] + allowed_overlap < miny[t] ||
                        maxy[t] + allowed_overlap < miny[i]) atomicMin(selected, i);
                        else
                            flag[i] = 1;
                    }
                }
                __syncthreads();
                if(selected[0] < n) break;
            }
        }
        //if(threadIdx.x == 0) printf("\n");
        if(threadIdx.x == 0)
            siz[++siz[0]] = cur - selected[1], selected[1] = cur;
        __syncthreads();
    }
}
*/
__global__ void generateBatch(int n, int *minx, int *maxx, int *miny, int *maxy, int *res, int *siz, int *used) {
    extern __shared__ int shared[];
    if(threadIdx.x == 0) shared[0] = 0;
    __syncthreads();
    while(shared[0] < n) {
        if(threadIdx.x == 0) shared[2] = 0;
        __syncthreads();
        for(int i = 0; i < n; i++) if(used[i] == 0) {
            if(threadIdx.x == 0) shared[1] = 0;
            __syncthreads();
            #define allowed_overlap 2
            if(threadIdx.x < shared[2]) {
                if (maxx[i] + allowed_overlap < minx[res[threadIdx.x + shared[0]]] || 
                    maxx[res[threadIdx.x + shared[0]]] + allowed_overlap < minx[i] ||
                    maxy[i] + allowed_overlap < miny[res[threadIdx.x + shared[0]]] ||
                    maxy[res[threadIdx.x + shared[0]]] + allowed_overlap < miny[i]) {}
                else
                    shared[1] = 1;
            }
            __syncthreads();
            if(threadIdx.x == 0 && shared[1] == 0)
                res[shared[0] + shared[2]++] = i, used[i] = 1;
            __syncthreads();
        }
        if(threadIdx.x == 0) {
            siz[++siz[0]] = shared[2], shared[0] += shared[2];
            if(shared[2] > blockDim.x) printf("ERROR in kernel\n");
        }
        __syncthreads();
    }
}

void GPURouter::route(vector<GrNet> &nets, int iter) {
    logger.info("GPU Routing start... DIRECTION: %d", DIRECTION);

    double prtime = 0, prpreparetime = 0, batchgentime = 0, timer2 = 0;

    vector<int> netsToRoute;
    if(iter > 0) {
        logger.info("Maze Routing...");
        ripupOverflowNets<<<BLOCK_NUMBER(NET_NUM), BLOCK_SIZE>>> (isOverflowNet, routes, routesOffset, wires, vias, NET_NUM);
        cudaDeviceSynchronize();
        for(size_t netId = 0; netId < nets.size(); netId++) 
            if(isOverflowNet[netId] && !nets[netId].noroute) {
                netsToRoute.emplace_back(netId);
                // if(nets[netId].noroute) printf("ERROR: net %d is noroute but overflow\n", netId);
            }
    } else {
        logger.info("Pattern Routing...");
        // int cnt = 0;
        for(int i = 0; i < nets.size(); i++) 
            if(!nets[i].noroute) netsToRoute.emplace_back(i);
            /*else {
                auto &temp = nets[i].getPins();
                for(auto e : temp)
                    for(auto f : e)
                        allpins[cnt++] = f;
                if(cnt > MAX_BATCH_SIZE * MAX_PIN_SIZE_PER_NET) std::cerr << "ERROR!\n";
            }
        markUnrouteUsage<<<BLOCK_NUMBER(cnt), BLOCK_SIZE>>> (allpins, vias, cnt);
        cudaDeviceSynchronize();*/
    }
    vector<int> batchSizes;
    {
        double t = clock();
        constexpr bool check_vis_correctness = false;
        std::vector<int> s = netsToRoute;
        int margin = 0; // NOTE: margin == 0 is okay for PR, but do not check for MR
        //int margin = iter ? 0 : 2;
        std::sort(s.begin(), s.end(), [&] (int l, int r) {
            int area_l = nets[l].area();
            int area_r = nets[r].area();
            if (area_l == area_r) {
                return l > r;
            }
            return area_l > area_r;
            //return nets[l].area() * 1.0 / nets[l].getPins().size() > nets[r].area() * 1.0 / nets[r].getPins().size();
        });
        // std::sort(s.begin(), s.end(), [&] (int l, int r) {
        //     int hpwl_l = nets[l].hpwl();
        //     int hpwl_r = nets[r].hpwl();
        //     if (hpwl_l == hpwl_r) return l > r;
        //     return hpwl_l < hpwl_r;
        // });
        //for(int i = 0; i < s.size(); i++)
        //    printf("%d, [%d, %d] [%d, %d]\n", s[i], nets[s[i]].lowerx, nets[s[i]].upperx, nets[s[i]].lowery, nets[s[i]].uppery);
        /*int n = netsToRoute.size();
        nt *minx, *maxx, *miny, *maxy, *res, *siz, *used, *flag;
        cudaMallocManaged(&minx, n * sizeof(int));
        cudaMallocManaged(&maxx, n * sizeof(int));
        cudaMallocManaged(&miny, n * sizeof(int));
        cudaMallocManaged(&maxy, n * sizeof(int));
        cudaMallocManaged(&res, n * sizeof(int));
        cudaMallocManaged(&siz, (n + 1) * sizeof(int));
        cudaMalloc(&used, n * sizeof(int));
        cudaMalloc(&flag, n * sizeof(int));

        siz[0] = 0;
        for(int i = 0; i < n; i++) {
            auto &net = nets[s[i]];
            minx[i] = net.lowerx;
            maxx[i] = net.upperx;
            miny[i] = net.lowery;
            maxy[i] = net.uppery;
        }

        generateBatch<<<1, 1024, 3 * sizeof(int)>>> (n, minx, maxx, miny, maxy, res, siz, used);
        cudaDeviceSynchronize();

        for(int i = 0; i < n; i++)
            netsToRoute[i] = s[res[i]];
        for(int i = 1; i <= siz[0]; i++)
            batchSizes.emplace_back(siz[i]);*/
        /*int startpos = 0;
        for(auto e : batchSizes) {
            for(int i = startpos; i < startpos + e; i++)
                for(int j= startpos; j < i; j++) {
                    int n1 = netsToRoute[i], n2 = netsToRoute[j];
                    if(maxx[n1] + 2 < minx[n2] ||
                       maxx[n2] + 2 < minx[n1] ||
                       maxy[n1] + 2 < miny[n2] ||
                       maxy[n2] + 2 < miny[n2]) {}
                    else
                        printf("ERROR: overlap %d %d!\n", i, j);
                }
                
            startpos += e;
        }*/
        //static std::vector<std::vector<int>> vis(2000, std::vector<int> (2000, 0));
        /*RangedBitset<1600> test;
        test.set(10, 128, 1);
        printf("%d\n", test.check_all(9, 128, 1));
        printf("%d\n", test.check_all(13, 128, 1));
        printf("%d\n", test.check_all(9, 128, 0));
        printf("%d\n", test.check_all(13, 128, 0));
        exit(0);*/
        // const int LEN = 10;
        if (vis.size() == 0) {
            vis.resize(2000, std::vector<short>(2000, 0));
            visLL.resize(2000, std::vector<short>(2000, 0));
            visRR.resize(2000, std::vector<short>(2000, 0));
        }
        auto noConflict = [&] (int netId) {
            const auto &net = nets[netId];
            //int blockL = net.lowery / LEN, blockR = net.uppery / LEN;
            //if(blockL == blockR) {
                // for(int i = net.lowerx; i <= net.upperx; i++) 
                //     for(int j = net.lowery; j <= net.uppery; j++)
                //         if(vis[i][j]) return false;
                for(int i = max(0, net.lowerx - margin); i <= net.upperx + margin; i++) 
                    for(int j = max(0, net.lowery - margin); j <= net.uppery + margin; j++)
                        if(vis[i][j]) return false;
            /*} else {
                for(int i = net.lowerx; i <= net.upperx; i++) {
                    if(visLL[i][net.lowery] || visRR[i][net.uppery]) return false;
                    for(int j = blockL + 1; j < blockR; j++)
                        if(visLL[i][j * LEN]) return false;
                }
            }*/
            return true;
        };
        auto insert = [&] (int netId) {
            //double t = clock();
            const auto &net = nets[netId];           
            int xl = max(0, net.lowerx - margin), xr = net.upperx + margin;
            int yl = max(0, net.lowery - margin), yr = net.uppery + margin;
            //int blockL = yl / LEN - (yl % LEN == 0), blockR = yr / LEN + (yr % LEN == 0);
            for(int i = xl; i <= xr; i++) {
                for(int j = yl; j <= yr; j++) {
                    if (check_vis_correctness) {
                        vis[i][j] += 1;
                    } else {
                        vis[i][j] = 1;
                    }
                }
                /*for(int j = yl; j <= (blockR - 1) * LEN; j++)
                    visLL[i][j] = 1;
                for(int j = (blockL + 1) * LEN; j <= yr; j++)
                    visRR[i][j] = 1;*/
            }
            //modify_cnt += clock() - t;
        };        
        auto remove = [&] (int netId) {
            const auto &net = nets[netId];           
            int xl = max(0, net.lowerx - margin), xr = net.upperx + margin;
            int yl = max(0, net.lowery - margin), yr = net.uppery + margin;
            //int blockL = yl / LEN - (yl % LEN == 0), blockR = yr / LEN + (yr % LEN == 0);
            for(int i = xl; i <= xr; i++) {
                for(int j = yl; j <= yr; j++) {
                    if (check_vis_correctness) {
                        vis[i][j] -= 1;
                    } else {
                        vis[i][j] = 0;
                    }
                }
                /*for(int j = yl; j <= (blockR - 1) * LEN; j++)
                    visLL[i][j] = 0;
                for(int j = (blockL + 1) * LEN; j <= yr; j++)
                    visRR[i][j] = 0;*/
            }
        };
        auto checkVis = [&] () {
            for (int i = 0; i < 2000; i++) {
                for (int j = 0; j < 2000; j++) {
                    if (vis[i][j] > 1) {
                        std::cout << "ERROR in batch generation" << std::endl;
                        return;
                    }
                }
            }
        };
        netsToRoute.clear();
        int lastUnroute = 0;
        // FIXME: if batch generation allows bbox overlap, data race would happen in PR kernel
        while(netsToRoute.size() < s.size()) {
            int sz = netsToRoute.size();
            // int last = lastUnroute;
            int cnt = 0;
            for(size_t i = lastUnroute; i < s.size(); i++) if(s[i] != -1) {
                bool no_conflict = 1;
                no_conflict = noConflict(s[i]);
                // older version, allow bbox overlap but much faster
                // if(nets[s[i]].area() * 2 < netsToRoute.size() - sz)
                //     no_conflict = noConflict(s[i]);
                // else for(size_t j = sz; j < netsToRoute.size(); j++) {
                //     const auto &a = nets[s[i]];
                //     const auto &b = nets[netsToRoute[j]];
                //     if(!(a.upperx + margin < b.lowerx || b.upperx + margin < a.lowerx || 
                //          a.uppery + margin < b.lowery || b.uppery + margin < a.lowery)) {
                //         no_conflict = 0;
                //         break;
                //     }
                // }
                if(no_conflict)
                    netsToRoute.emplace_back(s[i]), insert(s[i]), s[i] = -1, cnt = 0;
                else
                    cnt++;
                //if(cnt >= 100) break;
                if(iter && netsToRoute.size() - sz == MAX_BATCH_SIZE) break;
                if(!iter && netsToRoute.size() - sz == 250) break;
            }
            while(lastUnroute < s.size() && s[lastUnroute] == -1) lastUnroute++;
            if (check_vis_correctness) checkVis();
            for(int i = sz; i < netsToRoute.size(); i++)
                remove(netsToRoute[i]);
            batchSizes.emplace_back(netsToRoute.size() - sz);
        }
        batchgentime += clock() - t;
        logger.info("INFO: Batch Generation Time %.4f", batchgentime / CLOCKS_PER_SEC);
    }
    reverse(batchSizes.begin(), batchSizes.end());
    reverse(netsToRoute.begin(), netsToRoute.end());
    logger.info("number of batches: %d;  number of nets: %d", batchSizes.size(), netsToRoute.size());
    int startpos = 0; 
    int sumofpins = 0;
    // int lowestpins = 0;
    // int batch_cnt = 0;
    double mrtime = 0, costtime = 0;
    for(auto batchSize : batchSizes) {
        calculateWireCost<<<BLOCK_NUMBER(LAYER * N * N), BLOCK_SIZE>>> (cost, wireDist, fixed, fixedLength, wires, vias, capacity, unitShortCostDiscounted, logisticSlope, N, LAYER);
        calculateViaCost<<<BLOCK_NUMBER((LAYER - 1) * N * N), BLOCK_SIZE>>> (wires, fixed, capacity, viaCost, unitViaMultiplier, unitViaCost, logisticSlope, N, LAYER);   
        calculateCostSum<<<N, N / 2, N * sizeof(int64_t)>>> (LAYER, N, cost, costSum);
        if(iter == 0) {
            int offset = batchSize;
            int gbPinOffset = batchSize;
            if(points == nullptr) {
                cudaMallocManaged(&points, 20000000 * sizeof(int));
                cudaMallocManaged(&gbpoints, 10000000 * sizeof(int));
            }
            // double prepare_part_time = 0;
            double prepare_detailed_time = 0;
            double t = clock();

            //if(startpos == 11394)
                //logger.info("net id %d", netsToRoute[startpos]);
            //netsToRoute[startpos] = 8026;
            for(int i = 0; i < batchSize; i++) {
                int netId = netsToRoute[startpos + i];
                points[i] = offset;
                gbpoints[i] = gbPinOffset;
                offset += prepare(prepare_detailed_time, nets[netId], points + offset, routesOffsetCPU[netId], gbpoints + gbPinOffset, gbPinOffset, X, Y, N, LAYER, DIRECTION);
            }
            if(offset > 20000000 || gbPinOffset > 10000000)
                logger.error("ERROR offset %d %d", offset, gbPinOffset);
            timer2 += prepare_detailed_time;
            prpreparetime += clock() - t;
            t = clock();
            //cudaMemcpy(cuda_points, points, offset * sizeof(int), cudaMemcpyHostToDevice);
            patternRoute(points, batchSize, costSum, viaCost, dist, prev, wires, vias, routes, gbpoints, gbpinRoutes, X, Y, N, LAYER, DIRECTION);
            
            prtime += clock() - t;
        } else {
            calculateCellResource<<<BLOCK_NUMBER(LAYER * N * N), BLOCK_SIZE>>> (cell_resource, wires, fixed, vias, capacity, N, LAYER, LAYER * N * N);
            calculateCoarseCost<<<LAYER * cgxsize, cgysize>>> (cell_resource, gpuMR.cost, wires, fixed, vias, capacity, N, cgxsize, cgysize, X, Y, LAYER, DIRECTION, COARSENING_SCALE);    
            calculateCoarseVia<<<(LAYER - 1) * cgxsize, cgysize>>> (cell_resource, gpuMR.via, wires, fixed, vias, capacity, N, LAYER, cgxsize, cgysize, X, Y, DIRECTION, COARSENING_SCALE);
            gpuMR.run(DIRECTION, iter);
            int pin10 = 0;
            for(int i = 0; i < batchSize; i++) {
                int netId = netsToRoute[startpos + i], cur = 0, *pins = allpins + i * MAX_PIN_SIZE_PER_NET;
                auto& vec = nets[netId].getPins();
                pins[cur++] = routesOffsetCPU[netId];
                pins[cur++] = vec.size();
                sumofpins += vec.size();
                if(vec.size() <= 10) pin10++;
                for(auto a : vec) {
                    pins[cur++] = a.size();
                    for(auto b : a) {
                        pins[cur++] = b;
                        //if(b / N / N >= 5)
                            //std::cerr << "upper lower pins exist. " << b / N / N << std::endl;
                    }
                }
                if(cur >= MAX_PIN_SIZE_PER_NET) {
                    std::cerr << "ERROR: NOT ENOUGH FOR PINS cur: " << cur << " MAX_PIN_SIZE_PER_NET: " << MAX_PIN_SIZE_PER_NET << std::endl;
                    exit(-1);
                }
            }
            //printf("%d / %d = %.2lf\n", pin10, batchSize, pin10 * 1.0 / batchSize);
            initMap<<<BLOCK_NUMBER(batchSize * LAYER * N * N), BLOCK_SIZE>>> (dist, prev, N * N * LAYER, batchSize * N * N * LAYER);
            setStartCells<<<batchSize, 1>>> (dist, allpins, MAX_PIN_SIZE_PER_NET, N * N * LAYER);
            double t = clock();
            gpuMR.getResults(costtime, costSum, allpins, dist, prev, cost, viaCost, wires, vias, routes, N, COARSENING_SCALE, DIRECTION, batchSize, netsToRoute[startpos]);
            cudaDeviceSynchronize();
            mrtime += (clock() - t) * 1.0 / CLOCKS_PER_SEC;
        }
        //std::cerr << "pins: " << sumofpins << ' ' << 1.0 * sumofpins / netsToRoute.size() << std::endl;
        startpos += batchSize;
    }
    if(!iter) {
        logger.info("INFO: PR Prepare Time %.4f", prpreparetime / CLOCKS_PER_SEC);
        logger.info("INFO: PR Kernel Time %.4f", prtime / CLOCKS_PER_SEC);
    } else {
        logger.info("INFO: MR Func time %.4f", mrtime);
        logger.info("INFO: MR Cost calc time %.4f", costtime / CLOCKS_PER_SEC);
    }
    //output<<<1, 1>>> (wires, vias, LAYER, X, Y, N, DIRECTION);
    
    //gpuMR.query();
    //if(iter == db::setting.rrrIterLimit - 1) {
    if(1) {
        markOverflowWires<<<BLOCK_NUMBER(LAYER * N * N), BLOCK_SIZE>>> (capacity, wires, vias, fixed, isOverflowWire, N, LAYER);
        markOverflowVias<<<BLOCK_NUMBER((LAYER - 1) * N * N), BLOCK_SIZE>>> (capacity, wires, vias, fixed, isOverflowVia, N, LAYER);
        markOverflowNets<<<BLOCK_NUMBER(NET_NUM), BLOCK_SIZE>>> (isOverflowVia, isOverflowWire, isOverflowNet, routes, routesOffset, NET_NUM);
        cudaDeviceSynchronize();
        int cnt = 0;
        for(size_t netId = 0; netId < nets.size(); netId++) 
            if(isOverflowNet[netId]) {
                cnt++;
                if(nets[netId].noroute) printf("ERROR: net %zu is noroute but overflow\n", netId);
            }
        logger.info("Final Overflow Net Number: %d", cnt);
        numOvflNets = cnt;
    }
}

void GPURouter::setFromNets(vector<GrNet> &nets, int numPlPin_) {
    NET_NUM = nets.size();
    pinNumCPU = new int[NET_NUM];
    routesOffsetCPU = new int[NET_NUM + 1];
    routesOffsetCPU[0] = 0;
    for(int i = 0; i < NET_NUM; i++) {
        pinNumCPU[i] = nets[i].getPins().size();
        routesOffsetCPU[i + 1] = pinNumCPU[i] * MAX_ROUTE_LEN_PER_PIN + routesOffsetCPU[i];
    }
    cudaMallocManaged(&isOverflowNet, NET_NUM * sizeof(int));
    cudaMalloc(&routes, routesOffsetCPU[NET_NUM] * sizeof(int));
    cudaMemset(routes, 0, routesOffsetCPU[NET_NUM] * sizeof(int));
    logger.info("total routes: %d", routesOffsetCPU[NET_NUM]);
    cudaMalloc(&routesOffset, (NET_NUM + 1) * sizeof(int));
    cudaMemcpy(routesOffset, routesOffsetCPU, sizeof(int) * (NET_NUM + 1), cudaMemcpyHostToDevice);

    for (int netId = nets.size() - 1; netId >= 0; netId--) {
        auto& lastGrNet = nets[netId];
        if (lastGrNet.pin2gbpinId.size() > 0) {
            numGbPin = lastGrNet.pin2gbpinId[lastGrNet.pin2gbpinId.size() - 1] + 1;
            break;
        }
    }
    // numRoutes, routeId, routeId, routeId, routeId, numVias
    cudaMalloc(&gbpinRoutes, 6 * numGbPin * sizeof(int));
    cudaMemset(gbpinRoutes, 0, 6 * numGbPin * sizeof(int));

    // number of pins in placement database
    numPlPin = numPlPin_;
    std::vector<int> gbpin2netIdCPU(numGbPin);
    std::vector<int> plPinId2gbPinIdCPU(numPlPin, -1);
    for (int netId = 0; netId < nets.size(); netId ++) {
        for (int pinId = 0; pinId < nets[netId].getPins().size(); pinId++) {
            int gbpinId = nets[netId].pin2gbpinId[pinId];
            gbpin2netIdCPU[gbpinId] = netId;
            std::vector<int>& gpdbPinIds = nets[netId].pin2gpdbPinIds[pinId];
            for (int gpdbPinId : gpdbPinIds) {
                plPinId2gbPinIdCPU[gpdbPinId] = gbpinId;
            }
        }
    }
    cudaMalloc(&gbpin2netId, numGbPin * sizeof(int));
    cudaMemcpy(gbpin2netId, gbpin2netIdCPU.data(), numGbPin * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&plPinId2gbPinId, numPlPin * sizeof(int));
    cudaMemcpy(plPinId2gbPinId, plPinId2gbPinIdCPU.data(), numPlPin * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
}

void GPURouter::setToNets(vector<GrNet> &nets) {
    int *routesCPU = new int[routesOffsetCPU[NET_NUM]];
    int mx = 0;
    cudaMemcpy(routesCPU, routes, sizeof(int) * routesOffsetCPU[NET_NUM], cudaMemcpyDeviceToHost);
    int num_net_use_too_many_route = 0;
    for(size_t netId = 0; netId < nets.size(); netId++) {
        vector<int> wires, vias;
        int *routesSub = routesCPU + routesOffsetCPU[netId];
        mx = max(mx, routesSub[0] / pinNumCPU[netId]);
        if(routesSub[0] > routesOffsetCPU[netId + 1] - routesOffsetCPU[netId]) {
            num_net_use_too_many_route++;
            // std::cerr << "ERROR: too many routesSub! Please set MAX_ROUTE_LEN_PER_PIN larger than " << routesSub[0] / pinNumCPU[netId] << std::endl;
        }
        for(int i = 1; i < routesSub[0]; i += 2) {
            if (routesSub[i + 1] > 0) {
                wires.emplace_back(routesSub[i]);
                wires.emplace_back(routesSub[i + 1]);
            } else if (routesSub[i + 1] == -1) {
                vias.emplace_back(routesSub[i]);
            }
        }
        nets[netId].setWires(wires);
        nets[netId].setVias(vias);
        //nets[netId].useExtraVias();
    };
    logger.info("max routes[0] = %d", mx);
    if (num_net_use_too_many_route) {
        std::cerr << "ERROR: there are " << num_net_use_too_many_route << " nets use too many route segments!";
        std::cerr << " Please set MAX_ROUTE_LEN_PER_PIN (" << MAX_ROUTE_LEN_PER_PIN << ") larger than " << mx << std::endl;
    }
    delete[] routesCPU;
}

}  // namespace gr