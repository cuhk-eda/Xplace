#include <iostream>
#include <map>

#include "PatternRoute.h"

namespace gr {

__managed__ int debug;

#define INF 1000000000
__global__ void initMap(int *map) { map[blockIdx.x * blockDim.x + threadIdx.x] = INF; }
template <typename T>
__device__ void inline cudaSwap(T &a, T &b) {
    T c(a);
    a = b;
    b = c;
}

__device__ void viaSweep(int *map, int *prev, int *viaCost, int x, int y, int LAYER, int N, int DIRECTION) {
    map += 4 * LAYER * N * N;
    prev += 4 * LAYER * N * N;
    for (int i = 1; i < LAYER; i++) {
        int cur = i * N * N + (((i & 1) ^ DIRECTION) ? x * N + y : y * N + x);
        prev[cur + LAYER * N * N] = cur;
    }
    for (int i = 2; i < LAYER; i++) {
        int cur = i * N * N + (((i & 1) ^ DIRECTION) ? x * N + y : y * N + x);
        int last = (i - 1) * N * N + (((i & 1) ^ DIRECTION) ? y * N + x : x * N + y);
        if (map[cur] > map[last] + viaCost[last])
            map[cur] = map[last] + viaCost[last], prev[cur] = prev[last], prev[cur + LAYER * N * N] = last;
    }
    for (int i = LAYER - 2; i >= 1; i--) {
        int cur = i * N * N + (((i & 1) ^ DIRECTION) ? x * N + y : y * N + x);
        int last = (i + 1) * N * N + (((i & 1) ^ DIRECTION) ? y * N + x : x * N + y);
        if (map[cur] > map[last] + viaCost[cur])
            map[cur] = map[last] + viaCost[cur], prev[cur] = prev[last], prev[cur + LAYER * N * N] = last;
    }
}
__global__ void cudaLshapePR(int *map,
                             int *points,
                             int64_t *wireCostSum,
                             int *viaCost,
                             int *prev,
                             int *wires,
                             int *vias,
                             int *routes,
                             int LAYER,
                             int N,
                             int DIRECTION) {
    // printf("%d offset %d\n", threadIdx.x, points[threadIdx.x]);
    points += points[blockIdx.x];
    routes += points[0];
    routes[0] = 1;
    int node_cnt = points[1];
    points += 2;
    for (int t = 6 * (node_cnt - 1); t >= 0; t -= 6) {
        int tox = points[t] / N, toy = points[t] % N, minlayer = points[t + 1], maxlayer = points[t + 2];
        /*if(points[t + 3] == -1)
            printf("no child\n");
        else
            printf("child!\n");*/
        for (int j = 1; j <= 3; j++)
            if (points[t + 2 + j] != -1) {
                int fromx = points[t + 2 + j] / N, fromy = points[t + 2 + j] % N;
                // printf("%d %d to %d %d\n", fromx, fromy, tox, toy);
                if (fromx != tox && fromy != toy) {
                    for (int i = 0; i < LAYER; i++) {
                        if ((i & 1) ^ DIRECTION)
                            map[4 * LAYER * N * N + i * N * N + fromx * N + toy] =
                                map[4 * LAYER * N * N + i * N * N + tox * N + fromy] = INF;
                        else
                            map[4 * LAYER * N * N + i * N * N + toy * N + fromx] =
                                map[4 * LAYER * N * N + i * N * N + fromy * N + tox] = INF;
                    }
                    for (int i = 1; i < LAYER; i++) {
                        int last = i * N * N + (((i & 1) ^ DIRECTION) ? fromx * N + fromy : fromy * N + fromx);
                        int cur = i * N * N + (((i & 1) ^ DIRECTION) ? fromx * N + toy : fromy * N + tox);
                        int cost =
                            last < cur ? wireCostSum[last] - wireCostSum[cur] : wireCostSum[cur] - wireCostSum[last];
                        if (map[4 * LAYER * N * N + cur] > map[last] + cost)
                            map[4 * LAYER * N * N + cur] = map[last] + cost, prev[4 * LAYER * N * N + cur] = last;
                    }
                    viaSweep(map, prev, viaCost, fromx, toy, LAYER, N, DIRECTION);
                    viaSweep(map, prev, viaCost, tox, fromy, LAYER, N, DIRECTION);
                    for (int i = 1; i < LAYER; i++) {
                        int last = i * N * N + (((i & 1) ^ DIRECTION) ? tox * N + fromy : toy * N + fromx);
                        int cur = i * N * N + (((i & 1) ^ DIRECTION) ? tox * N + toy : toy * N + tox);
                        int cost =
                            last < cur ? wireCostSum[last] - wireCostSum[cur] : wireCostSum[cur] - wireCostSum[last];
                        if (map[cur + j * LAYER * N * N] > map[4 * LAYER * N * N + last] + cost)
                            map[cur + j * LAYER * N * N] = map[4 * LAYER * N * N + last] + cost,
                                                      prev[cur + j * LAYER * N * N] = prev[4 * LAYER * N * N + last];
                    }
                } else if (fromx == tox) {
                    for (int i = 1; i < LAYER; i++)
                        if ((i & 1) ^ DIRECTION) {
                            int last = i * N * N + fromx * N + fromy, cur = i * N * N + tox * N + toy;
                            int cost = last < cur ? wireCostSum[last] - wireCostSum[cur]
                                                  : wireCostSum[cur] - wireCostSum[last];
                            if (map[cur + j * LAYER * N * N] > map[last] + cost)
                                map[cur + j * LAYER * N * N] = map[last] + cost, prev[cur + j * LAYER * N * N] = last;
                        }
                } else if (fromy == toy) {
                    for (int i = 1; i < LAYER; i++)
                        if (!(i & 1) ^ DIRECTION) {
                            int last = i * N * N + fromy * N + fromx, cur = i * N * N + toy * N + tox;
                            int cost = last < cur ? wireCostSum[last] - wireCostSum[cur]
                                                  : wireCostSum[cur] - wireCostSum[last];
                            if (map[cur + j * LAYER * N * N] > map[last] + cost)
                                map[cur + j * LAYER * N * N] = map[last] + cost, prev[cur + j * LAYER * N * N] = last;
                        }
                } else
                    printf("ERROR: points on the same location!\n");
            }
        if (debug) printf("\n");
        for (int lower = 0; lower < LAYER; lower++) {
            int viaSum = 0, minCost = INF, bestpos[3] = {0, 0, 0},
                costs[3] = {
                    points[t + 3] == -1 ? 0 : INF, points[t + 4] == -1 ? 0 : INF, points[t + 5] == -1 ? 0 : INF};
            for (int upper = lower; upper < LAYER; upper++) {
                int idx = upper * N * N + (((upper & 1) ^ DIRECTION) ? tox * N + toy : toy * N + tox);
                for (int j = 0; j < 3; j++)
                    if (points[t + 3 + j] != -1) {
                        if (map[idx + (j + 1) * LAYER * N * N] < costs[j])
                            costs[j] = map[idx + (j + 1) * LAYER * N * N], bestpos[j] = upper;
                    }
                // if(upper == 0)
                //     printf("cost[0] = %d\n", costs[0]);
                if (1LL * costs[0] + costs[1] + costs[2] < minCost) minCost = costs[0] + costs[1] + costs[2];
                if (minlayer == -1 || (lower <= minlayer && maxlayer <= upper)) {
                    // printf("upper %d viaSUm %d\n", upper, viaSum);
                    for (int i = lower; i <= upper; i++) {
                        int index = i * N * N + (((i & 1) ^ DIRECTION) ? tox * N + toy : toy * N + tox);
                        if (map[index] > minCost + viaSum)
                            map[index] = minCost + viaSum,
                            prev[index] =
                                (((lower * LAYER + upper) * LAYER + bestpos[2]) * LAYER + bestpos[1]) * LAYER +
                                bestpos[0];
                    }
                }
                viaSum += viaCost[idx];
            }
        }
        // for (int i = 0; i < LAYER; i++) {
        //     int index = i * N * N + (((i & 1) ^ DIRECTION) ? tox * N + toy : toy * N + tox);
        //     int lower = prev[index] / LAYER / LAYER / LAYER / LAYER,
        //         upper = prev[index] / LAYER / LAYER / LAYER % LAYER;
        //     // printf("layer %d, cost %d, lower %d, upper %d, bestpos %d\n", i, map[index], lower, upper, prev[index]
        //     %
        //     // LAYER);
        // }
    }
    int minTotalDist = INF;
    for (int i = 0; i < LAYER; i++) {
        int x = points[0] / N, y = points[0] % N;
        int idx = i * N * N + (((i & 1) ^ DIRECTION) ? x * N + y : y * N + x);
        if (map[idx] < minTotalDist) minTotalDist = map[idx], map[4 * LAYER * N * N + points[0]] = i;
    }
    if (minTotalDist == INF || minTotalDist < 0) {
        printf("INF is too small!\n");

        return;
    }
    for (int t = 0; t < 6 * node_cnt; t += 6) {
        int x = points[t] / N, y = points[t] % N, layer = map[4 * LAYER * N * N + points[t]];
        int idx = layer * N * N + (((layer & 1) ^ DIRECTION) ? x * N + y : y * N + x);
        int minlayer = prev[idx] / LAYER / LAYER / LAYER / LAYER, maxlayer = prev[idx] / LAYER / LAYER / LAYER % LAYER;
        // printf("%d [%d, %d] exit/min/max layer\n", layer, minlayer, maxlayer);
        for (int i = minlayer; i < maxlayer; i++) {
            int id = i * N * N + (((i & 1) ^ DIRECTION) ? x * N + y : y * N + x);
            routes[routes[0]++] = id;
            routes[routes[0]++] = -1;
            atomicAdd(vias + id, 1);
        }
        int bestpos = prev[idx];
        for (int j = 0; j < 3; j++) {
            int child = points[t + 3 + j];
            if (points[t + 3 + j] == -1) break;
            int entrylayer = bestpos % LAYER,
                entry = entrylayer * N * N + (((entrylayer & 1) ^ DIRECTION) ? x * N + y : y * N + x);
            // printf("(%d, %d, %d) ", entrylayer, x, y);
            bestpos /= LAYER;
            int last = prev[(j + 1) * LAYER * N * N + entry];
            int lastlayer = last / N / N, lastx = last / N % N, lasty = last % N;
            if (!(lastlayer & 1) ^ DIRECTION) cudaSwap(lastx, lasty);
            if (lastx != x && lasty != y) {
                int _x = ((entrylayer & 1) ^ DIRECTION) ? x : lastx;
                int _y = ((entrylayer & 1) ^ DIRECTION) ? lasty : y;
                int p1 = ((entrylayer & 1) ^ DIRECTION) ? entry - y + lasty : entry - x + lastx;
                int p2 = ((lastlayer & 1) ^ DIRECTION) ? last - lasty + y : last - lastx + x;
                routes[routes[0]++] = min(entry, p1);
                routes[routes[0]++] = max(entry, p1) - min(entry, p1);
                for (int i = min(entry, p1); i < max(entry, p1); i++) atomicAdd(wires + i, 1);
                routes[routes[0]++] = min(last, p2);
                routes[routes[0]++] = max(last, p2) - min(last, p2);
                for (int i = min(last, p2); i < max(last, p2); i++) atomicAdd(wires + i, 1);
                for (int i = min(lastlayer, entrylayer); i < max(lastlayer, entrylayer); i++) {
                    int id = i * N * N + (((i & 1) ^ DIRECTION) ? _x * N + _y : _y * N + _x);
                    routes[routes[0]++] = id;
                    routes[routes[0]++] = -1;
                    atomicAdd(vias + id, 1);
                }
            } else {
                routes[routes[0]++] = min(last, entry);
                routes[routes[0]++] = max(last, entry) - min(last, entry);
                for (int i = min(last, entry); i < max(last, entry); i++) atomicAdd(wires + i, 1);
            }
            // if(x == 46 && y == 47)
            //     printf("(%d %d %d) <- (%d %d %d)\n", entrylayer, entry / N % N, entry % N, last / N / N, last / N %
            //     N, last % N);
            map[4 * LAYER * N * N + child] = lastlayer;
        }
    }
    if (debug)
        for (int i = 1; i < routes[0]; i += 2) {
            int layer = routes[i] / N / N, x = routes[i] / N % N, y = routes[i] % N;
            if ((layer & 1) ^ DIRECTION) {
                if (routes[i + 1] == -1)
                    printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, x, y, layer + 1, x, y);
                else
                    printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, x, y, layer, x, y + routes[i + 1]);
            } else {
                if (routes[i + 1] == -1)
                    printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, y, x, layer + 1, y, x);
                else
                    printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, y, x, layer, y + routes[i + 1], x);
            }
            /*if((layer & 1) ^ DIRECTION) {
                if(routes[i + 1] == -1)
                    printf("[%d, %d, %d], [%d, %d, %d], ", layer, x, y, layer + 1, x, y);
                else
                    printf("[%d, %d, %d], [%d, %d, %d], ", layer, x, y, layer, x, y + routes[i + 1]);
            } else {
                if(routes[i + 1] == -1)
                    printf("[%d, %d, %d], [%d, %d, %d], ", layer, y, x, layer + 1, y, x);
                else
                    printf("[%d, %d, %d], [%d, %d, %d], ", layer, y, x, layer, y + routes[i + 1], x);
            }*/
        }
}
#define FLAG -1
__global__ void cudaPatternRoute(int *map,
                                 int *points,
                                 int64_t *wireCostSum,
                                 int *viaCost,
                                 int *prev,
                                 int *prev2,
                                 int *wires,
                                 int *vias,
                                 int *routes,
                                 int LAYER,
                                 int N,
                                 int DIRECTION) {
    // printf("%d offset %d\n", threadIdx.x, points[threadIdx.x]);
    points += points[blockIdx.x];
    routes += points[0];
    routes[0] = 1;
    int node_cnt = points[1];
    points += 2;
    for (int t = 6 * (node_cnt - 1); t >= 0; t -= 6) {
        int tox = points[t] / N, toy = points[t] % N, minlayer = points[t + 1], maxlayer = points[t + 2];
        // if(points[-2] == FLAG) printf("POINT (%d, %d)\n", tox, toy);
        /*if(points[t + 3] == -1)
            printf("no child\n");
        else
            printf("child!\n");*/
        for (int j = 1; j <= 3; j++)
            if (points[t + 2 + j] != -1) {
                int fromx = points[t + 2 + j] / N, fromy = points[t + 2 + j] % N;
                if (fromx != tox && fromy != toy) {
                    // if(max(fromx, tox) - min(fromx, tox) > 3 && max(fromy, toy) - min(fromy, toy) > 3) printf("%d %d
                    // largeto %d %d\n", fromx, fromy, tox, toy);
                    for (int a = min(fromx, tox); a <= max(fromx, tox); a++)
                        for (int i = 1; i < LAYER; i++) {
                            if ((i & 1) ^ DIRECTION) {
                                map[4 * LAYER * N * N + i * N * N + a * N + toy] = INF;
                                map[4 * LAYER * N * N + i * N * N + a * N + fromy] = INF;
                            } else {
                                map[4 * LAYER * N * N + i * N * N + toy * N + a] = INF;
                                map[4 * LAYER * N * N + i * N * N + fromy * N + a] = INF;
                            }
                        }
                    for (int b = min(fromy, toy); b <= max(fromy, toy); b++)
                        for (int i = 1; i < LAYER; i++) {
                            if ((i & 1) ^ DIRECTION) {
                                map[4 * LAYER * N * N + i * N * N + tox * N + b] = INF;
                                map[4 * LAYER * N * N + i * N * N + fromx * N + b] = INF;
                            } else {
                                map[4 * LAYER * N * N + i * N * N + b * N + tox] = INF;
                                map[4 * LAYER * N * N + i * N * N + b * N + fromx] = INF;
                            }
                        }
                    for (int i = 1; i < LAYER; i++)
                        if ((i & 1) ^ DIRECTION) {
                            int last = i * N * N + fromx * N + fromy;
                            for (int b = min(fromy, toy); b <= max(fromy, toy); b++)
                                if (b != fromy) {
                                    int cur = i * N * N + fromx * N + b;
                                    int cost = last < cur ? wireCostSum[last] - wireCostSum[cur]
                                                          : wireCostSum[cur] - wireCostSum[last];
                                    map[4 * LAYER * N * N + cur] = map[last] + cost;
                                    prev[4 * LAYER * N * N + cur] = last;
                                    // printf("(%d, %d, %d) %d\n", i, fromx, b, map[4 * LAYER * N * N + cur]);
                                }
                        } else {
                            int last = i * N * N + fromy * N + fromx;
                            for (int b = min(fromx, tox); b <= max(fromx, tox); b++)
                                if (b != fromx) {
                                    int cur = i * N * N + fromy * N + b;
                                    int cost = last < cur ? wireCostSum[last] - wireCostSum[cur]
                                                          : wireCostSum[cur] - wireCostSum[last];
                                    map[4 * LAYER * N * N + cur] = map[last] + cost;
                                    prev[4 * LAYER * N * N + cur] = last;
                                    // printf("(%d, %d, %d) %d\n", i, b, fromy, map[4 * LAYER * N * N + cur]);
                                }
                        }
                    for (int b = min(fromy, toy); b <= max(fromy, toy); b++)
                        if (b != fromy) viaSweep(map, prev, viaCost, fromx, b, LAYER, N, DIRECTION);
                    for (int b = min(fromx, tox); b <= max(fromx, tox); b++)
                        if (b != fromx) viaSweep(map, prev, viaCost, b, fromy, LAYER, N, DIRECTION);
                    for (int i = 1; i < LAYER; i++)
                        if ((i & 1) ^ DIRECTION) {
                            for (int b = min(fromx, tox); b <= max(fromx, tox); b++) {
                                int last = i * N * N + b * N + fromy, cur = i * N * N + b * N + toy;
                                int cost = last < cur ? wireCostSum[last] - wireCostSum[cur]
                                                      : wireCostSum[cur] - wireCostSum[last];
                                if (map[4 * LAYER * N * N + last] + cost < map[4 * LAYER * N * N + cur]) {
                                    map[4 * LAYER * N * N + cur] = map[4 * LAYER * N * N + last] + cost;
                                    prev[4 * LAYER * N * N + cur] = prev[4 * LAYER * N * N + last];
                                }
                            }
                        } else {
                            for (int b = min(fromy, toy); b <= max(fromy, toy); b++) {
                                int last = i * N * N + b * N + fromx, cur = i * N * N + b * N + tox;
                                int cost = last < cur ? wireCostSum[last] - wireCostSum[cur]
                                                      : wireCostSum[cur] - wireCostSum[last];
                                if (map[4 * LAYER * N * N + last] + cost < map[4 * LAYER * N * N + cur]) {
                                    map[4 * LAYER * N * N + cur] = map[4 * LAYER * N * N + last] + cost;
                                    prev[4 * LAYER * N * N + cur] = prev[4 * LAYER * N * N + last];
                                }
                            }
                        }
                    for (int b = min(fromy, toy); b <= max(fromy, toy); b++)
                        if (b != toy) viaSweep(map, prev, viaCost, tox, b, LAYER, N, DIRECTION);
                    for (int b = min(fromx, tox); b <= max(fromx, tox); b++)
                        if (b != tox) viaSweep(map, prev, viaCost, b, toy, LAYER, N, DIRECTION);
                    for (int i = 1; i < LAYER; i++)
                        if ((i & 1) ^ DIRECTION) {
                            int cur = i * N * N + tox * N + toy;
                            for (int b = min(fromy, toy); b <= max(fromy, toy); b++)
                                if (b != toy) {
                                    int last = i * N * N + tox * N + b;
                                    int cost = last < cur ? wireCostSum[last] - wireCostSum[cur]
                                                          : wireCostSum[cur] - wireCostSum[last];
                                    if (map[4 * LAYER * N * N + last] + cost < map[cur + j * LAYER * N * N]) {
                                        map[cur + j * LAYER * N * N] = map[4 * LAYER * N * N + last] + cost;
                                        prev[cur + j * LAYER * N * N] = prev[4 * LAYER * N * N + last];
                                        prev2[cur + j * LAYER * N * N] = prev[5 * LAYER * N * N + last];
                                    }
                                }
                        } else {
                            int cur = i * N * N + toy * N + tox;
                            for (int b = min(fromx, tox); b <= max(fromx, tox); b++)
                                if (b != tox) {
                                    int last = i * N * N + toy * N + b;
                                    int cost = last < cur ? wireCostSum[last] - wireCostSum[cur]
                                                          : wireCostSum[cur] - wireCostSum[last];
                                    if (map[4 * LAYER * N * N + last] + cost < map[cur + j * LAYER * N * N]) {
                                        map[cur + j * LAYER * N * N] = map[4 * LAYER * N * N + last] + cost;
                                        prev[cur + j * LAYER * N * N] = prev[4 * LAYER * N * N + last];
                                        prev2[cur + j * LAYER * N * N] = prev[5 * LAYER * N * N + last];
                                    }
                                }
                        }
                } else if (fromx == tox) {
                    for (int i = 1; i < LAYER; i++)
                        if ((i & 1) ^ DIRECTION) {
                            int last = i * N * N + fromx * N + fromy, cur = i * N * N + tox * N + toy;
                            int cost = last < cur ? wireCostSum[last] - wireCostSum[cur]
                                                  : wireCostSum[cur] - wireCostSum[last];
                            if (map[cur + j * LAYER * N * N] > map[last] + cost)
                                map[cur + j * LAYER * N * N] = map[last] + cost, prev[cur + j * LAYER * N * N] = last;
                        } else
                            map[i * N * N + tox * N + toy + j * LAYER * N * N] = INF;
                } else if (fromy == toy) {
                    for (int i = 1; i < LAYER; i++)
                        if (!(i & 1) ^ DIRECTION) {
                            int last = i * N * N + fromy * N + fromx, cur = i * N * N + toy * N + tox;
                            int cost = last < cur ? wireCostSum[last] - wireCostSum[cur]
                                                  : wireCostSum[cur] - wireCostSum[last];
                            if (map[cur + j * LAYER * N * N] > map[last] + cost) {
                                map[cur + j * LAYER * N * N] = map[last] + cost, prev[cur + j * LAYER * N * N] = last;
                            }
                        } else {
                            map[i * N * N + toy * N + tox + j * LAYER * N * N] = INF;
                        }
                } else
                    printf("ERROR: points on the same location!\n");
            }
        if (debug) printf("\n");
        for (int lower = 0; lower < LAYER; lower++) {
            int viaSum = 0, minCost = INF, bestpos[3] = {0, 0, 0},
                costs[3] = {
                    points[t + 3] == -1 ? 0 : INF, points[t + 4] == -1 ? 0 : INF, points[t + 5] == -1 ? 0 : INF};
            for (int upper = lower; upper < LAYER; upper++) {
                int idx = upper * N * N + (((upper & 1) ^ DIRECTION) ? tox * N + toy : toy * N + tox);
                for (int j = 0; j < 3; j++)
                    if (points[t + 3 + j] != -1) {
                        if (map[idx + (j + 1) * LAYER * N * N] < costs[j])
                            costs[j] = map[idx + (j + 1) * LAYER * N * N], bestpos[j] = upper;
                    }
                if (1LL * costs[0] + costs[1] + costs[2] < minCost) minCost = costs[0] + costs[1] + costs[2];
                if (minlayer == -1 || (lower <= minlayer && maxlayer <= upper)) {
                    for (int i = lower; i <= upper; i++) {
                        int index = i * N * N + (((i & 1) ^ DIRECTION) ? tox * N + toy : toy * N + tox);
                        if (map[index] > minCost + viaSum)
                            map[index] = minCost + viaSum,
                            prev[index] =
                                (((lower * LAYER + upper) * LAYER + bestpos[2]) * LAYER + bestpos[1]) * LAYER +
                                bestpos[0];
                    }
                }
                viaSum += viaCost[idx];
            }
        }
        /*if(0) for(int i = 0; i < LAYER; i++) {
            int index = i * N * N + (((i & 1) ^ DIRECTION) ? tox * N + toy : toy * N + tox);
            int lower = prev[index] / LAYER / LAYER / LAYER / LAYER, upper = prev[index] / LAYER / LAYER / LAYER %
        LAYER; printf("layer %d, cost %d, lower %d, upper %d, bestpos %d\n", i, map[index], lower, upper, prev[index] %
        LAYER);
        }*/
    }
    int minTotalDist = INF;
    for (int i = 0; i < LAYER; i++) {
        int x = points[0] / N, y = points[0] % N;
        int idx = i * N * N + (((i & 1) ^ DIRECTION) ? x * N + y : y * N + x);

        if (map[idx] < minTotalDist) minTotalDist = map[idx], map[4 * LAYER * N * N + points[0]] = i;
    }
    // if(points[-2] == FLAG) printf("min total dist %d\n", minTotalDist);
    if (minTotalDist == INF || minTotalDist < 0) {
        printf("INF is too small!\n");
        printf("routes[0] = %d\n", points[-2]);
        return;
    }
    for (int t = 0; t < 6 * node_cnt; t += 6) {
        int x = points[t] / N, y = points[t] % N, layer = map[4 * LAYER * N * N + points[t]];
        // printf("point (%d, %d)\n", x, y);
        int idx = layer * N * N + (((layer & 1) ^ DIRECTION) ? x * N + y : y * N + x);
        int minlayer = prev[idx] / LAYER / LAYER / LAYER / LAYER, maxlayer = prev[idx] / LAYER / LAYER / LAYER % LAYER;
        // printf("layer range [%d, %d]\n", minlayer, maxlayer);
        for (int i = minlayer; i < maxlayer; i++) {
            int id = i * N * N + (((i & 1) ^ DIRECTION) ? x * N + y : y * N + x);
            routes[routes[0]++] = id;
            routes[routes[0]++] = -1;
            atomicAdd(vias + id, 1);
        }
        int bestpos = prev[idx];
        for (int j = 0; j < 3; j++) {
            int child = points[t + 3 + j];
            if (points[t + 3 + j] == -1) break;
            int entrylayer = bestpos % LAYER,
                entry = entrylayer * N * N + (((entrylayer & 1) ^ DIRECTION) ? x * N + y : y * N + x);
            // printf("(%d, %d, %d) ", entrylayer, x, y);
            bestpos /= LAYER;
            int last = prev[(j + 1) * LAYER * N * N + entry];
            int lastlayer = last / N / N, lastx = last / N % N, lasty = last % N;
            if (!(lastlayer & 1) ^ DIRECTION) cudaSwap(lastx, lasty);
            if (lastx != x && lasty != y) {
                int mid = prev2[(j + 1) * LAYER * N * N + entry];
                int midlayer = mid / N / N, midx = mid / N % N, midy = mid % N;
                if (!(midlayer & 1) ^ DIRECTION) cudaSwap(midx, midy);
                // printf("%d (%d, %d, %d) %d\n", entrylayer, midlayer, midx, midy, lastlayer);
                // entry-(wire)->p1-(via)->mid-(wire)->p3-(via)->p4-(wire)->last

                int p1 = ((entrylayer & 1) ^ DIRECTION) ? entry - y + midy : entry - x + midx;
                int p3 = ((midlayer & 1) ^ DIRECTION) ? mid - midy + lasty : mid - midx + lastx;
                int p4 = ((lastlayer & 1) ^ DIRECTION) ? last - lasty + midy : last - lastx + midx;
                /*printf("child %d of %d,   ", j, t);
                {
                    int temp = entry;
                    int templayer = temp / N / N, tempx = temp / N % N, tempy = temp % N;
                    if(!(templayer & 1) ^ DIRECTION) cudaSwap(tempx, tempy);
                    printf("(%d, %d, %d) -> ", templayer, tempx, tempy);
                }
                {
                    int temp = p1;
                    int templayer = temp / N / N, tempx = temp / N % N, tempy = temp % N;
                    if(!(templayer & 1) ^ DIRECTION) cudaSwap(tempx, tempy);
                    printf("(%d, %d, %d) -> ", templayer, tempx, tempy);
                }
                {
                    int temp = mid;
                    int templayer = temp / N / N, tempx = temp / N % N, tempy = temp % N;
                    if(!(templayer & 1) ^ DIRECTION) cudaSwap(tempx, tempy);
                    printf("(%d, %d, %d) -> ", templayer, tempx, tempy);
                }
                {
                    int temp = p3;
                    int templayer = temp / N / N, tempx = temp / N % N, tempy = temp % N;
                    if(!(templayer & 1) ^ DIRECTION) cudaSwap(tempx, tempy);
                    printf("(%d, %d, %d) -> ", templayer, tempx, tempy);
                }
                {
                    int temp = p4;
                    int templayer = temp / N / N, tempx = temp / N % N, tempy = temp % N;
                    if(!(templayer & 1) ^ DIRECTION) cudaSwap(tempx, tempy);
                    printf("(%d, %d, %d) -> ", templayer, tempx, tempy);
                }
                {
                    int temp = last;
                    int templayer = temp / N / N, tempx = temp / N % N, tempy = temp % N;
                    if(!(templayer & 1) ^ DIRECTION) cudaSwap(tempx, tempy);
                    printf("(%d, %d, %d)\n", templayer, tempx, tempy);
                }
                int cur = routes[0];*/
                if (entry != p1) {
                    routes[routes[0]++] = min(entry, p1);
                    routes[routes[0]++] = max(entry, p1) - min(entry, p1);
                    for (int i = min(entry, p1); i < max(entry, p1); i++) atomicAdd(wires + i, 1);
                }
                if (p1 != mid) {
                    for (int i = min(entrylayer, midlayer); i < max(entrylayer, midlayer); i++) {
                        int id = i * N * N + (((i & 1) ^ DIRECTION) ? midx * N + midy : midy * N + midx);
                        routes[routes[0]++] = id;
                        routes[routes[0]++] = -1;
                        atomicAdd(vias + id, 1);
                    }
                }
                if (mid != p3) {
                    routes[routes[0]++] = min(mid, p3);
                    routes[routes[0]++] = max(mid, p3) - min(mid, p3);
                    for (int i = min(mid, p3); i < max(mid, p3); i++) atomicAdd(wires + i, 1);
                }
                if (p3 != p4) {
                    int tempx = p3 / N % N, tempy = p3 % N;
                    if (!(p3 / N / N & 1) ^ DIRECTION) cudaSwap(tempx, tempy);
                    for (int i = min(midlayer, lastlayer); i < max(midlayer, lastlayer); i++) {
                        int id = i * N * N + (((i & 1) ^ DIRECTION) ? tempx * N + tempy : tempy * N + tempx);
                        routes[routes[0]++] = id;
                        routes[routes[0]++] = -1;
                        atomicAdd(vias + id, 1);
                    }
                }
                if (p4 != last) {
                    routes[routes[0]++] = min(p4, last);
                    routes[routes[0]++] = max(p4, last) - min(p4, last);
                    for (int i = min(p4, last); i < max(p4, last); i++) atomicAdd(wires + i, 1);
                }
                /*for(int i = cur; i < routes[0]; i += 2) {
                    int layer = routes[i] / N / N, x = routes[i] / N % N, y = routes[i] % N;
                    if((layer & 1) ^ DIRECTION) {
                        if(routes[i + 1] == -1)
                            printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, x, y, layer + 1, x, y);
                        else
                            printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, x, y, layer, x, y + routes[i + 1]);
                    } else {
                        if(routes[i + 1] == -1)
                            printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, y, x, layer + 1, y, x);
                        else
                            printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, y, x, layer, y + routes[i + 1], x);
                    }
                }
                printf("\n");*/

            } else {
                routes[routes[0]++] = min(last, entry);
                routes[routes[0]++] = max(last, entry) - min(last, entry);
                for (int i = min(last, entry); i < max(last, entry); i++) atomicAdd(wires + i, 1);
            }
            map[4 * LAYER * N * N + child] = lastlayer;
        }
    }
    if (debug)
        for (int i = 1; i < routes[0]; i += 2) {
            int layer = routes[i] / N / N, x = routes[i] / N % N, y = routes[i] % N;
            if ((layer & 1) ^ DIRECTION) {
                if (routes[i + 1] == -1)
                    printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, x, y, layer + 1, x, y);
                else
                    printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, x, y, layer, x, y + routes[i + 1]);
            } else {
                if (routes[i + 1] == -1)
                    printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, y, x, layer + 1, y, x);
                else
                    printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, y, x, layer, y + routes[i + 1], x);
            }
        }
    // printf("\n");
}
__global__ void cudaPatternRouteParallel(int *map,
                                         int *points,
                                         int64_t *wireCostSum,
                                         int *viaCost,
                                         int *prev,
                                         int *prev2,
                                         int *wires,
                                         int *vias,
                                         int *routes,
                                         int *gbpoints,
                                         int *gbpinRoutes,
                                         int LAYER,
                                         int N,
                                         int DIRECTION) {
    // printf("%d offset %d\n", threadIdx.x, points[threadIdx.x]);
    gbpoints += gbpoints[blockIdx.x];
    points += points[blockIdx.x];
    routes += points[0];
    if (threadIdx.x == 0) routes[0] = 1;
    __syncthreads();
    int node_cnt = points[1];
    points += 2;
    for (int t = 6 * (node_cnt - 1); t >= 0; t -= 6) {
        int tox = points[t] / N, toy = points[t] % N, minlayer = points[t + 1], maxlayer = points[t + 2];
        for (int j = 1; j <= 3; j++) {
            if (points[t + 2 + j] != -1) {
                int childPos = points[points[t + 2 + j]];
                int fromx = childPos / N, fromy = childPos % N;
                if (fromx != tox && fromy != toy) {
                    for (int a = min(fromx, tox) + threadIdx.x; a <= max(fromx, tox); a += blockDim.x) {
                        for (int i = 1; i < LAYER; i++) {
                            if ((i & 1) ^ DIRECTION) {
                                map[4 * LAYER * N * N + i * N * N + a * N + toy] = INF;
                                map[4 * LAYER * N * N + i * N * N + a * N + fromy] = INF;
                            } else {
                                map[4 * LAYER * N * N + i * N * N + toy * N + a] = INF;
                                map[4 * LAYER * N * N + i * N * N + fromy * N + a] = INF;
                            }
                        }
                    }
                    __syncthreads();
                    for (int b = min(fromy, toy) + threadIdx.x; b <= max(fromy, toy); b += blockDim.x) {
                        for (int i = 1; i < LAYER; i++) {
                            if ((i & 1) ^ DIRECTION) {
                                map[4 * LAYER * N * N + i * N * N + tox * N + b] = INF;
                                map[4 * LAYER * N * N + i * N * N + fromx * N + b] = INF;
                            } else {
                                map[4 * LAYER * N * N + i * N * N + b * N + tox] = INF;
                                map[4 * LAYER * N * N + i * N * N + b * N + fromx] = INF;
                            }
                        }
                    }
                    __syncthreads();
                    for (int i = 1; i < LAYER; i++) {
                        if ((i & 1) ^ DIRECTION) {
                            int last = i * N * N + fromx * N + fromy;
                            for (int b = min(fromy, toy) + threadIdx.x; b <= max(fromy, toy); b += blockDim.x) {
                                if (b != fromy) {
                                    int cur = i * N * N + fromx * N + b;
                                    int cost = last < cur ? wireCostSum[last] - wireCostSum[cur]
                                                          : wireCostSum[cur] - wireCostSum[last];
                                    map[4 * LAYER * N * N + cur] = map[last] + cost;
                                    prev[4 * LAYER * N * N + cur] = last;
                                }
                            }
                        } else {
                            int last = i * N * N + fromy * N + fromx;
                            for (int b = min(fromx, tox) + threadIdx.x; b <= max(fromx, tox); b += blockDim.x) {
                                if (b != fromx) {
                                    int cur = i * N * N + fromy * N + b;
                                    int cost = last < cur ? wireCostSum[last] - wireCostSum[cur]
                                                          : wireCostSum[cur] - wireCostSum[last];
                                    map[4 * LAYER * N * N + cur] = map[last] + cost;
                                    prev[4 * LAYER * N * N + cur] = last;
                                }
                            }
                        }
                        __syncthreads();
                    }
                    __syncthreads();
                    for (int b = min(fromy, toy) + threadIdx.x; b <= max(fromy, toy); b += blockDim.x)
                        if (b != fromy) viaSweep(map, prev, viaCost, fromx, b, LAYER, N, DIRECTION);
                    __syncthreads();
                    for (int b = min(fromx, tox) + threadIdx.x; b <= max(fromx, tox); b += blockDim.x)
                        if (b != fromx) viaSweep(map, prev, viaCost, b, fromy, LAYER, N, DIRECTION);
                    __syncthreads();
                    for (int i = 1; i < LAYER; i++) {
                        if ((i & 1) ^ DIRECTION) {
                            for (int b = min(fromx, tox) + threadIdx.x; b <= max(fromx, tox); b += blockDim.x) {
                                int last = i * N * N + b * N + fromy, cur = i * N * N + b * N + toy;
                                int cost = last < cur ? wireCostSum[last] - wireCostSum[cur]
                                                      : wireCostSum[cur] - wireCostSum[last];
                                if (map[4 * LAYER * N * N + last] + cost < map[4 * LAYER * N * N + cur]) {
                                    map[4 * LAYER * N * N + cur] = map[4 * LAYER * N * N + last] + cost;
                                    prev[4 * LAYER * N * N + cur] = prev[4 * LAYER * N * N + last];
                                }
                            }
                        } else {
                            for (int b = min(fromy, toy) + threadIdx.x; b <= max(fromy, toy); b += blockDim.x) {
                                int last = i * N * N + b * N + fromx, cur = i * N * N + b * N + tox;
                                int cost = last < cur ? wireCostSum[last] - wireCostSum[cur]
                                                      : wireCostSum[cur] - wireCostSum[last];
                                if (map[4 * LAYER * N * N + last] + cost < map[4 * LAYER * N * N + cur]) {
                                    map[4 * LAYER * N * N + cur] = map[4 * LAYER * N * N + last] + cost;
                                    prev[4 * LAYER * N * N + cur] = prev[4 * LAYER * N * N + last];
                                }
                            }
                        }
                        __syncthreads();
                    }
                    __syncthreads();
                    for (int b = min(fromy, toy) + threadIdx.x; b <= max(fromy, toy); b += blockDim.x)
                        if (b != toy) viaSweep(map, prev, viaCost, tox, b, LAYER, N, DIRECTION);
                    __syncthreads();
                    for (int b = min(fromx, tox) + threadIdx.x; b <= max(fromx, tox); b += blockDim.x)
                        if (b != tox) viaSweep(map, prev, viaCost, b, toy, LAYER, N, DIRECTION);
                    __syncthreads();
                    for (int i = 1 + threadIdx.x; i < LAYER; i += blockDim.x) {
                        if ((i & 1) ^ DIRECTION) {
                            int cur = i * N * N + tox * N + toy;
                            for (int b = min(fromy, toy); b <= max(fromy, toy); b++) {
                                if (b != toy) {
                                    int last = i * N * N + tox * N + b;
                                    int cost = last < cur ? wireCostSum[last] - wireCostSum[cur]
                                                          : wireCostSum[cur] - wireCostSum[last];
                                    if (map[4 * LAYER * N * N + last] + cost < map[cur + j * LAYER * N * N]) {
                                        map[cur + j * LAYER * N * N] = map[4 * LAYER * N * N + last] + cost;
                                        prev[cur + j * LAYER * N * N] = prev[4 * LAYER * N * N + last];
                                        prev2[cur + j * LAYER * N * N] = prev[5 * LAYER * N * N + last];
                                    }
                                }
                            }
                        } else {
                            int cur = i * N * N + toy * N + tox;
                            for (int b = min(fromx, tox); b <= max(fromx, tox); b++) {
                                if (b != tox) {
                                    int last = i * N * N + toy * N + b;
                                    int cost = last < cur ? wireCostSum[last] - wireCostSum[cur]
                                                          : wireCostSum[cur] - wireCostSum[last];
                                    if (map[4 * LAYER * N * N + last] + cost < map[cur + j * LAYER * N * N]) {
                                        map[cur + j * LAYER * N * N] = map[4 * LAYER * N * N + last] + cost;
                                        prev[cur + j * LAYER * N * N] = prev[4 * LAYER * N * N + last];
                                        prev2[cur + j * LAYER * N * N] = prev[5 * LAYER * N * N + last];
                                    }
                                }
                            }
                        }
                    }
                    __syncthreads();
                } else if (fromx == tox) {
                    for (int i = 1 + threadIdx.x; i < LAYER; i += blockDim.x) {
                        if ((i & 1) ^ DIRECTION) {
                            int last = i * N * N + fromx * N + fromy, cur = i * N * N + tox * N + toy;
                            int cost = last < cur ? wireCostSum[last] - wireCostSum[cur]
                                                  : wireCostSum[cur] - wireCostSum[last];
                            if (map[cur + j * LAYER * N * N] > map[last] + cost)
                                map[cur + j * LAYER * N * N] = map[last] + cost, prev[cur + j * LAYER * N * N] = last;
                        } else {
                            map[i * N * N + toy * N + tox + j * LAYER * N * N] = INF;
                        }
                    }
                    __syncthreads();
                } else if (fromy == toy) {
                    for (int i = 1 + threadIdx.x; i < LAYER; i += blockDim.x) {
                        if (!(i & 1) ^ DIRECTION) {
                            int last = i * N * N + fromy * N + fromx, cur = i * N * N + toy * N + tox;
                            int cost = last < cur ? wireCostSum[last] - wireCostSum[cur]
                                                  : wireCostSum[cur] - wireCostSum[last];
                            if (map[cur + j * LAYER * N * N] > map[last] + cost) {
                                map[cur + j * LAYER * N * N] = map[last] + cost, prev[cur + j * LAYER * N * N] = last;
                            }
                        } else {
                            map[i * N * N + tox * N + toy + j * LAYER * N * N] = INF;
                        }
                    }
                    __syncthreads();
                } else {
                    printf("ERROR: points on the same location!\n");
                }
                __syncthreads();
            }
            __syncthreads();
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int lower = 0; lower < LAYER; lower++) {
                int viaSum = 0, minCost = INF, bestpos[3] = {0, 0, 0},
                    costs[3] = {
                        points[t + 3] == -1 ? 0 : INF, points[t + 4] == -1 ? 0 : INF, points[t + 5] == -1 ? 0 : INF};
                for (int upper = lower; upper < LAYER; upper++) {
                    int idx = upper * N * N + (((upper & 1) ^ DIRECTION) ? tox * N + toy : toy * N + tox);
                    for (int j = 0; j < 3; j++)
                        if (points[t + 3 + j] != -1) {
                            if (map[idx + (j + 1) * LAYER * N * N] < costs[j])
                                costs[j] = map[idx + (j + 1) * LAYER * N * N], bestpos[j] = upper;
                        }
                    if (1LL * costs[0] + costs[1] + costs[2] < minCost) minCost = costs[0] + costs[1] + costs[2];
                    if (minlayer == -1 || (lower <= minlayer && maxlayer <= upper)) {
                        for (int i = lower; i <= upper; i++) {
                            int index = i * N * N + (((i & 1) ^ DIRECTION) ? tox * N + toy : toy * N + tox);
                            if (map[index] > minCost + viaSum)
                                map[index] = minCost + viaSum,
                                prev[index] =
                                    (((lower * LAYER + upper) * LAYER + bestpos[2]) * LAYER + bestpos[1]) * LAYER +
                                    bestpos[0];
                        }
                    }
                    // if(t == 0)
                    //     printf("[%d, %d] best = %d, cost = %d, via=%d\n", lower, upper, prev[25025] % LAYER,
                    //     costs[0], viaSum);
                    viaSum += viaCost[idx];
                }
                // if(t == 0)
                //     printf("mincost = %d, [%d, %d] bestpos = %d\n", minCost, minlayer, maxlayer, bestpos[0]);
            }
        }
        __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int minTotalDist = INF;
        for (int i = 0; i < LAYER; i++) {
            int x = points[0] / N, y = points[0] % N;
            int idx = i * N * N + (((i & 1) ^ DIRECTION) ? x * N + y : y * N + x);

            if (map[idx] < minTotalDist) {
                minTotalDist = map[idx];
                // TODO: use a N x N matrix to achieve pos2layer map so that the map size can be optimized
                map[5 * LAYER * N * N + points[0]] = i;
                // printf("layer = %d, idx=%d, bestpos = %d\n", i, idx, prev[idx] % LAYER);
            }
        }
        if (minTotalDist == INF || minTotalDist < 0) {
            printf("failed\n");
            routes[0] = -1;
            // printf("routes[0] = %d\n", points[-2]);
            // return;
        } else {
            // printf("total cost %d, layer %d\n", minTotalDist, map[5 * LAYER * N * N + points[0]]);
            for (int t = 0; t < 6 * node_cnt; t += 6) {
                int t_gbPinId = gbpoints[t / 6];
                int x = points[t] / N, y = points[t] % N, layer = map[5 * LAYER * N * N + points[t]];
                int idx = layer * N * N + (((layer & 1) ^ DIRECTION) ? x * N + y : y * N + x);
                int minlayer = prev[idx] / LAYER / LAYER / LAYER / LAYER,
                    maxlayer = prev[idx] / LAYER / LAYER / LAYER % LAYER;
                for (int i = minlayer; i < maxlayer; i++) {
                    int id = i * N * N + (((i & 1) ^ DIRECTION) ? x * N + y : y * N + x);
                    routes[routes[0]++] = id;
                    routes[routes[0]++] = -1;
                    vias[id]++;
                }
                // Assign #vias to gbpinRoutes
                if (t_gbPinId != -1) {
                    gbpinRoutes[t_gbPinId * 6 + 5] = maxlayer - minlayer;
                }
                // printf("best range [%d, %d]\n", minlayer, maxlayer);
                int bestpos = prev[idx];
                for (int j = 0; j < 3; j++) {
                    if (points[t + 3 + j] == -1) break;
                    int entry_routeId = -1, last_routeId = -1;
                    int child = points[points[t + 3 + j]];
                    int entrylayer = bestpos % LAYER,
                        entry = entrylayer * N * N + (((entrylayer & 1) ^ DIRECTION) ? x * N + y : y * N + x);
                    // if(t == 0)
                    //     printf("layer %d\n", entrylayer);
                    // printf("(%d, %d, %d) ", entrylayer, x, y);
                    bestpos /= LAYER;
                    int last = prev[(j + 1) * LAYER * N * N + entry];
                    int lastlayer = last / N / N, lastx = last / N % N, lasty = last % N;
                    if (!(lastlayer & 1) ^ DIRECTION) cudaSwap(lastx, lasty);
                    if (lastx != x && lasty != y) {
                        int mid = prev2[(j + 1) * LAYER * N * N + entry];
                        int midlayer = mid / N / N, midx = mid / N % N, midy = mid % N;
                        if (!(midlayer & 1) ^ DIRECTION) cudaSwap(midx, midy);
                        // printf("%d (%d, %d, %d) %d\n", entrylayer, midlayer, midx, midy, lastlayer);
                        // entry-(wire)->p1-(via)->mid-(wire)->p3-(via)->p4-(wire)->last

                        int p1 = ((entrylayer & 1) ^ DIRECTION) ? entry - y + midy : entry - x + midx;
                        int p3 = ((midlayer & 1) ^ DIRECTION) ? mid - midy + lasty : mid - midx + lastx;
                        int p4 = ((lastlayer & 1) ^ DIRECTION) ? last - lasty + midy : last - lastx + midx;
                        if (entry != p1) {
                            int encodeId = entry <= p1 ? routes[0] : -routes[0];
                            if (entry_routeId == -1) entry_routeId = encodeId;
                            last_routeId = encodeId;

                            routes[routes[0]++] = min(entry, p1);
                            routes[routes[0]++] = max(entry, p1) - min(entry, p1);
                            for (int i = min(entry, p1); i < max(entry, p1); i++) wires[i]++;
                            // atomicAdd(wires + i, 1);
                        }
                        if (p1 != mid) {
                            for (int i = min(entrylayer, midlayer); i < max(entrylayer, midlayer); i++) {
                                int id = i * N * N + (((i & 1) ^ DIRECTION) ? midx * N + midy : midy * N + midx);
                                routes[routes[0]++] = id;
                                routes[routes[0]++] = -1;
                                vias[id]++;
                                // atomicAdd(vias + id, 1);
                            }
                        }
                        if (mid != p3) {
                            int encodeId = mid <= p3 ? routes[0] : -routes[0];
                            if (entry_routeId == -1) entry_routeId = encodeId;
                            last_routeId = encodeId;

                            routes[routes[0]++] = min(mid, p3);
                            routes[routes[0]++] = max(mid, p3) - min(mid, p3);

                            for (int i = min(mid, p3); i < max(mid, p3); i++) wires[i]++;
                            // atomicAdd(wires + i, 1);
                        }
                        if (p3 != p4) {
                            int tempx = p3 / N % N, tempy = p3 % N;
                            if (!(p3 / N / N & 1) ^ DIRECTION) cudaSwap(tempx, tempy);
                            for (int i = min(midlayer, lastlayer); i < max(midlayer, lastlayer); i++) {
                                int id = i * N * N + (((i & 1) ^ DIRECTION) ? tempx * N + tempy : tempy * N + tempx);
                                routes[routes[0]++] = id;
                                routes[routes[0]++] = -1;
                                vias[id]++;
                                // atomicAdd(vias + id, 1);
                            }
                        }
                        if (p4 != last) {
                            int encodeId = p4 <= last ? routes[0] : -routes[0];
                            if (entry_routeId == -1) entry_routeId = encodeId;
                            last_routeId = encodeId;

                            routes[routes[0]++] = min(p4, last);
                            routes[routes[0]++] = max(p4, last) - min(p4, last);
                            for (int i = min(p4, last); i < max(p4, last); i++) wires[i]++;
                            // atomicAdd(wires + i, 1);
                        }

                    } else {
                        int encodeId = entry <= last ? routes[0] : -routes[0];
                        if (entry_routeId == -1) entry_routeId = encodeId;
                        last_routeId = encodeId;

                        routes[routes[0]++] = min(last, entry);
                        routes[routes[0]++] = max(last, entry) - min(last, entry);
                        for (int i = min(last, entry); i < max(last, entry); i++) wires[i]++;
                        // atomicAdd(wires + i, 1);
                    }
                    map[5 * LAYER * N * N + child] = lastlayer;

                    // Assign entry/last route to gbpinRoutes
                    if (t_gbPinId != -1) {
                        gbpinRoutes[t_gbPinId * 6 + 1 + gbpinRoutes[t_gbPinId * 6]] = entry_routeId;
                        gbpinRoutes[t_gbPinId * 6]++;
                        if (gbpinRoutes[t_gbPinId * 6] > 4) {
                            printf("Error: numRoutes for t_gbpin d%d\n", gbpinRoutes[t_gbPinId * 6]);
                        }
                    }
                    int c_gbPinId = gbpoints[points[t + 3 + j] / 6];
                    if (c_gbPinId != -1) {
                        // negative routeId to refer the last route
                        gbpinRoutes[c_gbPinId * 6 + 1 + gbpinRoutes[c_gbPinId * 6]] = -1 * last_routeId;
                        gbpinRoutes[c_gbPinId * 6]++;
                        if (gbpinRoutes[c_gbPinId * 6] > 4) {
                            printf("Error: numRoutes for c_gbpin %d\n", gbpinRoutes[c_gbPinId * 6]);
                        }
                    }
                }
            }
        }
        /*for(int i = 1; i < routes[0]; i += 2) {
            int layer = routes[i] / N / N, x = routes[i] / N % N, y = routes[i] % N;
            if((layer & 1) ^ DIRECTION) {
                if(routes[i + 1] == -1)
                    printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, x, y, layer + 1, x, y);
                else
                    printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, x, y, layer, x, y + routes[i + 1]);
            } else {
                if(routes[i + 1] == -1)
                    printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, y, x, layer + 1, y, x);
                else
                    printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, y, x, layer, y + routes[i + 1], x);
            }
        }
        printf("\n");*/
    }
    __syncthreads();
    /*__syncthreads();
    for(int i = 1; i < routes[0]; i += 2) if(routes[i + 1] != -1) {
        for(int j = routes[i] + threadIdx.x; j < routes[i] + routes[i + 1]; j += blockDim.x)
            wires[j]++;
    }*/
}
/*
__global__ void cudaPatternRoute(int *map, int *points, int64_t *wireCostSum, int *viaCost, int *prev, int *wires, int
*vias, int *routes, int LAYER, int N, int DIRECTION) {
    //printf("%d offset %d\n", threadIdx.x, points[threadIdx.x]);
    points += points[blockIdx.x];
    if(threadIdx.x == 0)
        routes += points[0], routes[0] = 1;
    int node_cnt = points[1];
    points += 2;
    for(int t = 6 * (node_cnt - 1); t >= 0; t -= 6) {
        int tox = points[t] / N, toy = points[t] % N, minlayer = points[t + 1], maxlayer = points[t + 2];
        for(int j = 1; j <= 3; j++) if(points[t + 2 + j] != -1) {
            int fromx = points[t + 2 + j] / N, fromy = points[t + 2 + j] % N;
            //printf("%d %d to %d %d\n", fromx, fromy, tox, toy);
            if(fromx != tox && fromy != toy) {
                {
                    int i = threadIdx.x;
                    if((i & 1) ^ DIRECTION)
                        map[4 * LAYER * N * N + i * N * N + fromx * N + toy] = map[4 * LAYER * N * N + i * N * N + tox *
N + fromy] = INF; else map[4 * LAYER * N * N + i * N * N + toy * N + fromx] = map[4 * LAYER * N * N + i * N * N + fromy
* N + tox] = INF;
                }
                __syncthreads();
                if(threadIdx.x) {
                    int i = threadIdx.x;
                    int last = i * N * N + (((i & 1) ^ DIRECTION) ? fromx * N + fromy : fromy * N + fromx);
                    int cur = i * N * N + (((i & 1) ^ DIRECTION) ? fromx * N + toy : fromy * N + tox);
                    int cost = last < cur ? wireCostSum[last] - wireCostSum[cur] : wireCostSum[cur] - wireCostSum[last];
                    if(map[4 * LAYER * N * N + cur] > map[last] + cost)
                        map[4 * LAYER * N * N + cur] = map[last] + cost, prev[4 * LAYER * N * N + cur] = last;
                }
                __syncthreads();
                if(threadIdx.x == 0) {
                    viaSweep(map, prev, viaCost, fromx, toy, LAYER, N, DIRECTION);
                    viaSweep(map, prev, viaCost, tox, fromy, LAYER, N, DIRECTION);
                }
                __syncthreads();
                if(threadIdx.x) {
                    int i = threadIdx.x;
                    int last = i * N * N + (((i & 1) ^ DIRECTION) ? tox * N + fromy : toy * N + fromx);
                    int cur = i * N * N + (((i & 1) ^ DIRECTION) ? tox * N + toy : toy * N + tox);
                    int cost = last < cur ? wireCostSum[last] - wireCostSum[cur] : wireCostSum[cur] - wireCostSum[last];
                    if(map[cur + j * LAYER * N * N] > map[4 * LAYER * N * N + last] + cost)
                        map[cur + j * LAYER * N * N] = map[4 * LAYER * N * N + last] + cost, prev[cur + j * LAYER * N *
N] = prev[4 * LAYER * N * N + last];
                }
            } else if(fromx == tox) {
                if(threadIdx.x) {
                    int i = threadIdx.x;
                    if((i & 1) ^ DIRECTION) {
                        int last = i * N * N + fromx * N + fromy, cur = i * N * N + tox * N + toy;
                        int cost = last < cur ? wireCostSum[last] - wireCostSum[cur] : wireCostSum[cur] -
wireCostSum[last]; if(map[cur + j * LAYER * N * N] > map[last] + cost) map[cur + j * LAYER * N * N] = map[last] + cost,
prev[cur + j * LAYER * N * N] = last;
                    }
                }
            } else if(fromy == toy) {
                if(threadIdx.x) {
                    int i = threadIdx.x;
                    if(!(i & 1) ^ DIRECTION) {
                        int last = i * N * N + fromy * N + fromx, cur = i * N * N + toy * N + tox;
                        int cost = last < cur ? wireCostSum[last] - wireCostSum[cur] : wireCostSum[cur] -
wireCostSum[last]; if(map[cur + j * LAYER * N * N] > map[last] + cost) map[cur + j * LAYER * N * N] = map[last] + cost,
prev[cur + j * LAYER * N * N] = last;
                    }
                }
            } else
                printf("ERROR: points on the same location!\n");
        }
        __syncthreads();
        if(threadIdx.x == 0) for(int lower = 0; lower < LAYER; lower++) {
            int viaSum = 0, minCost = INF, bestpos[3] = {0, 0, 0}, costs[3] = {points[t + 3] == -1 ? 0 : INF, points[t +
4] == -1 ? 0 : INF, points[t + 5] == -1 ? 0 : INF}; for(int upper = lower; upper < LAYER; upper++) { int idx = upper * N
* N + (((upper & 1) ^ DIRECTION) ? tox * N + toy : toy * N + tox); for(int j = 0; j < 3; j++) if(points[t + 3 + j] !=
-1) { if(map[idx + (j + 1) * LAYER * N * N] < costs[j]) costs[j] = map[idx + (j + 1) * LAYER * N * N], bestpos[j] =
upper;
                }
                //if(upper == 0)
                //    printf("cost[0] = %d\n", costs[0]);
                if(1LL * costs[0] + costs[1] + costs[2] < minCost)
                    minCost = costs[0] + costs[1] + costs[2];
                if(minlayer == -1 || (lower <= minlayer && maxlayer <= upper)) {
                    //printf("upper %d viaSUm %d\n", upper, viaSum);
                    for(int i = lower; i <= upper; i++)  {
                        int index = i * N * N + (((i & 1) ^ DIRECTION) ? tox * N + toy : toy * N + tox);
                        if(map[index] > minCost + viaSum)
                            map[index] = minCost + viaSum, prev[index] = (((lower * LAYER + upper) * LAYER + bestpos[2])
* LAYER + bestpos[1]) * LAYER + bestpos[0];
                    }
                }
                viaSum += viaCost[idx];
            }
        }
    }
    __syncthreads();
    if(threadIdx.x) return;
    int minTotalDist = INF;
    for(int i = 0; i < LAYER; i++) {
        int x = points[0] / N, y = points[0] % N;
        int idx = i * N * N + (((i & 1) ^ DIRECTION) ? x * N + y : y * N + x);
        if(map[idx] < minTotalDist)
            minTotalDist = map[idx], map[4 * LAYER * N * N + points[0]] = i;
    }
    if(minTotalDist == INF || minTotalDist < 0) {
        printf("INF is too small!\n");

        return;
    }
    for(int t = 0; t < 6 * node_cnt; t += 6) {
        int x = points[t] / N, y = points[t] % N, layer = map[4 * LAYER * N * N + points[t]];
        int idx = layer * N * N + (((layer & 1) ^ DIRECTION) ? x * N + y : y * N + x);
        int minlayer = prev[idx] / LAYER / LAYER / LAYER / LAYER, maxlayer = prev[idx] / LAYER / LAYER / LAYER % LAYER;
        //printf("%d [%d, %d] exit/min/max layer\n", layer, minlayer, maxlayer);
        for(int i = minlayer; i < maxlayer; i++) {
            int id = i * N * N + (((i & 1) ^ DIRECTION) ? x * N + y : y * N + x);
            routes[routes[0]++] = id;
            routes[routes[0]++] = -1;
            atomicAdd(vias + id, 1);
        }
        int bestpos = prev[idx];
        for(int j = 0; j < 3; j++) {
            int child = points[t + 3 + j];
            if(points[t + 3 + j] == -1) break;
            int entrylayer = bestpos % LAYER, entry = entrylayer * N * N + (((entrylayer & 1) ^ DIRECTION) ? x * N + y :
y * N + x);
            //printf("(%d, %d, %d) ", entrylayer, x, y);
            bestpos /= LAYER;
            int last = prev[(j + 1) * LAYER * N * N + entry];
            int lastlayer = last / N / N, lastx = last / N % N, lasty = last % N;
            if(!(lastlayer & 1) ^ DIRECTION) cudaSwap(lastx, lasty);
            if(lastx != x && lasty != y) {
                int _x = ((entrylayer & 1) ^ DIRECTION) ? x : lastx;
                int _y = ((entrylayer & 1) ^ DIRECTION) ? lasty : y;
                int p1 = ((entrylayer & 1) ^ DIRECTION) ? entry - y + lasty : entry - x + lastx;
                int p2 = ((lastlayer & 1) ^ DIRECTION) ? last - lasty + y : last - lastx + x;
                routes[routes[0]++] = min(entry, p1);
                routes[routes[0]++] = max(entry, p1) - min(entry, p1);
                for(int i = min(entry, p1); i < max(entry, p1); i++)
                    atomicAdd(wires + i, 1);
                routes[routes[0]++] = min(last, p2);
                routes[routes[0]++] = max(last, p2) - min(last, p2);
                for(int i = min(last, p2); i < max(last, p2); i++)
                    atomicAdd(wires + i, 1);
                for(int i = min(lastlayer, entrylayer); i < max(lastlayer, entrylayer); i++) {
                    int id = i * N * N + (((i & 1) ^ DIRECTION) ? _x * N + _y : _y * N + _x);
                    routes[routes[0]++] = id;
                    routes[routes[0]++] = -1;
                    atomicAdd(vias + id, 1);
                }
            } else {
                routes[routes[0]++] = min(last, entry);
                routes[routes[0]++] = max(last, entry) - min(last, entry);
                for(int i = min(last, entry); i < max(last, entry); i++)
                    atomicAdd(wires + i, 1);
            }
            //if(x == 46 && y == 47)
            //    printf("(%d %d %d) <- (%d %d %d)\n", entrylayer, entry / N % N, entry % N, last / N / N, last / N % N,
last % N); map[4 * LAYER * N * N + child] = lastlayer;
        }
    }
    if(debug) for(int i = 1; i < routes[0]; i += 2) {
        int layer = routes[i] / N / N, x = routes[i] / N % N, y = routes[i] % N;
        if((layer & 1) ^ DIRECTION) {
            if(routes[i + 1] == -1)
                printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, x, y, layer + 1, x, y);
            else
                printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, x, y, layer, x, y + routes[i + 1]);
        } else {
            if(routes[i + 1] == -1)
                printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, y, x, layer + 1, y, x);
            else
                printf("{%d, %d, %d}, {%d, %d, %d}, ", layer, y, x, layer, y + routes[i + 1], x);
        }
    }
}*/
__global__ void preprocess(int *wires, int *vias) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    wires[index] <<= 1;
    vias[index] <<= 1;
}
__global__ void postprocess(int *wires, int *vias) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    wires[index] = wires[index] / 2 + wires[index] % 2;
    vias[index] = vias[index] / 2 + vias[index] % 2;
}

__global__ void cudaWrite(int *wires, int *vias, int N, int LAYER) {
    for (int i = 0; i < LAYER; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++) printf("%d\n", vias[i * N * N + j * N + k]);
}

void patternRoute(int *points,
                  int batchSize,
                  int64_t *wireCostSum,
                  int *viaCost,
                  int *map,
                  int *prev,
                  int *wires,
                  int *vias,
                  int *routes,
                  int *gbpoints,
                  int *gbpinRoutes,
                  int X,
                  int Y,
                  int N,
                  int LAYER,
                  int DIRECTION) {
    // debug = 1;
    initMap<<<6 * LAYER * N * 2, N / 2>>>(map);
    // cudaDeviceSynchronize();
    // static int cnt = 11395;
    // cudaLshapePR<<<batchSize, 1>>> (map, points, wireCostSum, viaCost, prev, wires, vias, routes, LAYER, N,
    // DIRECTION);
    // for (int i = 0; i < batchSize; i++) {
    //     cudaPatternRouteParallel<<<1, 32>>>(
    //         map, points + points[i], wireCostSum, viaCost, prev, prev + 10 * LAYER * N * N,
    //         wires, vias, routes, gbpoints + gbpoints[i], gbpinRoutes, LAYER, N, DIRECTION);
    // }
    cudaPatternRouteParallel<<<batchSize, 32>>>(
        map, points, wireCostSum, viaCost, prev, prev + 10 * LAYER * N * N, 
        wires, vias, routes, gbpoints, gbpinRoutes, LAYER, N, DIRECTION);

    // cudaPatternRoute<<<batchSize, LAYER>>> (map, points, wireCostSum, viaCost, prev, wires, vias, routes, LAYER, N,
    // DIRECTION);
    {
        cudaDeviceSynchronize();
        int errorType = cudaGetLastError();
        if (errorType) {
            std::cerr << "PR1 CUDA ERROR: " << errorType << std::endl;
            exit(0);
        }
    }
    // exit(0);
    // if(--cnt == 0) exit(0);
}

}  // namespace gr