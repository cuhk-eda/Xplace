#pragma once
#include "common/common.h"
#include "gpugr/db/GrNet.h"

namespace gr {

int prepare(double &count,
            gr::GrNet &grNet,
            int *points,
            int routesOffset,
            int *gbpoints,
            int &gbPinOffset,
            int X,
            int Y,
            int N,
            int LAYER,
            int DIRECTION);

void prepareSingeNet(gr::GrNet &grNet, int routesOffset, int X, int Y, int N, int LAYER, int DIRECTION);

void prepareGrNets(std::vector<gr::GrNet> &grNets,
                   std::vector<int> &netsToRoute,
                   std::vector<int> &batchSizes,
                   std::vector<std::vector<int>> &points_cpu_vec,
                   std::vector<std::tuple<int, int, int>> &batchId2vec_info,
                   int *routesOffsetCPU,
                   int X,
                   int Y,
                   int N,
                   int LAYER,
                   int DIRECTION);

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
                  int DIRECTION);
}  // namespace gr