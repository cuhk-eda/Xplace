#include "PatternRoute.h"

#include <iostream>
#include <set>

#include "common/db/Database.h"
#include "common/utils/robin_hood.h"
#include "flute.h"

namespace gr {

using namespace Flute;

void prepareSingeNet(gr::GrNet &grNet, int routesOffset, int X, int Y, int N, int LAYER, int DIRECTION) {
    const std::vector<std::vector<int>> &pins = grNet.getPins();
    std::vector<int> &points = grNet.points;
    points.clear();

    robin_hood::unordered_map<int, std::vector<int>> loc2Pins;
    // double startTimer = clock();
    std::vector<int> xpos(pins.size()), ypos(pins.size());
    for (int i = 0; i < pins.size(); i++) {
        int layer = pins[i][0] / N / N, _x = pins[i][0] / N % N, _y = pins[i][0] % N;
        if (!(layer & 1) ^ DIRECTION) std::swap(_x, _y);
        xpos[i] = _x;
        ypos[i] = _y;
        loc2Pins[_x * N + _y].emplace_back(layer);
    }

    std::sort(xpos.begin(), xpos.end());
    std::sort(ypos.begin(), ypos.end());
    xpos.erase(std::unique(xpos.begin(), xpos.end()), xpos.end());
    ypos.erase(std::unique(ypos.begin(), ypos.end()), ypos.end());
    int degree = loc2Pins.size(), cur = 0;
    if (degree == 0) std::cerr << "ERROR: degree 0" << std::endl;
    // const int MAX_DEGREE = 100000;
    // if (degree > MAX_DEGREE) std::cerr << "Not Enough X and Y in Pattern Routing" << std::endl;

    int x[degree * 4], y[degree * 4];
    for (auto e : loc2Pins) x[cur] = e.first / N, y[cur] = e.first % N, cur++;

    Tree flutetree = flute(degree, x, y, 3);

    robin_hood::unordered_map<int, int> loc2node, node2loc;
    std::set<int> locations;
    int node_cnt = 0;
    for (int i = 0; i < degree * 2 - 2; i++) locations.insert(flutetree.branch[i].x * N + flutetree.branch[i].y);
    for (auto e : locations) node2loc[loc2node[e] = node_cnt++] = e;
    std::vector<robin_hood::unordered_set<int>> graph(node_cnt);

    std::vector<std::vector<int>> cntx(xpos.size(), std::vector<int>(ypos.size(), 0));
    std::vector<std::vector<int>> cnty(xpos.size(), std::vector<int>(ypos.size(), 0));
    std::vector<std::vector<int>> idx(xpos.size(), std::vector<int>(ypos.size(), -1));
    for (auto e : loc2node) {
        int x = std::lower_bound(xpos.begin(), xpos.end(), e.first / N) - xpos.begin();
        int y = std::lower_bound(ypos.begin(), ypos.end(), e.first % N) - ypos.begin();
        // printf("%d %d -> %d\n", x, y, e.second);
        idx[x][y] = e.second;
    }

    for (int i = 0; i < degree * 2 - 2; i++) {
        Branch &branch1 = flutetree.branch[i], &branch2 = flutetree.branch[branch1.n];
        int id1 = loc2node[branch1.x * N + branch1.y], id2 = loc2node[branch2.x * N + branch2.y];
        if (id1 == id2) continue;
        int x1 = node2loc[id1] / N, y1 = node2loc[id1] % N;
        int x2 = node2loc[id2] / N, y2 = node2loc[id2] % N;
        // printf("%d %d %d %d\n", x1, y1, x2, y2);
        x1 = std::lower_bound(xpos.begin(), xpos.end(), x1) - xpos.begin();
        x2 = std::lower_bound(xpos.begin(), xpos.end(), x2) - xpos.begin();
        y1 = std::lower_bound(ypos.begin(), ypos.end(), y1) - ypos.begin();
        y2 = std::lower_bound(ypos.begin(), ypos.end(), y2) - ypos.begin();
        // printf("%d %d %d %d\n", x1, y1, x2, y2);
        if (x1 != x2 && y1 != y2) {
            graph[id1].insert(id2), graph[id2].insert(id1);
            /*    if(locations.count(x1 * N + y2) || locations.count(x2 * N + y1))
                    std::cerr << "ERROR & ERROR: BAD FLUTE RESULTS\n";
                for(int i = std::min(x1, x2); i <= std::max(x1, x2); i++)
                    if(locations.count(i * N + y1) || locations.count(i * N + y2))
                        std::cerr << "ERROR: BAD FLUTE RESULTS\n";
                for(int i = std::min(y1, y2); i <= std::max(y1, y2); i++)
                    if(locations.count(x1 * N + i) || locations.count(x2 * N + i))
                        std::cerr << "ERROR: BAD FLUTE RESULTS\n";
            */
        } else {
            if (x1 == x2)
                for (int t = std::min(y1, y2); t < std::max(y1, y2); t++) cnty[x1][t]++;
            else
                for (int t = std::min(x1, x2); t < std::max(x1, x2); t++) cntx[t][y2]++;
        }
    }
    free(flutetree.branch);
    // printf("cnt = %d, %d %d\n", cntx[0][0], idx[0][0], idx[1][0]);
    for (int i = 0; i < xpos.size(); i++) {
        int last = -1;
        for (int j = 0; j < (int)ypos.size(); j++) {
            if (j && cnty[i][j - 1] == 0) last = -1;
            int cur = -1;
            if (idx[i][j] >= 0) cur = idx[i][j];
            if (cur >= 0 && last >= 0 && cur != last) graph[cur].insert(last), graph[last].insert(cur);
            if (cur >= 0) last = cur;
        }
    }
    for (int i = 0; i < ypos.size(); i++) {
        int last = -1;
        for (int j = 0; j < (int)xpos.size(); j++) {
            if (j && cntx[j - 1][i] == 0) last = -1;
            int cur = -1;
            if (idx[j][i] >= 0) cur = idx[j][i];
            if (cur >= 0 && last >= 0 && cur != last) graph[cur].insert(last), graph[last].insert(cur);
            // printf("i=%d, j=%d, cur=%d,last=%d\n", i, j, cur, last);
            if (cur >= 0) last = cur;
        }
    }
    std::vector<int> vis(node_cnt, 0);
    for (int i = 0; i < node_cnt; i++) {
        if (graph[i].size() > 4) {
            std::cerr << "ERROR in FLUTE Results\n";
            exit(-1);
        }
    }
    points.emplace_back(routesOffset);
    points.emplace_back(node_cnt);
    int len = 0;
    // for each point in points
    // points[0]: location
    // points[1, 2]: min and max layer
    // points[3, 4, 5, 6] children locations

    std::function<void(int)> dfs = [&](int x) {
        vis[x] = 1;
        int startlen = len;
        points.emplace_back(node2loc[x]);
        len++;
        if (loc2Pins.count(node2loc[x])) {
            auto temp = loc2Pins[node2loc[x]];
            points.emplace_back(*std::min_element(temp.begin(), temp.end()));
            points.emplace_back(*std::max_element(temp.begin(), temp.end()));
            len += 2;
        } else {
            points.emplace_back(-1);
            points.emplace_back(-1);
            len += 2;
        }
        for (auto e : graph[x]) {
            if (!vis[e]) {
                points.emplace_back(node2loc[e]);
                len++;
            }
        }
        if (len - startlen > 6) {
            printf(" %d ERROR in len\n", (int)graph[x].size());
        }
        while (len % 6 != 0) {
            points.emplace_back(-1);
            len++;
        }
        for (auto e : graph[x]) {
            if (!vis[e]) dfs(e);
        }
    };

    dfs(0);
    if (len != 6 * node_cnt) {
        for (int i = 0; i < pins.size(); i++) {
            int layer = pins[i][0] / N / N, _x = pins[i][0] / N % N, _y = pins[i][0] % N;
            if (!(layer & 1) ^ DIRECTION) std::swap(_x, _y);
            printf("(%d, %d)\n", _x, _y);
        }
        for (auto e : loc2node) printf("(%d, %d)_%d  ", e.first / N, e.first % N, e.second);
        puts("");
        for (int i = 0; i < node_cnt; i++)
            for (auto e : graph[i]) printf("E(%d, %d) ", i, e);
        puts("");
        std::cerr << "ERROR in pattern routing preparation" << std::endl;
    }
}

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
                   int DIRECTION) {
    // FIXME: the vanilla version of FLUTE cannot support multi-threads. If need MT, please
    // change the FLUTE to the version in https://github.com/The-OpenROAD-Project-Attic/flute3
    int totalBs = 0;
    int numThreads = 1;  // multi-thread is not supported by FLUTE
    for (int batchSize : batchSizes) {
        totalBs += batchSize;
    }
    auto thread_func = [&](int threadIdx) {
        for (int i = threadIdx; i < totalBs; i += numThreads) {
            int netId = netsToRoute[i];
            prepareSingeNet(grNets[netId], routesOffsetCPU[netId], X, Y, N, LAYER, DIRECTION);
        }
    };
    std::thread threads[numThreads];
    for (int j = 0; j < numThreads; j++) {
        threads[j] = std::thread(thread_func, j);
    }
    for (auto &t : threads) {
        t.join();
    }
    logger.info("Finish Flute %d");

    points_cpu_vec.clear();
    batchId2vec_info.clear();

    batchId2vec_info.resize(batchSizes.size());
    int startpos = 0;
    constexpr int MAX_POINTS_SIZE = 20000000;
    points_cpu_vec.push_back(std::vector<int>());
    points_cpu_vec.back().reserve(MAX_POINTS_SIZE);
    for (int batchId = 0; batchId < batchSizes.size(); batchId++) {
        int batchSize = batchSizes[batchId];
        if (batchSize == 0) continue;
        int offset = batchSize;
        std::vector<int> curBatch_points(batchSize, -1);
        for (int i = 0; i < batchSize; i++) {
            // the first batchSize elements indicate the offset
            curBatch_points[i] = offset;
            int netId = netsToRoute[startpos + i];
            offset += grNets[netId].points.size();
            // std::cout << batchSize << " " << i << " BigVecId " << points_cpu_vec.size() << " " <<
            // curBatch_points.size() << " " << points_cpu_vec.back().size() << " " << grNets[netId].getPins().size() <<
            // " " << grNets[netId].points.size() << std::endl;
            curBatch_points.insert(curBatch_points.end(),
                                   std::make_move_iterator(grNets[netId].points.begin()),
                                   std::make_move_iterator(grNets[netId].points.end()));
        }
        int startIdx, endIdx, inBigVecId;
        if (points_cpu_vec.back().size() + curBatch_points.size() < MAX_POINTS_SIZE) {
            startIdx = points_cpu_vec.back().size();
            endIdx = startIdx + curBatch_points.size();
            auto &tmp = points_cpu_vec.back();
            tmp.insert(tmp.end(),
                       std::make_move_iterator(curBatch_points.begin()),
                       std::make_move_iterator(curBatch_points.end()));
            inBigVecId = points_cpu_vec.size() - 1;
        } else {
            startIdx = 0;
            endIdx = curBatch_points.size();
            points_cpu_vec.emplace_back(std::move(curBatch_points));
            points_cpu_vec.back().reserve(MAX_POINTS_SIZE);
            inBigVecId = points_cpu_vec.size() - 1;
        }
        batchId2vec_info[batchId] = {inBigVecId, startIdx, endIdx};
        startpos += batchSize;
    }
    logger.info("#BigVec %d", points_cpu_vec.size());
}

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
            int DIRECTION) {
    auto &pins = grNet.getPins();
    // std::map<int, std::vector<int>> loc2Pins;
    robin_hood::unordered_map<int, std::vector<int>> loc2Pins;
    robin_hood::unordered_map<int, int> loc2pinIds;
    // double startTimer = clock();
    std::vector<int> xpos(pins.size()), ypos(pins.size());
    for (int i = 0; i < pins.size(); i++) {
        int layer = pins[i][0] / N / N, _x = pins[i][0] / N % N, _y = pins[i][0] % N;
        if (!(layer & 1) ^ DIRECTION) std::swap(_x, _y);
        xpos[i] = _x;
        ypos[i] = _y;
        auto& loc2PinsLayerVec = loc2Pins[_x * N + _y];
        loc2PinsLayerVec.emplace_back(layer);
        loc2pinIds[_x * N + _y] = i;
        // if(pins[i].size() > 1) {
        //     int layer = pins[i][pins[i].size() - 1] / N / N;
        //     loc2PinsLayerVec.emplace_back(layer);
        // }
    }
    std::sort(xpos.begin(), xpos.end());
    std::sort(ypos.begin(), ypos.end());
    xpos.erase(unique(xpos.begin(), xpos.end()), xpos.end());
    ypos.erase(unique(ypos.begin(), ypos.end()), ypos.end());
    int degree = loc2Pins.size(), cur = 0;
    if (degree == 0) std::cerr << "ERROR: degree 0" << std::endl;
    constexpr int MAX_DEGREE = 100000;
    if (degree > MAX_DEGREE) std::cerr << "Not Enough X and Y in Pattern Routing" << std::endl;
    int x[degree * 4], y[degree * 4];
    for (auto e : loc2Pins) x[cur] = e.first / N, y[cur] = e.first % N, cur++;

    Tree flutetree = flute(degree, x, y, 3);
    // count += clock() - startTimer;
    robin_hood::unordered_map<int, int> loc2node, node2loc;
    std::set<int> locations;
    int node_cnt = 0;
    for (int i = 0; i < degree * 2 - 2; i++) locations.insert(flutetree.branch[i].x * N + flutetree.branch[i].y);
    for (auto e : locations) {
        node2loc[loc2node[e] = node_cnt++] = e;
        if (!loc2pinIds.contains(e)) {
            // e is not a real pin position but is a pseudo pin generated by RSMT
            loc2pinIds[e] = -1;
        }
    }
    // std::vector<std::set<int>> graph(node_cnt);
    std::vector<robin_hood::unordered_set<int>> graph(node_cnt);

    std::vector<std::vector<int>> cntx(xpos.size(), std::vector<int>(ypos.size(), 0));
    std::vector<std::vector<int>> cnty(xpos.size(), std::vector<int>(ypos.size(), 0));
    std::vector<std::vector<int>> idx(xpos.size(), std::vector<int>(ypos.size(), -1));
    for (auto e : loc2node) {
        int x = lower_bound(xpos.begin(), xpos.end(), e.first / N) - xpos.begin();
        int y = lower_bound(ypos.begin(), ypos.end(), e.first % N) - ypos.begin();
        // printf("%d %d -> %d\n", x, y, e.second);
        idx[x][y] = e.second;
    }

    for (int i = 0; i < degree * 2 - 2; i++) {
        Branch &branch1 = flutetree.branch[i], &branch2 = flutetree.branch[branch1.n];
        int id1 = loc2node[branch1.x * N + branch1.y], id2 = loc2node[branch2.x * N + branch2.y];
        if (id1 == id2) continue;
        int x1 = node2loc[id1] / N, y1 = node2loc[id1] % N;
        int x2 = node2loc[id2] / N, y2 = node2loc[id2] % N;
        // printf("%d %d %d %d\n", x1, y1, x2, y2);
        x1 = lower_bound(xpos.begin(), xpos.end(), x1) - xpos.begin();
        x2 = lower_bound(xpos.begin(), xpos.end(), x2) - xpos.begin();
        y1 = lower_bound(ypos.begin(), ypos.end(), y1) - ypos.begin();
        y2 = lower_bound(ypos.begin(), ypos.end(), y2) - ypos.begin();
        // printf("%d %d %d %d\n", x1, y1, x2, y2);
        if (x1 != x2 && y1 != y2) {
            graph[id1].insert(id2), graph[id2].insert(id1);
            /*    if(locations.count(x1 * N + y2) || locations.count(x2 * N + y1))
                    std::cerr << "ERROR & ERROR: BAD FLUTE RESULTS\n";
                for(int i = min(x1, x2); i <= max(x1, x2); i++)
                    if(locations.count(i * N + y1) || locations.count(i * N + y2))
                        std::cerr << "ERROR: BAD FLUTE RESULTS\n";
                for(int i = min(y1, y2); i <= max(y1, y2); i++)
                    if(locations.count(x1 * N + i) || locations.count(x2 * N + i))
                        std::cerr << "ERROR: BAD FLUTE RESULTS\n";
            */
        } else {
            if (x1 == x2)
                for (int t = min(y1, y2); t < max(y1, y2); t++) cnty[x1][t]++;
            else
                for (int t = min(x1, x2); t < max(x1, x2); t++) cntx[t][y2]++;
        }
    }
    free(flutetree.branch);
    // NOTE: When a_x < b_x < c_x and a_y == b_y == c_y, FLUTE may report two edges A-B, A-C,
    //       here we fix it to A-B, B-C
    // printf("cnt = %d, %d %d\n", cntx[0][0], idx[0][0], idx[1][0]);
    for (int i = 0; i < xpos.size(); i++) {
        int last = -1;
        for (int j = 0; j < (int)ypos.size(); j++) {
            if (j && cnty[i][j - 1] == 0) last = -1;
            int cur = -1;
            if (idx[i][j] >= 0) cur = idx[i][j];
            if (cur >= 0 && last >= 0 && cur != last) graph[cur].insert(last), graph[last].insert(cur);
            if (cur >= 0) last = cur;
        }
    }
    for (int i = 0; i < ypos.size(); i++) {
        int last = -1;
        for (int j = 0; j < (int)xpos.size(); j++) {
            if (j && cntx[j - 1][i] == 0) last = -1;
            int cur = -1;
            if (idx[j][i] >= 0) cur = idx[j][i];
            if (cur >= 0 && last >= 0 && cur != last) graph[cur].insert(last), graph[last].insert(cur);
            // printf("i=%d, j=%d, cur=%d,last=%d\n", i, j, cur, last);
            if (cur >= 0) last = cur;
        }
    }
    // NOTE: Fix corner cases, graph[i] includes 4 straight edges and >= 1 bevel edges
    for (int i = 0; i < node_cnt; i++) {
        if (graph[i].size() > 4) {
            std::vector<int> movedIds;
            int thisX = node2loc[i] / N, thisY = node2loc[i] % N;
            for (auto childId : graph[i]) {
                int childX = node2loc[childId] / N, childY = node2loc[childId] % N;
                if (childX != thisX && childY != thisY) {
                    movedIds.emplace_back(childId);
                }
            }
            for (auto childId : movedIds) {
                std::queue<int> q;
                std::vector<bool> possibleSet(node_cnt, false);
                q.push(i);
                while (q.size() > 0) {
                    int cur = q.front();
                    q.pop();
                    if (possibleSet[cur]) continue;
                    possibleSet[cur] = true;
                    for (auto c : graph[cur]) {
                        if (!possibleSet[c] && c != childId) {
                            q.push(c);
                        }
                    }
                }
                if (graph[i].size() == 4) break;
                int childX = node2loc[childId] / N, childY = node2loc[childId] % N;
                int minDist = std::numeric_limits<int>::max();
                int new_i = -1;
                for (int j = 0; j < node_cnt; j++) {
                    if (!possibleSet[j]) continue;
                    if (j == i || j == childId) continue;
                    if (graph[j].size() < 4) {
                        int tarX = node2loc[j] / N, tarY = node2loc[j] % N;
                        int dist = std::abs(tarX - childX) + std::abs(tarY - childY);
                        if (dist == 0) continue;
                        if (dist < minDist) {
                            minDist = dist;
                            new_i = j;
                        }
                    }
                }
                if (new_i == -1) {
                    continue;
                }
                graph[i].erase(childId);
                graph[childId].erase(i);
                graph[childId].insert(new_i);
                graph[new_i].insert(childId);
            }
        }
    }
    for (int i = 0; i < node_cnt; i++) {
        if (graph[i].size() > 4) {
            std::cerr << "ERROR in FLUTE Results\n";
            printf("(%d %d) Childs: ", node2loc[i] / N, node2loc[i] % N);
            for (auto e : graph[i]) {
                printf("(%d %d) ", node2loc[e] / N, node2loc[e] % N);
            }
            printf("\nAll pts: ");
            for (int j = 0; j < node_cnt; j++) {
                printf("(%d %d) ", node2loc[j] / N, node2loc[j] % N);
            }
            std::cout << std::endl;
            exit(-1);
        }
    }
    std::vector<int> vis(node_cnt, 0);
    points[0] = routesOffset;
    points[1] = node_cnt;
    int len = 0;
    points += 2;
    // points[0]: location
    // points[1, 2]: min and max layer
    // points[3, 4, 5] children locations
    std::function<void(int)> dfs = [&](int x) {
        vis[x] = 1;
        int startlen = len;
        // points[0]: location
        int loc = node2loc[x];
        points[len++] = loc;
        if (loc2Pins.count(loc)) {
            // points[1, 2]: min and max layer
            auto temp = loc2Pins[loc];
            points[len++] = *std::min_element(temp.begin(), temp.end());
            points[len++] = *std::max_element(temp.begin(), temp.end());
        } else
            points[len++] = -1, points[len++] = -1;

        // points[3, 4, 5] children locations
        for (auto e : graph[x])
            if (!vis[e]) points[len++] = node2loc[e];
        if (len - startlen > 6) printf(" %d ERROR in len\n", (int)graph[x].size());
        while (len % 6 != 0) points[len++] = -1;
        for (auto e : graph[x])
            if (!vis[e]) dfs(e);
    };

    dfs(0);
    if (len != 6 * node_cnt) {
        printf("len: %d node_cnt: %d\n", len, node_cnt);
        for (int i = 0; i < pins.size(); i++) {
            int layer = pins[i][0] / N / N, _x = pins[i][0] / N % N, _y = pins[i][0] % N;
            if (!(layer & 1) ^ DIRECTION) std::swap(_x, _y);
            printf("(%d, %d)\n", _x, _y);
        }
        for (auto e : loc2node) printf("(%d, %d)_%d  ", e.first / N, e.first % N, e.second);
        puts("");
        for (int i = 0; i < node_cnt; i++)
            for (auto e : graph[i]) printf("E(%d, %d) ", i, e);
        puts("");
        std::cerr << "ERROR in pattern routing preparation" << std::endl;
    }

    // rewrite points child
    robin_hood::unordered_map<int, int> loc2point_id;
    for (int i = 0; i < node_cnt; i++) {
        loc2point_id[points[i * 6]] = i * 6;
    }
    for (int i = 0; i < node_cnt; i++) {
        for (int j = 3; j < 6; j++) {
            if (points[i * 6 + j] == -1) continue;
            points[i * 6 + j] = loc2point_id[points[i * 6 + j]];
        }
    }

    // for (int i = 0; i < node_cnt; i++) {
    //     int x = points[i * 6] / N, y = points[i * 6] % N;
    //     if (x > grNet.upperx || x < grNet.lowerx || y > grNet.uppery || y < grNet.lowery) {
    //         printf("pin: (%d, %d) out of boundary of net_bbox: (%d, %d, %d, %d)\n",
    //             x, y, grNet.lowerx, grNet.lowery, grNet.upperx, grNet.uppery);  
    //     }
    // }

    // gbpoints
    for (int i = 0; i < node_cnt; i++) {
        int pinId = loc2pinIds[points[i * 6]];
        if (pinId == -1) {
            gbpoints[i] = -1;
        } else {
            gbpoints[i] = grNet.pin2gbpinId[pinId];
        }
    }
    gbPinOffset += node_cnt;
    // if (node_cnt > pins.size() && node_cnt > 10) {
    //     std::cout << "node_cnt " << node_cnt << ", #gbpins " << pins.size() << std::endl;
    //     for (int i = 0; i < node_cnt; i++) {
    //         std::cout << points[i * 6] << " ";
    //     }
    //     std::cout << std::endl;
    //     for (int i = 0; i < node_cnt; i++) {
    //         std::cout << gbpoints[i] << " ";
    //     }
    //     std::cout << std::endl;
    //     for (int i = 0; i < node_cnt; i++) {
    //         int loc = points[i * 6];
    //         if (loc2pinIds[loc] != -1) {
    //             int pinid = loc2pinIds[loc];
    //             int layer = pins[pinid][0] / N / N, _x = pins[pinid][0] / N % N, _y = pins[pinid][0] % N;
    //             std::cout << _x * N + _y << " ";
    //         } else {
    //             std::cout << "xxxxxx" << " ";
    //         }
    //     }
    //     std::cout << std::endl;
    //     exit(0);
    // }

    return len + 2;
}

}  // namespace gr