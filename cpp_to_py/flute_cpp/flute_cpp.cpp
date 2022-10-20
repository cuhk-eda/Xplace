#include <torch/extension.h>

#include <mutex>
#include <thread>
#include <vector>

#include "flute.h"

// mt flute, DTYPE is int
void runJobsMT(int numJobs, int num_threads, const std::function<void(int)> &handle) {
    int numThreads = std::min(numJobs, num_threads);
    if (numThreads <= 1) {
        for (int i = 0; i < numJobs; ++i) {
            handle(i);
        }
    } else {
        int globalJobIdx = 0;
        std::mutex mtx;
        auto thread_func = [&](int threadIdx) {
            int jobIdx;
            while (true) {
                mtx.lock();
                jobIdx = globalJobIdx++;
                mtx.unlock();
                if (jobIdx >= numJobs) {
                    break;
                }
                handle(jobIdx);
            }
        };

        std::thread threads[numThreads];
        for (int i = 0; i < numThreads; i++) {
            threads[i] = std::thread(thread_func, i);
        }
        for (int i = 0; i < numThreads; i++) {
            threads[i].join();
        }
    }
}

int FluteRSMTWL(const std::vector<int> &xsvec, const std::vector<int> &ysvec) {
    int degree = xsvec.size();
    int rsmt_wl = 0;
    if (degree > 1) {
        Flute::DTYPE xs[5 * degree];
        Flute::DTYPE ys[5 * degree];
        for (int pt_cnt = 0; pt_cnt < degree; pt_cnt++) {
            xs[pt_cnt] = xsvec[pt_cnt];
            ys[pt_cnt] = ysvec[pt_cnt];
        }
        rsmt_wl = Flute::flute_wl(degree, xs, ys, FLUTE_ACCURACY);
    }
    return rsmt_wl;
}

std::vector<int> MultiNetsFluteRSMTWL(const std::vector<int> &pos_x,
                                      const std::vector<int> &pos_y,
                                      const std::vector<int64_t> &hyperedge_list,
                                      const std::vector<int64_t> &hyperedge_list_end,
                                      const int num_threads) {
    int num_hyperedges = hyperedge_list_end.size();
    std::vector<int> nets_rmst(num_hyperedges, 0.0);
    std::function<void(int)> singleNetFluteWL = [&](int i) {
        if (i >= num_hyperedges) return;
        int rsmt_wl = 0.0;
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = hyperedge_list_end[i - 1];
        }
        int64_t end_idx = hyperedge_list_end[i];
        int64_t degree = end_idx - start_idx;
        if (degree > 1) {
            Flute::DTYPE xs[5 * degree];
            Flute::DTYPE ys[5 * degree];
            int64_t pt_cnt = 0;
            for (int64_t idx = start_idx; idx < end_idx; idx++) {
                int64_t pos_idx = hyperedge_list[idx];
                xs[pt_cnt] = pos_x[pos_idx];
                ys[pt_cnt] = pos_y[pos_idx];
                pt_cnt++;
            }
            rsmt_wl = Flute::flute_wl(degree, xs, ys, FLUTE_ACCURACY);
        }
        nets_rmst[i] = rsmt_wl;
    };
    runJobsMT(num_hyperedges, num_threads, singleNetFluteWL);
    return nets_rmst;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flute_rsmt_wl", &FluteRSMTWL, "Get Flute wirelength");
    m.def("flute_rsmt_wl_mt", &MultiNetsFluteRSMTWL, "Get Multi Nets Flute wirelength (MT Version)");
    m.def("read_lut", &Flute::readLUT, "Read Flute LUT");
}
