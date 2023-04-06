#include <curand.h>
#include <curand_kernel.h>

#include "detailed_place_db.cuh"

#include "gpudp/dp/ism/apply_solution.cuh"
#include "gpudp/dp/ism/auction.cuh"
#include "gpudp/dp/ism/collect_independent_sets.cuh"
#include "gpudp/dp/ism/cost_matrix_construction.cuh"
#include "gpudp/dp/ism/cpu_state.cuh"
#include "gpudp/dp/ism/maximal_independent_set.cuh"
#include "gpudp/dp/ism/shuffle.cuh"

namespace dp {

#define DETERMINISTIC

#define NUM_NODE_SIZES 64  ///< number of different cell sizes

struct SizedBinIndex {
    int size_id;
    int bin_id;
};

template <typename T>
struct IndependentSetMatchingState {
    typedef T type;
    typedef int cost_type;

    int* ordered_nodes = nullptr;
    Space<T>* spaces = nullptr;  ///< array of cell spaces, each cell only consider the space on its left side except
                                 ///< for the left and right boundary
    int num_node_sizes;          ///< number of cell sizes considered
    int* independent_sets = nullptr;       ///< independent sets, length of batch_size*set_size
    int* independent_set_sizes = nullptr;  ///< size of each independent set
    int* selected_maximal_independent_set = nullptr;  ///< storing the selected maximum independent set
    int* select_scratch = nullptr;                    ///< temporary storage for selection kernel
    int num_selected;                                 ///< maximum independent set size
    int* device_num_selected;                         ///< maximum independent set size

    double* net_hpwls;  ///< HPWL for each net, use integer to get consistent values

    int* selected_markers = nullptr;  ///< must be int for cub to compute prefix sum
    unsigned char* dependent_markers = nullptr;
    int* independent_set_empty_flag = nullptr;  ///< a stopping flag for maximum independent set
    int num_independent_sets;  ///< host copy

    cost_type* cost_matrices = nullptr;       ///< cost matrices batch_size*set_size*set_size
    cost_type* cost_matrices_copy = nullptr;  ///< temporary copy of cost matrices
    int* solutions = nullptr;                 ///< batch_size*set_size
    char* auction_scratch = nullptr;          ///< temporary memory for auction solver
    char* stop_flags = nullptr;               ///< record stopping status from auction solver
    T* orig_x = nullptr;                      ///< original locations of cells for applying solutions
    T* orig_y = nullptr;
    cost_type* orig_costs = nullptr;      ///< original costs
    cost_type* solution_costs = nullptr;  ///< solution costs
    Space<T>* orig_spaces = nullptr;      ///< original spaces of cells for apply solutions

    int batch_size;  ///< pre-allocated number of independent sets
    int set_size;
    int cost_matrix_size;   ///< set_size*set_size
    int num_bins;           ///< num_bins_x*num_bins_y
    int* device_num_moved;  ///< device copy
    int num_moved;          ///< host copy, number of moved cells
    int large_number;       ///< a large number

    float auction_max_eps;       ///< maximum epsilon for auction solver
    float auction_min_eps;       ///< minimum epsilon for auction solver
    float auction_factor;        ///< decay factor for auction epsilon
    int auction_max_iterations;  ///< maximum iteration
    T skip_threshold;            ///< ignore connections if cells are far apart
};

template <typename T>
__global__ void iota(T* a, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        a[i] = i;
    }
}

__global__ void cost_matrix_init(int* cost_matrix, int set_size) {
    for (int i = blockIdx.x; i < set_size; i += gridDim.x) {
        for (int j = threadIdx.x; j < set_size; j += blockDim.x) {
            cost_matrix[i * set_size + j] = (i == j) ? 0 : cuda::numeric_limits<int>::max();
        }
    }
}

template <typename T>
__global__ void print_global(T* a, int n) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0) {
        printf("[%d]\n", n);
        for (int i = 0; i < n; ++i) {
            printf("%g ", (double)a[i]);
        }
        printf("\n");
    }
}

template <typename T>
__global__ void print_cost_matrix(const T* cost_matrix, int set_size, bool major) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0) {
        printf("[%dx%d]\n", set_size, set_size);
        for (int r = 0; r < set_size; ++r) {
            for (int c = 0; c < set_size; ++c) {
                if (major)  // column major
                {
                    printf("%g ", (double)cost_matrix[c * set_size + r]);
                } else {
                    printf("%g ", (double)cost_matrix[r * set_size + c]);
                }
            }
            printf("\n");
        }
        printf("\n");
    }
}

template <typename T>
__global__ void print_solution(const T* solution, int n) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0) {
        printf("[%d]\n", n);
        for (int i = 0; i < n; ++i) {
            printf("%g ", (double)solution[i]);
        }
        printf("\n");
    }
}

void construct_spaces(DetailedPlaceData& db,
                      const float* host_x,
                      const float* host_y,
                      const float* host_node_size_x,
                      const float* host_node_size_y,
                      std::vector<Space<float>>& host_spaces,
                      int num_threads) {
    std::vector<std::vector<int> > row2node_map(db.num_sites_y);
    db.make_row2node_map(host_x, host_y, host_node_size_x, host_node_size_y, db.num_nodes, row2node_map);

    // construct spaces
    host_spaces.resize(db.num_movable_nodes);
    for (int i = 0; i < db.num_sites_y; ++i) {
        for (unsigned int j = 0; j < row2node_map[i].size(); ++j) {
            auto const& row2nodes = row2node_map[i];
            int node_id = row2nodes[j];
            auto& space = host_spaces[node_id];
            if (node_id < db.num_movable_nodes) {
                auto left_bound = db.xl;
                if (j) {
                    left_bound = host_x[node_id];
                }
                space.xl = ceilDiv(left_bound - db.xl, db.site_width) * db.site_width + db.xl;

                auto right_bound = db.xh;
                if (j + 1 < row2nodes.size()) {
                    int right_node_id = row2nodes[j + 1];
                    right_bound = min(right_bound, host_x[right_node_id]);
                }
                space.xh = std::floor(right_bound);
                space.xh = floorDiv(space.xh - db.xl, db.site_width) * db.site_width + db.xl; 
            }
        }
    }
}

void independentSetMatchingCUDA(DPTorchRawDB& at_db, int num_bins_x, int num_bins_y, int batch_size, int set_size, int max_iters) {
    cudaSetDevice(at_db.node_size_x.get_device());
    DetailedPlaceData db(at_db);
    db.set_num_bins(num_bins_x, num_bins_y);
    // fix random seed
    std::srand(1000);

    IndependentSetMatchingState<float> state;

    // initialize host database
    DetailedPlaceCPUDB<float> host_db;
    init_cpu_db(db, host_db);

    state.batch_size = batch_size;
    state.set_size = set_size;
    state.cost_matrix_size = state.set_size * state.set_size;
    state.num_bins = db.num_bins_x * db.num_bins_y;
    state.num_moved = 0;
    state.large_number = ((db.xh - db.xl) + (db.yh - db.yl)) * set_size;
    state.skip_threshold = ((db.xh - db.xl) + (db.yh - db.yl)) * 0.01;
    state.auction_max_eps = 10.0;
    state.auction_min_eps = 1.0;
    state.auction_factor = 0.1;
    state.auction_max_iterations = 9999;

    checkCuda(cudaMemcpy(host_db.x.data(), db.x, sizeof(float) * db.num_nodes, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(host_db.y.data(), db.y, sizeof(float) * db.num_nodes, cudaMemcpyDeviceToHost));
    std::vector<Space<float>> host_spaces(db.num_movable_nodes);
    construct_spaces(db,
                     host_db.x.data(),
                     host_db.y.data(),
                     host_db.node_size_x.data(),
                     host_db.node_size_y.data(),
                     host_spaces,
                     db.num_threads);

    // initialize cuda state

    allocateCopyCuda(state.spaces, host_spaces.data(), db.num_movable_nodes);
    allocateCuda(state.ordered_nodes, db.num_movable_nodes, int);
    iota<<<ceilDiv(db.num_movable_nodes, 512), 512>>>(state.ordered_nodes, db.num_movable_nodes);
    allocateCuda(state.independent_sets, state.batch_size * state.set_size, int);
    allocateCuda(state.independent_set_sizes, state.batch_size, int);
    allocateCuda(state.selected_maximal_independent_set, db.num_movable_nodes, int);
    allocateCuda(state.select_scratch, db.num_movable_nodes, int);
    allocateCuda(state.device_num_selected, 1, int);
    allocateCuda(state.orig_x, state.batch_size * state.set_size, float);
    allocateCuda(state.orig_y, state.batch_size * state.set_size, float);
    allocateCuda(state.orig_spaces, state.batch_size * state.set_size, Space<float>);
    allocateCuda(state.selected_markers, db.num_nodes, int);
    allocateCuda(state.dependent_markers, db.num_nodes, unsigned char);
    allocateCuda(state.independent_set_empty_flag, 1, int);
    allocateCuda(state.cost_matrices,
                 state.batch_size * state.set_size * state.set_size,
                 typename IndependentSetMatchingState<float>::cost_type);
    allocateCuda(state.cost_matrices_copy,
                 state.batch_size * state.set_size * state.set_size,
                 typename IndependentSetMatchingState<float>::cost_type);
    allocateCuda(state.solutions, state.batch_size * state.set_size, int);
    allocateCuda(
        state.orig_costs, state.batch_size * state.set_size, typename IndependentSetMatchingState<float>::cost_type);
    allocateCuda(state.solution_costs,
                 state.batch_size * state.set_size,
                 typename IndependentSetMatchingState<float>::cost_type);
    allocateCuda(state.net_hpwls, db.num_nets, typename std::remove_pointer<decltype(state.net_hpwls)>::type);
    allocateCopyCuda(state.device_num_moved, &state.num_moved, 1);

    init_auction<float>(state.batch_size, state.set_size, state.auction_scratch, state.stop_flags);

    Shuffler<int, unsigned int> shuffler(2023ULL, state.ordered_nodes, db.num_movable_nodes);

    // initialize host state
    IndependentSetMatchingCPUState<float> host_state;
    init_cpu_state(db, state, host_state);

    // initialize kmeans state
    KMeansState<float> kmeans_state;
    init_kmeans(db, state, kmeans_state);

    std::vector<float> hpwls(max_iters + 1);
    hpwls[0] = compute_total_hpwl(db, db.x, db.y, state.net_hpwls);
    logger.info("initial hpwl %g", hpwls[0]);
    for (int iter = 0; iter < max_iters; ++iter) {
        shuffler();
        checkCuda(cudaDeviceSynchronize());

        maximal_independent_set(db, state);
        checkCuda(cudaDeviceSynchronize());

        collect_independent_sets(db, state, kmeans_state, host_db, host_state);
        checkCuda(cudaDeviceSynchronize());

        cost_matrix_construction(db, state);
        checkCuda(cudaDeviceSynchronize());

        // solve independent sets
        // print_cost_matrix<<<1, 1>>>(state.cost_matrices + state.cost_matrix_size*3, state.set_size, 0);
        linear_assignment_auction(state.cost_matrices,
                                  state.solutions,
                                  state.num_independent_sets,
                                  state.set_size,
                                  state.auction_scratch,
                                  state.stop_flags,
                                  state.auction_max_eps,
                                  state.auction_min_eps,
                                  state.auction_factor,
                                  state.auction_max_iterations);
        checkCuda(cudaDeviceSynchronize());
        // print_solution<<<1, 1>>>(state.solutions + state.set_size*3, state.set_size);

        // apply solutions
        apply_solution(db, state);
        checkCuda(cudaDeviceSynchronize());

        hpwls[iter + 1] = compute_total_hpwl(db, db.x, db.y, state.net_hpwls);
        if ((iter % (max(max_iters / 10, 1))) == 0 || iter + 1 == max_iters) {
            logger.info("iteration %d, target hpwl %g, delta %g(%g%%), %d independent sets, moved %g%% cells",
                        iter,
                        hpwls[iter + 1],
                        hpwls[iter + 1] - hpwls[0],
                        (hpwls[iter + 1] - hpwls[0]) / hpwls[0] * 100,
                        state.num_independent_sets,
                        state.num_moved / (double)db.num_movable_nodes * 100);
        }
    }

    // destroy state
    cudaFree(state.spaces);
    cudaFree(state.ordered_nodes);
    cudaFree(state.independent_sets);
    cudaFree(state.independent_set_sizes);
    cudaFree(state.selected_maximal_independent_set);
    cudaFree(state.select_scratch);
    cudaFree(state.device_num_selected);
    cudaFree(state.net_hpwls);
    cudaFree(state.cost_matrices);
    cudaFree(state.cost_matrices_copy);
    cudaFree(state.solutions);
    cudaFree(state.orig_costs);
    cudaFree(state.solution_costs);
    cudaFree(state.orig_x);
    cudaFree(state.orig_y);
    cudaFree(state.orig_spaces);
    cudaFree(state.selected_markers);
    cudaFree(state.dependent_markers);
    cudaFree(state.independent_set_empty_flag);
    cudaFree(state.device_num_moved);
    destroy_auction(state.auction_scratch, state.stop_flags);
    destroy_kmeans(kmeans_state);
}

}  // namespace dp