#pragma once

#include "gpudp/dp/detailed_place_db.cuh"
#include "cpu_state.cuh"

namespace dp {

#define DETERMINISTIC

template <typename T>
struct KMeansState {
#ifdef DETERMINISTIC
    typedef long long int coordinate_type;
#else
    typedef T coordinate_type;
#endif

    coordinate_type* centers_x;  // To ensure determinism, use fixed point numbers
    coordinate_type* centers_y;
    T* weights;
    int* partition_sizes;
    int* node2centers_map;
    int num_seeds;

#ifdef DETERMINISTIC
    static constexpr T scale = 16384;
#else
    static constexpr T scale = 1;
#endif
};

/// @brief A wrapper for atomicAdd
/// As CUDA atomicAdd does not support for long long int, using unsigned long long int is equivalent.
template <typename T>
inline __device__ T atomicAddWrapper(T* address, T value) {
    return atomicAdd(address, value);
}

/// @brief Template specialization for long long int
template <>
inline __device__ long long int atomicAddWrapper<long long int>(long long int* address, long long int value) {
    return atomicAdd((unsigned long long int*)address, (unsigned long long int)value);
}

template <typename T>
__global__ void fill_array_kernel(T* array, int n, T v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    array[i] = v;
  }
}

template <typename T>
inline void fill_array(T* array, int n, T v) {
  fill_array_kernel<<<ceilDiv(n, 512), 512>>>(array, n, v);
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void init_kmeans(const DetailedPlaceDBType& db,
                 const IndependentSetMatchingStateType& state,
                 KMeansState<typename DetailedPlaceDBType::type>& kmeans_state) {
    typedef typename DetailedPlaceDBType::type T;

    allocateCuda(kmeans_state.centers_x,
                 state.batch_size,
                 typename KMeansState<typename DetailedPlaceDBType::type>::coordinate_type);
    allocateCuda(kmeans_state.centers_y,
                 state.batch_size,
                 typename KMeansState<typename DetailedPlaceDBType::type>::coordinate_type);
    allocateCuda(kmeans_state.weights, state.batch_size, T);
    allocateCuda(kmeans_state.partition_sizes, state.batch_size, int);
    allocateCuda(kmeans_state.node2centers_map, db.num_movable_nodes, int);
}

template <typename T>
void destroy_kmeans(KMeansState<T>& kmeans_state) {
    cudaFree(kmeans_state.centers_x);
    cudaFree(kmeans_state.centers_y);
    cudaFree(kmeans_state.weights);
    cudaFree(kmeans_state.partition_sizes);
    cudaFree(kmeans_state.node2centers_map);
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void prepare_kmeans(const DetailedPlaceDBType& db,
                    const IndependentSetMatchingStateType& state,
                    KMeansState<typename DetailedPlaceDBType::type>& kmeans_state) {
    // need at least 1 seed; otherwise, it will cause problem in later kernels
    kmeans_state.num_seeds = max(min(state.num_selected / state.set_size, state.batch_size), 1);
    // set weights to 1.0
    fill_array(kmeans_state.weights, kmeans_state.num_seeds, (typename DetailedPlaceDBType::type)1.0);
}

template <typename T>
__inline__ __device__ T kmeans_distance(T node_x, T node_y, T center_x, T center_y) {
    T distance = fabs(node_x - center_x) + fabs(node_y - center_y);
    return distance;
}

template <typename T>
struct ItemWithIndex {
    T value;
    int index;
};

template <typename T>
struct ReduceMinOP {
    __host__ __device__ ItemWithIndex<T> operator()(const ItemWithIndex<T>& a, const ItemWithIndex<T>& b) const {
        return (a.value < b.value) ? a : b;
    }
};

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType, int ThreadsPerBlock = 128>
__global__ void kmeans_find_centers_kernel(DetailedPlaceDBType db,
                                           IndependentSetMatchingStateType state,
                                           KMeansState<typename DetailedPlaceDBType::type> kmeans_state) {
    assert(blockIdx.x < state.num_selected);
    int node_id = state.selected_maximal_independent_set[blockIdx.x];
    assert(node_id < db.num_movable_nodes);
    auto node_x = db.x[node_id];
    auto node_y = db.y[node_id];

    typedef cub::BlockReduce<ItemWithIndex<typename DetailedPlaceDBType::type>, ThreadsPerBlock> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    ItemWithIndex<typename DetailedPlaceDBType::type> thread_data;

    thread_data.value = cuda::numeric_limits<typename DetailedPlaceDBType::type>::max();
    thread_data.index = cuda::numeric_limits<int>::max();
    for (int center_id = threadIdx.x; center_id < kmeans_state.num_seeds; center_id += ThreadsPerBlock) {
        assert(center_id < kmeans_state.num_seeds);
        // scale back to floating point numbers
        typename DetailedPlaceDBType::type center_x =
            kmeans_state.centers_x[center_id] / KMeansState<typename DetailedPlaceDBType::type>::scale;
        typename DetailedPlaceDBType::type center_y =
            kmeans_state.centers_y[center_id] / KMeansState<typename DetailedPlaceDBType::type>::scale;
        typename DetailedPlaceDBType::type weight = kmeans_state.weights[center_id];

        typename DetailedPlaceDBType::type distance = kmeans_distance(node_x, node_y, center_x, center_y) * weight;
        if (distance < thread_data.value) {
            thread_data.value = distance;
            thread_data.index = center_id;
        }
    }
    if (threadIdx.x < kmeans_state.num_seeds) {
        assert(thread_data.index < kmeans_state.num_seeds);
    }

    __syncthreads();

    // Compute the block-wide max for thread0
    ItemWithIndex<typename DetailedPlaceDBType::type> aggregate =
        BlockReduce(temp_storage)
            .Reduce(thread_data, ReduceMinOP<typename DetailedPlaceDBType::type>(), kmeans_state.num_seeds);

    __syncthreads();

    if (threadIdx.x == 0) {
        assert(blockIdx.x < state.num_selected);
        kmeans_state.node2centers_map[blockIdx.x] = aggregate.index;
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void init_kmeans_seeds_kernel(DetailedPlaceDBType db,
                                         IndependentSetMatchingStateType state,
                                         KMeansState<typename DetailedPlaceDBType::type> kmeans_state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kmeans_state.num_seeds) {
        assert(db.num_movable_nodes - i - 1 < db.num_movable_nodes && db.num_movable_nodes - i - 1 >= 0);
        int random_number = state.ordered_nodes[db.num_movable_nodes - i - 1];
        random_number = random_number % state.num_selected;
        int node_id = state.selected_maximal_independent_set[random_number];
        assert(node_id < db.num_movable_nodes);
        // scale up for fixed point numbers
        kmeans_state.centers_x[i] = db.x[node_id] * KMeansState<typename DetailedPlaceDBType::type>::scale;
        kmeans_state.centers_y[i] = db.y[node_id] * KMeansState<typename DetailedPlaceDBType::type>::scale;
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void init_kmeans_seeds(const DetailedPlaceDBType& db,
                       IndependentSetMatchingStateType& state,
                       KMeansState<typename DetailedPlaceDBType::type>& kmeans_state) {
    init_kmeans_seeds_kernel<<<ceilDiv(kmeans_state.num_seeds, 256), 256>>>(db, state, kmeans_state);
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void reset_kmeans_partition_sizes_kernel(DetailedPlaceDBType db,
                                                    IndependentSetMatchingStateType state,
                                                    KMeansState<typename DetailedPlaceDBType::type> kmeans_state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kmeans_state.num_seeds) {
        kmeans_state.partition_sizes[i] = 0;
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void compute_kmeans_partition_sizes_kernel(DetailedPlaceDBType db,
                                                      IndependentSetMatchingStateType state,
                                                      KMeansState<typename DetailedPlaceDBType::type> kmeans_state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < state.num_selected) {
        int center_id = kmeans_state.node2centers_map[i];
        assert(center_id < kmeans_state.num_seeds);
        atomicAdd(kmeans_state.partition_sizes + center_id, 1);
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void reset_kmeans_centers_kernel(DetailedPlaceDBType db,
                                            IndependentSetMatchingStateType state,
                                            KMeansState<typename DetailedPlaceDBType::type> kmeans_state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kmeans_state.num_seeds) {
        if (kmeans_state.partition_sizes[i]) {
            kmeans_state.centers_x[i] = 0;
            kmeans_state.centers_y[i] = 0;
        }
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void compute_kmeans_centers_sum_kernel(DetailedPlaceDBType db,
                                                  IndependentSetMatchingStateType state,
                                                  KMeansState<typename DetailedPlaceDBType::type> kmeans_state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < state.num_selected) {
        int node_id = state.selected_maximal_independent_set[i];
        int center_id = kmeans_state.node2centers_map[i];
        assert(center_id < kmeans_state.num_seeds);
        assert(node_id < db.num_movable_nodes);
        // scale up for fixed point numbers
        atomicAddWrapper<typename KMeansState<typename DetailedPlaceDBType::type>::coordinate_type>(
            kmeans_state.centers_x + center_id, db.x[node_id] * KMeansState<typename DetailedPlaceDBType::type>::scale);
        atomicAddWrapper<typename KMeansState<typename DetailedPlaceDBType::type>::coordinate_type>(
            kmeans_state.centers_y + center_id, db.y[node_id] * KMeansState<typename DetailedPlaceDBType::type>::scale);
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void compute_kmeans_centers_div_kernel(DetailedPlaceDBType db,
                                                  IndependentSetMatchingStateType state,
                                                  KMeansState<typename DetailedPlaceDBType::type> kmeans_state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kmeans_state.num_seeds) {
        int s = kmeans_state.partition_sizes[i];
        if (s) {
            kmeans_state.centers_x[i] /= s;
            kmeans_state.centers_y[i] /= s;
        }
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void kmeans_update_centers(const DetailedPlaceDBType& db,
                           IndependentSetMatchingStateType& state,
                           KMeansState<typename DetailedPlaceDBType::type>& kmeans_state) {
    // reset partition_sizes to 0
    reset_kmeans_partition_sizes_kernel<<<ceilDiv(kmeans_state.num_seeds, 256), 256>>>(db, state, kmeans_state);
    // compute partition sizes
    compute_kmeans_partition_sizes_kernel<<<ceilDiv(state.num_selected, 256), 256>>>(db, state, kmeans_state);
    // reset kmeans centers to 0
    reset_kmeans_centers_kernel<<<ceilDiv(kmeans_state.num_seeds, 256), 256>>>(db, state, kmeans_state);
    // compute kmeans centers sum
    compute_kmeans_centers_sum_kernel<<<ceilDiv(state.num_selected, 256), 256>>>(db, state, kmeans_state);
    // compute kmeans centers div
    compute_kmeans_centers_div_kernel<<<ceilDiv(kmeans_state.num_seeds, 256), 256>>>(db, state, kmeans_state);
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void compute_kmeans_weights_kernel(DetailedPlaceDBType db,
                                              IndependentSetMatchingStateType state,
                                              KMeansState<typename DetailedPlaceDBType::type> kmeans_state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kmeans_state.num_seeds) {
        int s = kmeans_state.partition_sizes[i];
        auto& w = kmeans_state.weights[i];
        if (s > state.set_size) {
            auto ratio = s / (typename DetailedPlaceDBType::type)state.set_size;
            ratio = 1.0 + 0.5 * log(ratio);
            w *= ratio;
        }
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void kmeans_update_weights(const DetailedPlaceDBType& db,
                           IndependentSetMatchingStateType& state,
                           KMeansState<typename DetailedPlaceDBType::type>& kmeans_state) {
    compute_kmeans_weights_kernel<<<ceilDiv(kmeans_state.num_seeds, 256), 256>>>(db, state, kmeans_state);
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void kmeans_collect_sets_cuda2cpu(const DetailedPlaceDBType& db,
                                  IndependentSetMatchingStateType& state,
                                  KMeansState<typename DetailedPlaceDBType::type>& kmeans_state) {
    std::vector<int> selected_nodes(state.num_selected);
    checkCuda(cudaMemcpy(selected_nodes.data(),
                         state.selected_maximal_independent_set,
                         sizeof(int) * state.num_selected,
                         cudaMemcpyDeviceToHost));
    std::vector<int> node2centers_map(state.num_selected);
    checkCuda(cudaMemcpy(node2centers_map.data(),
                         kmeans_state.node2centers_map,
                         sizeof(int) * state.num_selected,
                         cudaMemcpyDeviceToHost));

    std::vector<int> flat_independent_sets(state.batch_size * state.set_size, std::numeric_limits<int>::max());
    std::vector<int> independent_set_sizes(state.batch_size, 0);
    // directly use flat array
    for (int i = 0; i < state.num_selected; ++i) {
        int node_id = selected_nodes.at(i);
        int center_id = node2centers_map.at(i);
        int& size = independent_set_sizes.at(center_id);
        if (size < state.set_size) {
            flat_independent_sets.at(center_id * state.set_size + size) = node_id;
            ++size;
        }
    }
    checkCuda(cudaMemcpy(state.independent_sets,
                         flat_independent_sets.data(),
                         sizeof(int) * state.batch_size * state.set_size,
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(state.independent_set_sizes,
                         independent_set_sizes.data(),
                         sizeof(int) * state.batch_size,
                         cudaMemcpyHostToDevice));

    // statistics
    logger.debug(
        "from %d nodes, collect %d sets, avg %d nodes, min/max %d/%d nodes",
        state.num_selected,
        state.num_independent_sets,
        std::accumulate(independent_set_sizes.begin(), independent_set_sizes.begin() + state.num_independent_sets, 0) /
            state.num_independent_sets,
        *std::min_element(independent_set_sizes.begin(), independent_set_sizes.begin() + state.num_independent_sets),
        *std::max_element(independent_set_sizes.begin(), independent_set_sizes.begin() + state.num_independent_sets));
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void partition_kmeans(const DetailedPlaceDBType& db,
                      IndependentSetMatchingStateType& state,
                      KMeansState<typename DetailedPlaceDBType::type>& kmeans_state) {
    prepare_kmeans(db, state, kmeans_state);
    init_kmeans_seeds(db, state, kmeans_state);

    for (int iter = 0; iter < 2; ++iter) {
        // for each node, find centers
        kmeans_find_centers_kernel<DetailedPlaceDBType, IndependentSetMatchingStateType, 256>
            <<<state.num_selected, 256>>>(db, state, kmeans_state);
        // for each center, adjust itself
        kmeans_update_centers(db, state, kmeans_state);
        // for each partition, update weight
        kmeans_update_weights(db, state, kmeans_state);
    }

    state.num_independent_sets = kmeans_state.num_seeds;
    kmeans_collect_sets_cuda2cpu(db, state, kmeans_state);
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void collect_independent_sets(const DetailedPlaceDBType& db,
                              IndependentSetMatchingStateType& state,
                              KMeansState<typename DetailedPlaceDBType::type>& kmeans_state,
                              DetailedPlaceCPUDB<typename DetailedPlaceDBType::type>& host_db,
                              IndependentSetMatchingCPUState<typename DetailedPlaceDBType::type>& host_state) {
    partition_kmeans(db, state, kmeans_state);
}

}  // namespace dp