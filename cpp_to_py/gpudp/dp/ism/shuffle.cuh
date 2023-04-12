#pragma once

#include <curand.h>
#include <curand_kernel.h>

#include "gpudp/dp/utils.cuh"

namespace dp {

template <typename T, typename V>
__global__ void print_shuffle(const T* values, const V* keys, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("values[%d]\n", n);
        for (int i = 0; i < n; ++i) {
            printf("%d ", int(values[i]));
        }
        printf("\n");
        printf("keys[%d]\n", n);
        for (int i = 0; i < n; ++i) {
            printf("%d ", int(keys[i]));
        }
        printf("\n");
    }
}

/// @brief A shuffler that can be repeatedly called.
/// @tparam T value type
/// @tparam V key type
template <typename T, typename V>
class Shuffler {
public:
    /// @brief constructor
    /// @param seed random seed
    /// @param values data array that will be manipulated
    /// @param n length of array
    Shuffler(size_t seed, T* values, int n) {
        /* Create pseudo-random number generator */
        checkCurand(curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_DEFAULT));

        /* Set seed */
        checkCurand(curandSetPseudoRandomGeneratorSeed(m_gen, seed));

        m_values_in = values;
        allocateCuda(m_keys_in, n, V);
        allocateCuda(m_keys_out, n, V);
        allocateCuda(m_values_out, n, T);
        m_temp_storage = NULL;
        m_temp_storage_bytes = 0;
        m_num_items = n;
    }
    /// @brief destructor
    ~Shuffler() {
        if (m_temp_storage) {
            cudaFree(m_temp_storage);
        }
        cudaFree(m_keys_in);
        cudaFree(m_keys_out);
        cudaFree(m_values_out);
        checkCurand(curandDestroyGenerator(m_gen));
    }
    /// @brief top API to shuffle data. It can be called repeatedly.
    void operator()() {
        /* Generate n floats on device */
        checkCurand(curandGenerate(m_gen, m_keys_in, m_num_items));

        // Determine temporary device storage requirements
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes, m_keys_in, m_keys_out, m_values_in, m_values_out, m_num_items);

        // Allocate temporary storage
        // re-allocate if different size
        if (m_temp_storage_bytes != temp_storage_bytes) {
            if (m_temp_storage_bytes) {
                cudaFree(m_temp_storage);
                m_temp_storage = NULL;
            }
            m_temp_storage_bytes = temp_storage_bytes;
            logger.debug("allocate %lu bytes in shuffler for length %d*(%d+%d)",
                         m_temp_storage_bytes,
                         m_num_items,
                         sizeof(T),
                         sizeof(V));
            checkCuda(cudaMalloc(&m_temp_storage, m_temp_storage_bytes));
        }

        // Run sorting operation
        cub::DeviceRadixSort::SortPairs(
            m_temp_storage, m_temp_storage_bytes, m_keys_in, m_keys_out, m_values_in, m_values_out, m_num_items);

        // copy back to m_values_in, not necessary
        // As m_values_in corresponds to external data, copying back the output can allow in-place manipulation
        checkCuda(cudaMemcpy(m_values_in, m_values_out, sizeof(T) * m_num_items, cudaMemcpyDeviceToDevice));
    }

protected:
    curandGenerator_t m_gen;      ///< random number generator
    V* m_keys_in;                 ///< on device, to store real key
    T* m_values_in;               ///< on device, to store real data
    V* m_keys_out;                ///< on device, a buffer
    T* m_values_out;              ///< on device, a buffer
    void* m_temp_storage;         ///< on device, temporary storage for sorting
    size_t m_temp_storage_bytes;  ///< number of bytes for m_temp_storage
    int m_num_items;              ///< length of array
};

}  // namespace dp