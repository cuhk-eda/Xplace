#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <limits.h>

#include "cub/cub.cuh"

#define checkCuda(expression)                                                                                \
    {                                                                                                        \
        cudaError_t status = (expression);                                                                   \
        if (status != cudaSuccess) {                                                                         \
            printf("CUDA Runtime Error: %s at %s:%d\n", cudaGetErrorString(expression), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                                                                         \
        }                                                                                                    \
    }

#define checkCurand(expression)                                    \
    {                                                              \
        curandStatus_t status = (expression);                      \
        if (status != CURAND_STATUS_SUCCESS) {                     \
            printf("Curand Error at %s:%d\n", __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    }

#define allocateCuda(var, size, type)                                                \
    {                                                                                \
        cudaError_t status = cudaMalloc(&(var), (size) * sizeof(type));              \
        if (status != cudaSuccess) {                                                 \
            printf("cudaMalloc failed for " #var " at %s:%d\n", __FILE__, __LINE__); \
        }                                                                            \
    }

#define allocateCopyCuda(var, rhs, size)                                                          \
    {                                                                                             \
        allocateCuda(var, size, decltype(*rhs));                                                  \
        checkCuda(cudaMemcpy(var, rhs, sizeof(decltype(*rhs)) * (size), cudaMemcpyHostToDevice)); \
    }

#define allocateCopyCpu(var, rhs, size, T)                                                         \
    {                                                                                              \
        var = (T*)malloc(sizeof(T) * (size));                                                      \
        checkCuda(cudaMemcpy((void*)var, (void*)rhs, sizeof(T) * (size), cudaMemcpyDeviceToHost)); \
    }


// For cuda::numeric_limits
namespace cuda {  // namespace cuda

template <typename T>
struct numeric_limits_base {
    typedef T type;
};
template <typename T>
struct numeric_limits : public numeric_limits_base<T> {};

template <>
struct numeric_limits<char> : public numeric_limits_base<char> {
    /** The minimum finite value, or for floating types with
      denormalization, the minimum positive normalized value.  */
    __host__ __device__ static constexpr type min() noexcept { return CHAR_MIN; }

    /** The maximum finite value.  */
    __host__ __device__ static constexpr type max() noexcept { return CHAR_MAX; }

    /** A finite value x such that there is no other finite value y
     *  where y < x.  */
    __host__ __device__ static constexpr type lowest() noexcept { return CHAR_MIN; }

    /** A the machine epsilon.  */
    __host__ __device__ static constexpr type epsilon() noexcept { return 0; }
};

template <>
struct numeric_limits<unsigned char> : public numeric_limits_base<unsigned char> {
    /** The minimum finite value, or for floating types with
      denormalization, the minimum positive normalized value.  */
    __host__ __device__ static constexpr type min() noexcept { return 0; }

    /** The maximum finite value.  */
    __host__ __device__ static constexpr type max() noexcept { return UCHAR_MAX; }

    /** A finite value x such that there is no other finite value y
     *  where y < x.  */
    __host__ __device__ static constexpr type lowest() noexcept { return 0; }

    /** A the machine epsilon.  */
    __host__ __device__ static constexpr type epsilon() noexcept { return 0; }
};

template <>
struct numeric_limits<short> : public numeric_limits_base<short> {
    /** The minimum finite value, or for floating types with
      denormalization, the minimum positive normalized value.  */
    __host__ __device__ static constexpr type min() noexcept { return SHRT_MIN; }

    /** The maximum finite value.  */
    __host__ __device__ static constexpr type max() noexcept { return SHRT_MAX; }

    /** A finite value x such that there is no other finite value y
     *  where y < x.  */
    __host__ __device__ static constexpr type lowest() noexcept { return SHRT_MIN; }

    /** A the machine epsilon.  */
    __host__ __device__ static constexpr type epsilon() noexcept { return 0; }
};

template <>
struct numeric_limits<unsigned short> : public numeric_limits_base<unsigned short> {
    /** The minimum finite value, or for floating types with
      denormalization, the minimum positive normalized value.  */
    __host__ __device__ static constexpr type min() noexcept { return 0; }

    /** The maximum finite value.  */
    __host__ __device__ static constexpr type max() noexcept { return USHRT_MAX; }

    /** A finite value x such that there is no other finite value y
     *  where y < x.  */
    __host__ __device__ static constexpr type lowest() noexcept { return 0; }

    /** A the machine epsilon.  */
    __host__ __device__ static constexpr type epsilon() noexcept { return 0; }
};

template <>
struct numeric_limits<int> : public numeric_limits_base<int> {
    /** The minimum finite value, or for floating types with
      denormalization, the minimum positive normalized value.  */
    __host__ __device__ static constexpr type min() noexcept { return INT_MIN; }

    /** The maximum finite value.  */
    __host__ __device__ static constexpr type max() noexcept { return INT_MAX; }

    /** A finite value x such that there is no other finite value y
     *  where y < x.  */
    __host__ __device__ static constexpr type lowest() noexcept { return INT_MIN; }

    /** A the machine epsilon.  */
    __host__ __device__ static constexpr type epsilon() noexcept { return 0; }
};

template <>
struct numeric_limits<unsigned int> : public numeric_limits_base<unsigned int> {
    /** The minimum finite value, or for floating types with
      denormalization, the minimum positive normalized value.  */
    __host__ __device__ static constexpr type min() noexcept { return 0; }

    /** The maximum finite value.  */
    __host__ __device__ static constexpr type max() noexcept { return UINT_MAX; }

    /** A finite value x such that there is no other finite value y
     *  where y < x.  */
    __host__ __device__ static constexpr type lowest() noexcept { return 0; }

    /** A the machine epsilon.  */
    __host__ __device__ static constexpr type epsilon() noexcept { return 0; }
};

template <>
struct numeric_limits<long> : public numeric_limits_base<long> {
    /** The minimum finite value, or for floating types with
      denormalization, the minimum positive normalized value.  */
    __host__ __device__ static constexpr type min() noexcept { return LONG_MIN; }

    /** The maximum finite value.  */
    __host__ __device__ static constexpr type max() noexcept { return LONG_MAX; }

    /** A finite value x such that there is no other finite value y
     *  where y < x.  */
    __host__ __device__ static constexpr type lowest() noexcept { return LONG_MIN; }

    /** A the machine epsilon.  */
    __host__ __device__ static constexpr type epsilon() noexcept { return 0; }
};

template <>
struct numeric_limits<unsigned long> : public numeric_limits_base<unsigned long> {
    /** The minimum finite value, or for floating types with
      denormalization, the minimum positive normalized value.  */
    __host__ __device__ static constexpr type min() noexcept { return 0; }

    /** The maximum finite value.  */
    __host__ __device__ static constexpr type max() noexcept { return ULONG_MAX; }

    /** A finite value x such that there is no other finite value y
     *  where y < x.  */
    __host__ __device__ static constexpr type lowest() noexcept { return 0; }

    /** A the machine epsilon.  */
    __host__ __device__ static constexpr type epsilon() noexcept { return 0; }
};

template <>
struct numeric_limits<long long> : public numeric_limits_base<long long> {
    /** The minimum finite value, or for floating types with
      denormalization, the minimum positive normalized value.  */
    __host__ __device__ static constexpr type min() noexcept { return LLONG_MIN; }

    /** The maximum finite value.  */
    __host__ __device__ static constexpr type max() noexcept { return LLONG_MAX; }

    /** A finite value x such that there is no other finite value y
     *  where y < x.  */
    __host__ __device__ static constexpr type lowest() noexcept { return LLONG_MIN; }

    /** A the machine epsilon.  */
    __host__ __device__ static constexpr type epsilon() noexcept { return 0; }
};

template <>
struct numeric_limits<unsigned long long> : public numeric_limits_base<unsigned long long> {
    /** The minimum finite value, or for floating types with
      denormalization, the minimum positive normalized value.  */
    __host__ __device__ static constexpr type min() noexcept { return 0; }

    /** The maximum finite value.  */
    __host__ __device__ static constexpr type max() noexcept { return ULLONG_MAX; }

    /** A finite value x such that there is no other finite value y
     *  where y < x.  */
    __host__ __device__ static constexpr type lowest() noexcept { return 0; }

    /** A the machine epsilon.  */
    __host__ __device__ static constexpr type epsilon() noexcept { return 0; }
};

template <>
struct numeric_limits<float> : public numeric_limits_base<float> {
    /** The minimum finite value, or for floating types with
      denormalization, the minimum positive normalized value.  */
    __host__ __device__ static constexpr type min() noexcept { return FLT_MIN; }

    /** The maximum finite value.  */
    __host__ __device__ static constexpr type max() noexcept { return FLT_MAX; }

    /** A finite value x such that there is no other finite value y
     *  where y < x.  */
    __host__ __device__ static constexpr type lowest() noexcept { return -FLT_MAX; }

    /** A the machine epsilon.  */
    __host__ __device__ static constexpr type epsilon() noexcept { return FLT_EPSILON; }
};

template <>
struct numeric_limits<double> : public numeric_limits_base<double> {
    /** The minimum finite value, or for floating types with
      denormalization, the minimum positive normalized value.  */
    __host__ __device__ static constexpr type min() noexcept { return DBL_MIN; }

    /** The maximum finite value.  */
    __host__ __device__ static constexpr type max() noexcept { return DBL_MAX; }

    /** A finite value x such that there is no other finite value y
     *  where y < x.  */
    __host__ __device__ static constexpr type lowest() noexcept { return -DBL_MAX; }

    /** A the machine epsilon.  */
    __host__ __device__ static constexpr type epsilon() noexcept { return DBL_EPSILON; }
};

template <>
struct numeric_limits<long double> : public numeric_limits_base<long double> {
    /** The minimum finite value, or for floating types with
      denormalization, the minimum positive normalized value.  */
    __host__ __device__ static constexpr type min() noexcept { return LDBL_MIN; }

    /** The maximum finite value.  */
    __host__ __device__ static constexpr type max() noexcept { return LDBL_MAX; }

    /** A finite value x such that there is no other finite value y
     *  where y < x.  */
    __host__ __device__ static constexpr type lowest() noexcept { return -LDBL_MAX; }

    /** A the machine epsilon.  */
    __host__ __device__ static constexpr type epsilon() noexcept { return LDBL_EPSILON; }
};
}  // namespace cuda
