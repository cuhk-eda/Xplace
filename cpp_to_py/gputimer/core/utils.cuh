#pragma once

#include "gputimer/base.h"
namespace gt {

template <typename T>
__global__ void debugPrint(T *arr, int size) {
    for (int i = 0; i < size; i++) {
        if constexpr (std::is_same_v<T, int>) {
            printf("%d %d\n", i, arr[i]);
        } else if constexpr (std::is_same_v<T, float>) {
            printf("%d %f\n", i, arr[i]);
        } else if constexpr (std::is_same_v<T, index_type>) {
            printf("%d %d\n", i, arr[i]);
        }
    }
    printf("\n");
}
template <typename T>
__global__ void debugPrint1(T *arr, int size, int m) {
    for (int i = 0; i < size; i++) {
        printf("%d", i);
        for (int j = 0; j < m; j++) {
            if constexpr (std::is_same_v<T, int>) {
                printf(" %d ", arr[m * i + j]);
            } else if constexpr (std::is_same_v<T, float>) {
                printf(" %f ", arr[m * i + j]);
            } else if constexpr (std::is_same_v<T, index_type>) {
                printf(" %d ", arr[m * i + j]);
            }
        }
        printf("\n");
    }
    printf("\n");
}
template <typename T>
__global__ void debugPrintIdx(int idx, T *arr) {
    printf("idx: %d, value: %d\n", idx, arr[idx]);
}

template <typename T>
__global__ void reset(float *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = nanf("");
    }
}

template <typename T>
__global__ void reset_batch(float *array, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) array[index] = nanf("");
}

template <typename T>
__global__ void reset_val(T *array, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        if constexpr (std::is_same_v<T, int>) {
            array[index] = -1;
        }
        if constexpr (std::is_same_v<T, float>) {
            array[index] = nanf("");
        }
    }
}

template <typename T>
__global__ void device_copy(T *src, T *dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = src[i];
    }
}

template <typename T>
__global__ void device_copy_batch(T *src, T *dst, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) dst[index] = src[index];
}


}  // namespace gt