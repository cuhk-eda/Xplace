/**
 * @file   dct2_fft2_cuda_kernel.cu
 * @author Zixuan Jiang, Jiaqi Gu
 * @date   Apr 2019
 * @brief  Refernece: Byeong Lee, "A new algorithm to compute the discrete cosine Transform,"
 *      in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 32, no. 6, pp. 1243-1245, December 1984.
 *      The preprocess and postprocess of 2d dct and 2d idct are discussed in the original paper.
 *      idct(idxst(x)) and idxst(idct(x)) are similar to the idct2d(x),
 *      except tiny modifications on preprocessing and postprocessing
 */

#include <ATen/cuda/CUDAContext.h>
#include <float.h>
#include <math.h>
#include <torch/extension.h>

#include "cuda_runtime.h"

// ComplexType
template <typename T>
struct ComplexType {
    T x;
    T y;
    __host__ __device__ ComplexType() {
        x = 0;
        y = 0;
    }

    __host__ __device__ ComplexType(T real, T imag) {
        x = real;
        y = imag;
    }

    __host__ __device__ ~ComplexType() {}
};

template <typename T>
inline __host__ __device__ ComplexType<T> complexMul(const ComplexType<T> &x, const ComplexType<T> &y) {
    ComplexType<T> res;
    res.x = x.x * y.x - x.y * y.y;
    res.y = x.x * y.y + x.y * y.x;
    return res;
}

template <typename T>
inline __host__ __device__ T RealPartOfMul(const ComplexType<T> &x, const ComplexType<T> &y) {
    return x.x * y.x - x.y * y.y;
}

template <typename T>
inline __host__ __device__ T ImaginaryPartOfMul(const ComplexType<T> &x, const ComplexType<T> &y) {
    return x.x * y.y + x.y * y.x;
}

template <typename T>
inline __host__ __device__ ComplexType<T> complexAdd(const ComplexType<T> &x, const ComplexType<T> &y) {
    ComplexType<T> res;
    res.x = x.x + y.x;
    res.y = x.y + y.y;
    return res;
}

template <typename T>
inline __host__ __device__ ComplexType<T> complexSubtract(const ComplexType<T> &x, const ComplexType<T> &y) {
    ComplexType<T> res;
    res.x = x.x - y.x;
    res.y = x.y - y.y;
    return res;
}

template <typename T>
inline __host__ __device__ ComplexType<T> complexConj(const ComplexType<T> &x) {
    ComplexType<T> res;
    res.x = x.x;
    res.y = -x.y;
    return res;
}

template <typename T>
inline __host__ __device__ ComplexType<T> complexMulConj(const ComplexType<T> &x, const ComplexType<T> &y) {
    ComplexType<T> res;
    res.x = x.x * y.x - x.y * y.y;
    res.y = -(x.x * y.y + x.y * y.x);
    return res;
}

#define TPB (16)

inline __device__ int INDEX(const int hid, const int wid, const int N) { return (hid * N + wid); }

// dct2_fft2
template <typename T>
__global__ void dct2dPreprocess(const T *x, T *y, const int M, const int N, const int halfN) {
    const int wid = blockDim.x * blockIdx.x + threadIdx.x;
    const int hid = blockDim.y * blockIdx.y + threadIdx.y;
    if (hid < M && wid < N) {
        int index;
        int cond = (((hid & 1) == 0) << 1) | ((wid & 1) == 0);
        switch (cond) {
            case 0:
                index = INDEX(2 * M - (hid + 1), N - (wid + 1) / 2, halfN);
                break;
            case 1:
                index = INDEX(2 * M - (hid + 1), wid / 2, halfN);
                break;
            case 2:
                index = INDEX(hid, N - (wid + 1) / 2, halfN);
                break;
            case 3:
                index = INDEX(hid, wid / 2, halfN);
                break;
            default:
                break;
        }
        y[index] = x[INDEX(hid, wid, N)];
    }
}

template <typename T>
void dct2dPreprocessCudaLauncher(const T *x, T *y, const int M, const int N) {
    dim3 gridSize((N + TPB - 1) / TPB, (M + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    auto stream = at::cuda::getCurrentCUDAStream();
    dct2dPreprocess<T><<<gridSize, blockSize, 0, stream>>>(x, y, M, N, N / 2);
}

template <typename T, typename TComplex>
__global__ void __launch_bounds__(TPB *TPB, 4) dct2dPostprocess(const TComplex *V,
                                                                T *y,
                                                                const int M,
                                                                const int N,
                                                                const int halfM,
                                                                const int halfN,
                                                                const T two_over_MN,
                                                                const T four_over_MN,
                                                                const TComplex *__restrict__ expkM,
                                                                const TComplex *__restrict__ expkN) {
    const int wid = blockDim.x * blockIdx.x + threadIdx.x;
    const int hid = blockDim.y * blockIdx.y + threadIdx.y;
    if (hid < halfM && wid < halfN) {
        int cond = ((hid != 0) << 1) | (wid != 0);
        switch (cond) {
            case 0: {
                y[0] = V[0].x * four_over_MN;
                y[halfN] = RealPartOfMul(expkN[halfN], V[halfN]) * four_over_MN;
                y[INDEX(halfM, 0, N)] = expkM[halfM].x * V[INDEX(halfM, 0, halfN + 1)].x * four_over_MN;
                y[INDEX(halfM, halfN, N)] =
                    expkM[halfM].x * RealPartOfMul(expkN[halfN], V[INDEX(halfM, halfN, halfN + 1)]) * four_over_MN;
                break;
            }

            case 1: {
                ComplexType<T> tmp;

                tmp = V[wid];
                y[wid] = RealPartOfMul(expkN[wid], tmp) * four_over_MN;
                y[N - wid] = -ImaginaryPartOfMul(expkN[wid], tmp) * four_over_MN;

                tmp = V[INDEX(halfM, wid, halfN + 1)];
                y[INDEX(halfM, wid, N)] = expkM[halfM].x * RealPartOfMul(expkN[wid], tmp) * four_over_MN;
                y[INDEX(halfM, N - wid, N)] = -expkM[halfM].x * ImaginaryPartOfMul(expkN[wid], tmp) * four_over_MN;
                break;
            }

            case 2: {
                ComplexType<T> tmp1, tmp2, tmp_up, tmp_down;
                tmp1 = V[INDEX(hid, 0, halfN + 1)];
                tmp2 = V[INDEX(M - hid, 0, halfN + 1)];
                tmp_up.x = expkM[hid].x * (tmp1.x + tmp2.x) + expkM[hid].y * (tmp2.y - tmp1.y);
                tmp_down.x = -expkM[hid].y * (tmp1.x + tmp2.x) + expkM[hid].x * (tmp2.y - tmp1.y);
                y[INDEX(hid, 0, N)] = tmp_up.x * two_over_MN;
                y[INDEX(M - hid, 0, N)] = tmp_down.x * two_over_MN;

                tmp1 = complexAdd(V[INDEX(hid, halfN, halfN + 1)], V[INDEX(M - hid, halfN, halfN + 1)]);
                tmp2 = complexSubtract(V[INDEX(hid, halfN, halfN + 1)], V[INDEX(M - hid, halfN, halfN + 1)]);
                tmp_up.x = expkM[hid].x * tmp1.x - expkM[hid].y * tmp2.y;
                tmp_up.y = expkM[hid].x * tmp1.y + expkM[hid].y * tmp2.x;
                tmp_down.x = -expkM[hid].y * tmp1.x - expkM[hid].x * tmp2.y;
                tmp_down.y = -expkM[hid].y * tmp1.y + expkM[hid].x * tmp2.x;
                y[INDEX(hid, halfN, N)] = RealPartOfMul(expkN[halfN], tmp_up) * two_over_MN;
                y[INDEX(M - hid, halfN, N)] = RealPartOfMul(expkN[halfN], tmp_down) * two_over_MN;
                break;
            }

            case 3: {
                ComplexType<T> tmp1, tmp2, tmp_up, tmp_down;
                tmp1 = complexAdd(V[INDEX(hid, wid, halfN + 1)], V[INDEX(M - hid, wid, halfN + 1)]);
                tmp2 = complexSubtract(V[INDEX(hid, wid, halfN + 1)], V[INDEX(M - hid, wid, halfN + 1)]);
                tmp_up.x = expkM[hid].x * tmp1.x - expkM[hid].y * tmp2.y;
                tmp_up.y = expkM[hid].x * tmp1.y + expkM[hid].y * tmp2.x;
                tmp_down.x = -expkM[hid].y * tmp1.x - expkM[hid].x * tmp2.y;
                tmp_down.y = -expkM[hid].y * tmp1.y + expkM[hid].x * tmp2.x;
                y[INDEX(hid, wid, N)] = RealPartOfMul(expkN[wid], tmp_up) * two_over_MN;
                y[INDEX(M - hid, wid, N)] = RealPartOfMul(expkN[wid], tmp_down) * two_over_MN;
                y[INDEX(hid, N - wid, N)] = -ImaginaryPartOfMul(expkN[wid], tmp_up) * two_over_MN;
                y[INDEX(M - hid, N - wid, N)] = -ImaginaryPartOfMul(expkN[wid], tmp_down) * two_over_MN;
                break;
            }

            default:
                break;
        }
    }
}

template <typename T>
void dct2dPostprocessCudaLauncher(
    const T *x, T *y, const int M, const int N, const T *__restrict__ expkM, const T *__restrict__ expkN) {
    dim3 gridSize((N / 2 + TPB - 1) / TPB, (M / 2 + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    auto stream = at::cuda::getCurrentCUDAStream();
    dct2dPostprocess<T, ComplexType<T>><<<gridSize, blockSize, 0, stream>>>((ComplexType<T> *)x,
                                                                            y,
                                                                            M,
                                                                            N,
                                                                            M / 2,
                                                                            N / 2,
                                                                            (T)(2. / (M * N)),
                                                                            (T)(4. / (M * N)),
                                                                            (ComplexType<T> *)expkM,
                                                                            (ComplexType<T> *)expkN);
}

// idct2_fft2
template <typename T, typename TComplex>
__global__ void __launch_bounds__(TPB *TPB, 4) idct2_fft2Preprocess(const T *input,
                                                                    TComplex *output,
                                                                    const int M,
                                                                    const int N,
                                                                    const int halfM,
                                                                    const int halfN,
                                                                    const TComplex *__restrict__ expkM,
                                                                    const TComplex *__restrict__ expkN) {
    const int wid = blockDim.x * blockIdx.x + threadIdx.x;
    const int hid = blockDim.y * blockIdx.y + threadIdx.y;
    if (hid < halfM && wid < halfN) {
        int cond = ((hid != 0) << 1) | (wid != 0);
        switch (cond) {
            case 0: {
                T tmp1;
                TComplex tmp_up;

                output[0].x = input[0];
                output[0].y = 0;

                tmp1 = input[halfN];
                tmp_up.x = tmp1;
                tmp_up.y = tmp1;
                output[halfN] = complexConj(complexMul(expkN[halfN], tmp_up));

                tmp1 = input[INDEX(halfM, 0, N)];
                tmp_up.x = tmp1;
                tmp_up.y = tmp1;
                output[INDEX(halfM, 0, halfN + 1)] = complexConj(complexMul(expkM[halfM], tmp_up));

                tmp1 = input[INDEX(halfM, halfN, N)];
                tmp_up.x = 0;
                tmp_up.y = 2 * tmp1;
                output[INDEX(halfM, halfN, halfN + 1)] =
                    complexConj(complexMul(complexMul(expkM[halfM], expkN[halfN]), tmp_up));
                break;
            }

            case 1: {
                TComplex tmp_up;
                tmp_up.x = input[wid];
                tmp_up.y = input[N - wid];
                output[wid] = complexConj(complexMul(expkN[wid], tmp_up));

                T tmp1 = input[INDEX(halfM, wid, N)];
                T tmp2 = input[INDEX(halfM, N - wid, N)];
                tmp_up.x = tmp1 - tmp2;
                tmp_up.y = tmp1 + tmp2;
                output[INDEX(halfM, wid, halfN + 1)] =
                    complexConj(complexMul(complexMul(expkM[halfM], expkN[wid]), tmp_up));
                break;
            }

            case 2: {
                T tmp1, tmp3;
                TComplex tmp_up, tmp_down;

                tmp1 = input[INDEX(hid, 0, N)];
                tmp3 = input[INDEX(M - hid, 0, N)];
                tmp_up.x = tmp1;
                tmp_up.y = tmp3;
                tmp_down.x = tmp3;
                tmp_down.y = tmp1;

                output[INDEX(hid, 0, halfN + 1)] = complexConj(complexMul(expkM[hid], tmp_up));
                output[INDEX(M - hid, 0, halfN + 1)] = complexConj(complexMul(expkM[M - hid], tmp_down));

                tmp1 = input[INDEX(hid, halfN, N)];
                tmp3 = input[INDEX(M - hid, halfN, N)];
                tmp_up.x = tmp1 - tmp3;
                tmp_up.y = tmp3 + tmp1;
                tmp_down.x = tmp3 - tmp1;
                tmp_down.y = tmp1 + tmp3;

                output[INDEX(hid, halfN, halfN + 1)] =
                    complexConj(complexMul(complexMul(expkM[hid], expkN[halfN]), tmp_up));
                output[INDEX(M - hid, halfN, halfN + 1)] =
                    complexConj(complexMul(complexMul(expkM[M - hid], expkN[halfN]), tmp_down));
                break;
            }

            case 3: {
                T tmp1 = input[INDEX(hid, wid, N)];
                T tmp2 = input[INDEX(hid, N - wid, N)];
                T tmp3 = input[INDEX(M - hid, wid, N)];
                T tmp4 = input[INDEX(M - hid, N - wid, N)];
                TComplex tmp_up, tmp_down;
                tmp_up.x = tmp1 - tmp4;
                tmp_up.y = tmp3 + tmp2;
                tmp_down.x = tmp3 - tmp2;
                tmp_down.y = tmp1 + tmp4;

                output[INDEX(hid, wid, halfN + 1)] =
                    complexConj(complexMul(complexMul(expkM[hid], expkN[wid]), tmp_up));
                output[INDEX(M - hid, wid, halfN + 1)] =
                    complexConj(complexMul(complexMul(expkM[M - hid], expkN[wid]), tmp_down));
                break;
            }

            default:
                break;
        }
    }
}

template <typename T>
void idct2_fft2PreprocessCudaLauncher(
    const T *x, T *y, const int M, const int N, const T *__restrict__ expkM, const T *__restrict__ expkN) {
    dim3 gridSize((N / 2 + TPB - 1) / TPB, (M / 2 + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    auto stream = at::cuda::getCurrentCUDAStream();
    idct2_fft2Preprocess<T, ComplexType<T>><<<gridSize, blockSize, 0, stream>>>(
        x, (ComplexType<T> *)y, M, N, M / 2, N / 2, (ComplexType<T> *)expkM, (ComplexType<T> *)expkN);
}

template <typename T>
__global__ void idct2_fft2Postprocess(const T *x, T *y, const int M, const int N, const int halfN, const int MN) {
    const int wid = blockDim.x * blockIdx.x + threadIdx.x;
    const int hid = blockDim.y * blockIdx.y + threadIdx.y;
    if (hid < M && wid < N) {
        int cond = ((hid < M / 2) << 1) | (wid < N / 2);
        int index;
        switch (cond) {
            case 0:
                index = INDEX(((M - hid) << 1) - 1, ((N - wid) << 1) - 1, N);
                break;
            case 1:
                index = INDEX(((M - hid) << 1) - 1, wid << 1, N);
                break;
            case 2:
                index = INDEX(hid << 1, ((N - wid) << 1) - 1, N);
                break;
            case 3:
                index = INDEX(hid << 1, wid << 1, N);
                break;
            default:
                break;
        }
        y[index] = x[INDEX(hid, wid, N)] * MN;
    }
}

template <typename T>
void idct2_fft2PostprocessCudaLauncher(const T *x, T *y, const int M, const int N) {
    dim3 gridSize((N + TPB - 1) / TPB, (M + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    auto stream = at::cuda::getCurrentCUDAStream();
    idct2_fft2Postprocess<T><<<gridSize, blockSize, 0, stream>>>(x, y, M, N, N / 2, M * N);
}

// idct_idxst
// Adpated from idct2d_preprocess(). The only change is the reordered input
// if (wid != 0)
//     new_input[hid][wid] = input[hid][N - wid];
// else
//     new_input[hid][0] = 0
template <typename T, typename TComplex>
__global__ void __launch_bounds__(TPB *TPB, 4) idct_idxstPreprocess(const T *input,
                                                                    TComplex *output,
                                                                    const int M,
                                                                    const int N,
                                                                    const int halfM,
                                                                    const int halfN,
                                                                    const TComplex *__restrict__ expkM,
                                                                    const TComplex *__restrict__ expkN) {
    const int wid = blockDim.x * blockIdx.x + threadIdx.x;
    const int hid = blockDim.y * blockIdx.y + threadIdx.y;
    if (hid < halfM && wid < halfN) {
        int cond = ((hid != 0) << 1) | (wid != 0);
        switch (cond) {
            case 0: {
                T tmp1;
                TComplex tmp_up;

                output[0].x = 0;
                output[0].y = 0;

                tmp1 = input[halfN];
                tmp_up.x = tmp1;
                tmp_up.y = tmp1;
                output[halfN] = complexConj(complexMul(expkN[halfN], tmp_up));

                output[INDEX(halfM, 0, halfN + 1)].x = 0;
                output[INDEX(halfM, 0, halfN + 1)].y = 0;

                tmp1 = input[INDEX(halfM, halfN, N)];
                tmp_up.x = 0;
                tmp_up.y = 2 * tmp1;
                output[INDEX(halfM, halfN, halfN + 1)] =
                    complexConj(complexMul(complexMul(expkM[halfM], expkN[halfN]), tmp_up));
                break;
            }

            case 1: {
                TComplex tmp_up;
                tmp_up.x = input[N - wid];
                tmp_up.y = input[wid];
                output[wid] = complexConj(complexMul(expkN[wid], tmp_up));

                T tmp1 = input[INDEX(halfM, N - wid, N)];
                T tmp2 = input[INDEX(halfM, wid, N)];
                tmp_up.x = tmp1 - tmp2;
                tmp_up.y = tmp1 + tmp2;
                output[INDEX(halfM, wid, halfN + 1)] =
                    complexConj(complexMul(complexMul(expkM[halfM], expkN[wid]), tmp_up));
                break;
            }

            case 2: {
                T tmp1, tmp3;
                TComplex tmp_up, tmp_down;

                output[INDEX(hid, 0, halfN + 1)].x = 0;
                output[INDEX(hid, 0, halfN + 1)].y = 0;
                output[INDEX(M - hid, 0, halfN + 1)].x = 0;
                output[INDEX(M - hid, 0, halfN + 1)].y = 0;

                tmp1 = input[INDEX(hid, halfN, N)];
                tmp3 = input[INDEX(M - hid, halfN, N)];
                tmp_up.x = tmp1 - tmp3;
                tmp_up.y = tmp3 + tmp1;
                tmp_down.x = tmp3 - tmp1;
                tmp_down.y = tmp1 + tmp3;

                output[INDEX(hid, halfN, halfN + 1)] =
                    complexConj(complexMul(complexMul(expkM[hid], expkN[halfN]), tmp_up));
                output[INDEX(M - hid, halfN, halfN + 1)] =
                    complexConj(complexMul(complexMul(expkM[M - hid], expkN[halfN]), tmp_down));
                break;
            }

            case 3: {
                T tmp1 = input[INDEX(hid, N - wid, N)];
                T tmp2 = input[INDEX(hid, wid, N)];
                T tmp3 = input[INDEX(M - hid, N - wid, N)];
                T tmp4 = input[INDEX(M - hid, wid, N)];
                TComplex tmp_up, tmp_down;
                tmp_up.x = tmp1 - tmp4;
                tmp_up.y = tmp3 + tmp2;
                tmp_down.x = tmp3 - tmp2;
                tmp_down.y = tmp1 + tmp4;

                output[INDEX(hid, wid, halfN + 1)] =
                    complexConj(complexMul(complexMul(expkM[hid], expkN[wid]), tmp_up));
                output[INDEX(M - hid, wid, halfN + 1)] =
                    complexConj(complexMul(complexMul(expkM[M - hid], expkN[wid]), tmp_down));
                break;
            }

            default:
                break;
        }
    }
}

template <typename T>
void idct_idxstPreprocessCudaLauncher(
    const T *x, T *y, const int M, const int N, const T *__restrict__ expkM, const T *__restrict__ expkN) {
    dim3 gridSize((N / 2 + TPB - 1) / TPB, (M / 2 + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    auto stream = at::cuda::getCurrentCUDAStream();
    idct_idxstPreprocess<T, ComplexType<T>><<<gridSize, blockSize, 0, stream>>>(
        x, (ComplexType<T> *)y, M, N, M / 2, N / 2, (ComplexType<T> *)expkM, (ComplexType<T> *)expkN);
}

// Adpated from idct2d_postprocess() with changes on sign and scale
// if (wid % 2 == 1)
//     new_output[hid][wid] = -output[hid][wid];
// else
//     new_output[hid][wid] = output[hid][wid];
template <typename T>
__global__ void idct_idxstPostprocess(const T *x, T *y, const int M, const int N, const int halfN, const int MN) {
    const int wid = blockDim.x * blockIdx.x + threadIdx.x;
    const int hid = blockDim.y * blockIdx.y + threadIdx.y;
    if (hid < M && wid < N) {
        int cond = ((hid < M / 2) << 1) | (wid < N / 2);
        int index;
        switch (cond) {
            case 0:
                index = INDEX(((M - hid) << 1) - 1, ((N - wid) << 1) - 1, N);
                y[index] = -x[INDEX(hid, wid, N)] * MN;
                break;
            case 1:
                index = INDEX(((M - hid) << 1) - 1, wid << 1, N);
                y[index] = x[INDEX(hid, wid, N)] * MN;
                break;
            case 2:
                index = INDEX(hid << 1, ((N - wid) << 1) - 1, N);
                y[index] = -x[INDEX(hid, wid, N)] * MN;
                break;
            case 3:
                index = INDEX(hid << 1, wid << 1, N);
                y[index] = x[INDEX(hid, wid, N)] * MN;
                break;
            default:
                break;
        }
    }
}

template <typename T>
void idct_idxstPostprocessCudaLauncher(const T *x, T *y, const int M, const int N) {
    dim3 gridSize((N + TPB - 1) / TPB, (M + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    auto stream = at::cuda::getCurrentCUDAStream();
    idct_idxstPostprocess<T><<<gridSize, blockSize, 0, stream>>>(x, y, M, N, N / 2, M * N);
}

// idxst_idct
// Adpated from idct2d_preprocess(). The only change is the reordered input
// if (hid != 0)
//     new_input[hid][wid] = input[M - hid][wid];
// else
//     new_input[0][wid] = 0
template <typename T, typename TComplex>
__global__ void __launch_bounds__(TPB *TPB, 4) idxst_idctPreprocess(const T *input,
                                                                    TComplex *output,
                                                                    const int M,
                                                                    const int N,
                                                                    const int halfM,
                                                                    const int halfN,
                                                                    const TComplex *__restrict__ expkM,
                                                                    const TComplex *__restrict__ expkN) {
    const int wid = blockDim.x * blockIdx.x + threadIdx.x;
    const int hid = blockDim.y * blockIdx.y + threadIdx.y;
    if (hid < halfM && wid < halfN) {
        int cond = ((hid != 0) << 1) | (wid != 0);
        switch (cond) {
            case 0: {
                T tmp1;
                TComplex tmp_up;

                output[0].x = 0;
                output[0].y = 0;

                output[halfN].x = 0;
                output[halfN].y = 0;

                tmp1 = input[INDEX(halfM, 0, N)];
                tmp_up.x = tmp1;
                tmp_up.y = tmp1;
                output[INDEX(halfM, 0, halfN + 1)] = complexConj(complexMul(expkM[halfM], tmp_up));

                tmp1 = input[INDEX(halfM, halfN, N)];
                tmp_up.x = 0;
                tmp_up.y = 2 * tmp1;
                output[INDEX(halfM, halfN, halfN + 1)] =
                    complexConj(complexMul(complexMul(expkM[halfM], expkN[halfN]), tmp_up));
                break;
            }

            case 1: {
                output[wid].x = 0;
                output[wid].y = 0;

                TComplex tmp_up;
                T tmp1 = input[INDEX(halfM, wid, N)];
                T tmp2 = input[INDEX(halfM, N - wid, N)];
                tmp_up.x = tmp1 - tmp2;
                tmp_up.y = tmp1 + tmp2;
                output[INDEX(halfM, wid, halfN + 1)] =
                    complexConj(complexMul(complexMul(expkM[halfM], expkN[wid]), tmp_up));
                break;
            }

            case 2: {
                T tmp1, tmp3;
                TComplex tmp_up, tmp_down;

                tmp1 = input[INDEX(M - hid, 0, N)];
                tmp3 = input[INDEX(hid, 0, N)];
                tmp_up.x = tmp1;
                tmp_up.y = tmp3;
                tmp_down.x = tmp3;
                tmp_down.y = tmp1;

                output[INDEX(hid, 0, halfN + 1)] = complexConj(complexMul(expkM[hid], tmp_up));
                output[INDEX(M - hid, 0, halfN + 1)] = complexConj(complexMul(expkM[M - hid], tmp_down));

                tmp1 = input[INDEX(M - hid, halfN, N)];
                tmp3 = input[INDEX(hid, halfN, N)];
                tmp_up.x = tmp1 - tmp3;
                tmp_up.y = tmp3 + tmp1;
                tmp_down.x = tmp3 - tmp1;
                tmp_down.y = tmp1 + tmp3;

                output[INDEX(hid, halfN, halfN + 1)] =
                    complexConj(complexMul(complexMul(expkM[hid], expkN[halfN]), tmp_up));
                output[INDEX(M - hid, halfN, halfN + 1)] =
                    complexConj(complexMul(complexMul(expkM[M - hid], expkN[halfN]), tmp_down));
                break;
            }

            case 3: {
                T tmp1 = input[INDEX(M - hid, wid, N)];
                T tmp2 = input[INDEX(M - hid, N - wid, N)];
                T tmp3 = input[INDEX(hid, wid, N)];
                T tmp4 = input[INDEX(hid, N - wid, N)];
                TComplex tmp_up, tmp_down;
                tmp_up.x = tmp1 - tmp4;
                tmp_up.y = tmp3 + tmp2;
                tmp_down.x = tmp3 - tmp2;
                tmp_down.y = tmp1 + tmp4;

                output[INDEX(hid, wid, halfN + 1)] =
                    complexConj(complexMul(complexMul(expkM[hid], expkN[wid]), tmp_up));
                output[INDEX(M - hid, wid, halfN + 1)] =
                    complexConj(complexMul(complexMul(expkM[M - hid], expkN[wid]), tmp_down));
                break;
            }

            default:
                break;
        }
    }
}

template <typename T>
void idxst_idctPreprocessCudaLauncher(
    const T *x, T *y, const int M, const int N, const T *__restrict__ expkM, const T *__restrict__ expkN) {
    dim3 gridSize((N / 2 + TPB - 1) / TPB, (M / 2 + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    auto stream = at::cuda::getCurrentCUDAStream();
    idxst_idctPreprocess<T, ComplexType<T>><<<gridSize, blockSize, 0, stream>>>(
        x, (ComplexType<T> *)y, M, N, M / 2, N / 2, (ComplexType<T> *)expkM, (ComplexType<T> *)expkN);
}

// Adpated from idct2d_postprocess() with changes on sign and scale
// if (hid % 2 == 1)
//     new_output[hid][wid] = -output[hid][wid];
// else
//     new_output[hid][wid] = output[hid][wid];
template <typename T>
__global__ void idxst_idctPostprocess(const T *x, T *y, const int M, const int N, const int halfN, const int MN) {
    const int wid = blockDim.x * blockIdx.x + threadIdx.x;
    const int hid = blockDim.y * blockIdx.y + threadIdx.y;
    if (hid < M && wid < N) {
        int cond = ((hid < M / 2) << 1) | (wid < N / 2);
        int index;
        switch (cond) {
            case 0:
                index = INDEX(((M - hid) << 1) - 1, ((N - wid) << 1) - 1, N);
                y[index] = -x[INDEX(hid, wid, N)] * MN;
                break;
            case 1:
                index = INDEX(((M - hid) << 1) - 1, wid << 1, N);
                y[index] = -x[INDEX(hid, wid, N)] * MN;
                break;
            case 2:
                index = INDEX(hid << 1, ((N - wid) << 1) - 1, N);
                y[index] = x[INDEX(hid, wid, N)] * MN;
                break;
            case 3:
                index = INDEX(hid << 1, wid << 1, N);
                y[index] = x[INDEX(hid, wid, N)] * MN;
                break;
            default:
                break;
        }
    }
}

template <typename T>
void idxst_idctPostprocessCudaLauncher(const T *x, T *y, const int M, const int N) {
    dim3 gridSize((N + TPB - 1) / TPB, (M + TPB - 1) / TPB, 1);
    dim3 blockSize(TPB, TPB, 1);
    auto stream = at::cuda::getCurrentCUDAStream();
    idxst_idctPostprocess<T><<<gridSize, blockSize, 0, stream>>>(x, y, M, N, N / 2, M * N);
}

// launch

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void dct2_fft2_forward_cuda(at::Tensor x, at::Tensor expkM, at::Tensor expkN, at::Tensor out, at::Tensor buf) {
    cudaSetDevice(x.get_device());
    CHECK_INPUT(x);
    CHECK_INPUT(expkM);
    CHECK_INPUT(expkN);
    CHECK_INPUT(out);
    CHECK_INPUT(buf);

    auto N = x.size(-1);
    auto M = x.numel() / N;

    dct2dPreprocessCudaLauncher<float>(x.data_ptr<float>(), out.data_ptr<float>(), M, N);

    buf = at::view_as_real(at::fft_rfft2(out, c10::nullopt, {-2, -1}, "backward")).contiguous();

    dct2dPostprocessCudaLauncher<float>(
        buf.data_ptr<float>(), out.data_ptr<float>(), M, N, expkM.data_ptr<float>(), expkN.data_ptr<float>());
}

void idct2_fft2_forward_cuda(at::Tensor x, at::Tensor expkM, at::Tensor expkN, at::Tensor out, at::Tensor buf) {
    cudaSetDevice(x.get_device());
    CHECK_INPUT(x);
    CHECK_INPUT(expkM);
    CHECK_INPUT(expkN);
    CHECK_INPUT(out);
    CHECK_INPUT(buf);

    auto N = x.size(-1);
    auto M = x.numel() / N;

    idct2_fft2PreprocessCudaLauncher<float>(
        x.data_ptr<float>(), buf.data_ptr<float>(), M, N, expkM.data_ptr<float>(), expkN.data_ptr<float>());

    auto y = at::fft_irfft2(at::view_as_complex(buf), {{M, N}}, {-2, -1}, "backward").contiguous();

    idct2_fft2PostprocessCudaLauncher<float>(y.data_ptr<float>(), out.data_ptr<float>(), M, N);
}

void idct_idxst_forward_cuda(at::Tensor x, at::Tensor expkM, at::Tensor expkN, at::Tensor out, at::Tensor buf) {
    cudaSetDevice(x.get_device());
    CHECK_INPUT(x);
    CHECK_INPUT(expkM);
    CHECK_INPUT(expkN);
    CHECK_INPUT(out);
    CHECK_INPUT(buf);

    auto N = x.size(-1);
    auto M = x.numel() / N;

    idct_idxstPreprocessCudaLauncher<float>(
        x.data_ptr<float>(), buf.data_ptr<float>(), M, N, expkM.data_ptr<float>(), expkN.data_ptr<float>());

    auto y = at::fft_irfft2(at::view_as_complex(buf), {{M, N}}, {-2, -1}, "backward").contiguous();

    idct_idxstPostprocessCudaLauncher<float>(y.data_ptr<float>(), out.data_ptr<float>(), M, N);
}

void idxst_idct_forward_cuda(at::Tensor x, at::Tensor expkM, at::Tensor expkN, at::Tensor out, at::Tensor buf) {
    cudaSetDevice(x.get_device());
    CHECK_INPUT(x);
    CHECK_INPUT(expkM);
    CHECK_INPUT(expkN);
    CHECK_INPUT(out);
    CHECK_INPUT(buf);

    auto N = x.size(-1);
    auto M = x.numel() / N;

    idxst_idctPreprocessCudaLauncher<float>(
        x.data_ptr<float>(), buf.data_ptr<float>(), M, N, expkM.data_ptr<float>(), expkN.data_ptr<float>());

    auto y = at::fft_irfft2(at::view_as_complex(buf), {{M, N}}, {-2, -1}, "backward").contiguous();

    idxst_idctPostprocessCudaLauncher<float>(y.data_ptr<float>(), out.data_ptr<float>(), M, N);
}