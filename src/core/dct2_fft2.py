# The implementation of DCT2 mainly follows DREAMPlace. We rewrite the code 
# structure and use DCT2FFT2Cache to cache the intermediate results.

import numpy as np
import torch
from cpp_to_py import dct_cuda


class DCT2FFT2Cache:
    def __init__(self) -> None:
        self.expks_cache = {}
        self.dct2_cache = {}
        self.idct2_cache = {}
        self.idct_idxst_cache = {}
        self.idxst_idct_cache = {}

    def reset(self):
        self.expks_cache = {}
        self.dct2_cache = {}
        self.idct2_cache = {}
        self.idct_idxst_cache = {}
        self.idxst_idct_cache = {}


dct2_fft2_cache = DCT2FFT2Cache()


def precompute_expk(N, dtype, device):
    # Compute exp(-j*pi*u/(2N)) = cos(pi*u/(2N)) - j * sin(pi*u/(2N))
    pik_by_2N = torch.arange(N, dtype=dtype, device=device)
    pik_by_2N.mul_(np.pi / (2 * N))
    # cos, -sin
    expk = torch.stack([pik_by_2N.cos(), -pik_by_2N.sin()], dim=-1)
    return expk.contiguous()


def dct2(x):
    if x.shape not in dct2_fft2_cache.expks_cache.keys():
        expkM = precompute_expk(x.size(-2), dtype=x.dtype, device=x.device)
        expkN = precompute_expk(x.size(-1), dtype=x.dtype, device=x.device)
        dct2_fft2_cache.expks_cache[x.shape] = (expkM, expkN)
    expkM, expkN = dct2_fft2_cache.expks_cache[x.shape]

    if x.shape not in dct2_fft2_cache.dct2_cache.keys():
        out = torch.empty(x.size(-2), x.size(-1), dtype=x.dtype, device=x.device)
        buf = torch.empty(
            x.size(-2), x.size(-1) // 2 + 1, 2, dtype=x.dtype, device=x.device
        )
        dct2_fft2_cache.dct2_cache[x.shape] = (out, buf)
    out, buf = dct2_fft2_cache.dct2_cache[x.shape]
    dct_cuda.dct2_fft2(x, expkM, expkN, out, buf)
    return out


def idct2(x):
    if x.shape not in dct2_fft2_cache.expks_cache.keys():
        expkM = precompute_expk(x.size(-2), dtype=x.dtype, device=x.device)
        expkN = precompute_expk(x.size(-1), dtype=x.dtype, device=x.device)
        dct2_fft2_cache.expks_cache[x.shape] = (expkM, expkN)
    expkM, expkN = dct2_fft2_cache.expks_cache[x.shape]

    if x.shape not in dct2_fft2_cache.idct2_cache.keys():
        out = torch.empty(x.size(-2), x.size(-1), dtype=x.dtype, device=x.device)
        buf = torch.empty(
            x.size(-2), x.size(-1) // 2 + 1, 2, dtype=x.dtype, device=x.device
        )
        dct2_fft2_cache.idct2_cache[x.shape] = (out, buf)
    out, buf = dct2_fft2_cache.idct2_cache[x.shape]

    dct_cuda.idct2_fft2(x, expkM, expkN, out, buf)
    return out


def idct_idxst(x):
    if x.shape not in dct2_fft2_cache.expks_cache.keys():
        expkM = precompute_expk(x.size(-2), dtype=x.dtype, device=x.device)
        expkN = precompute_expk(x.size(-1), dtype=x.dtype, device=x.device)
        dct2_fft2_cache.expks_cache[x.shape] = (expkM, expkN)
    expkM, expkN = dct2_fft2_cache.expks_cache[x.shape]

    if x.shape not in dct2_fft2_cache.idct_idxst_cache.keys():
        out = torch.empty(x.size(-2), x.size(-1), dtype=x.dtype, device=x.device)
        buf = torch.empty(
            x.size(-2), x.size(-1) // 2 + 1, 2, dtype=x.dtype, device=x.device
        )
        dct2_fft2_cache.idct_idxst_cache[x.shape] = (out, buf)
    out, buf = dct2_fft2_cache.idct_idxst_cache[x.shape]

    dct_cuda.idct_idxst(x, expkM, expkN, out, buf)
    return out


def idxst_idct(x):
    if x.shape not in dct2_fft2_cache.expks_cache.keys():
        expkM = precompute_expk(x.size(-2), dtype=x.dtype, device=x.device)
        expkN = precompute_expk(x.size(-1), dtype=x.dtype, device=x.device)
        dct2_fft2_cache.expks_cache[x.shape] = (expkM, expkN)
    expkM, expkN = dct2_fft2_cache.expks_cache[x.shape]

    if x.shape not in dct2_fft2_cache.idxst_idct_cache.keys():
        out = torch.empty(x.size(-2), x.size(-1), dtype=x.dtype, device=x.device)
        buf = torch.empty(
            x.size(-2), x.size(-1) // 2 + 1, 2, dtype=x.dtype, device=x.device
        )
        dct2_fft2_cache.idxst_idct_cache[x.shape] = (out, buf)
    out, buf = dct2_fft2_cache.idxst_idct_cache[x.shape]

    dct_cuda.idxst_idct(x, expkM, expkN, out, buf)
    return out
