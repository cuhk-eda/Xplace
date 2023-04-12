#pragma once

#include "gpudp/dp/detailed_place_db.cuh"

namespace dp {

template <typename T>
__host__ __device__ bool adjust_pos(T& x, T width, const Space<T>& space) {
    // the order is very tricky for numerical stability
    x = min(x, space.xh - width);
    x = max(x, space.xl);
    return width + space.xl <= space.xh;
}

}  // namespace dp