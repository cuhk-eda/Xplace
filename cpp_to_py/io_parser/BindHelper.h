#pragma once
#include "common/common.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace Xplace {

void bindGPDatabase(pybind11::module& m);

}  // namespace Xplace