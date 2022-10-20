#pragma once

// STL libraries
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <numeric>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Torch library
#include <torch/extension.h>

// Pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Cairo
#include <cairo-pdf.h>
#include <cairo-ps.h>
#include <cairo-svg.h>
#include <cairo.h>

using index_type = int64_t;