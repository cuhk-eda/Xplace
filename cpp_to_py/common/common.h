#pragma once

// STL libraries
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <random>
#include <memory>
#include <algorithm>
#include <list>
#include <map>
#include <set>
#include <queue>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <cstring>

#include <thread>

// Torch library
#include <torch/extension.h>

// utils
#include "common/utils/utils.h"

// Pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

using namespace std;

using utils::assert_msg;
using utils::logger;