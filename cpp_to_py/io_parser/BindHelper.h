#pragma once
#include "common/common.h"
#include "common/db/Database.h"
#include "io_parser/gp/GPDatabase.h"

namespace py = pybind11;

namespace Xplace {

void bindGPDatabase(pybind11::module& m);

}  // namespace Xplace