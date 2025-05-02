#pragma once
#include "common/common.h"
#include "common/lib/Helper.h"

namespace gt::sdc {

using token_iter_t = std::vector<std::string>::iterator;

std::vector<std::string> tokenize(const std::string&);

};  // namespace gt::sdc
