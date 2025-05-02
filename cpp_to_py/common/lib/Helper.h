#pragma once

#include <dirent.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <forward_list>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <ratio>
#include <regex>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <stack>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#if defined(__has_include) && __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std {
namespace filesystem = experimental::filesystem;
};
#endif

#include "common/common.h"
#include "units.hpp"

namespace gt {

using namespace std::chrono_literals;
using namespace std::literals::string_literals;

enum Split {
  MIN = 0,
  MAX = 1
};

enum Tran {
  RISE = 0,
  FALL = 1
};

constexpr int MAX_SPLIT = 2;
constexpr int MAX_TRAN = 2;

constexpr std::initializer_list<Split> SPLIT = {MIN, MAX};
constexpr std::initializer_list<Tran> TRAN = {RISE, FALL};
constexpr std::initializer_list<std::pair<Split, Tran>> SPLIT_TRAN = {{MIN, RISE}, {MIN, FALL}, {MAX, RISE}, {MAX, FALL}};

#define for_each_el(el) for (auto el : SPLIT)
#define for_each_el_rf_if(el, rf, c) \
    for (auto [el, rf] : SPLIT_TRAN) \
        if (c)

};  // namespace gt
