cmake_minimum_required(VERSION 3.12)

project(xplace C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message(STATUS PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR})
set(PATH_THIRDPARTY_ROOT ${PROJECT_SOURCE_DIR}/thirdparty)
set(XPLACE_LIB_DIR ${PROJECT_SOURCE_DIR}/cpp_to_py/cpybin)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_INSTALL_RPATH ${XPLACE_LIB_DIR})

# set _GLIBCXX_USE_CXX11_ABI for torch
if(NOT CMAKE_CXX_ABI)
    set(CMAKE_CXX_ABI 0 CACHE STRING
        "Choose the value for _GLIBCXX_USE_CXX11_ABI, options are: 0|1."
        FORCE)
endif(NOT CMAKE_CXX_ABI)
message(STATUS "CMAKE_CXX_ABI: _GLIBCXX_USE_CXX11_ABI=${CMAKE_CXX_ABI}")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${CMAKE_CXX_ABI})

# PyBind11
add_subdirectory(${PATH_THIRDPARTY_ROOT}/pybind11)
message(STATUS "PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")

# Flute
add_subdirectory(${PATH_THIRDPARTY_ROOT}/flute)
message(STATUS "FLUTE_INCLUDE_DIR: ${FLUTE_INCLUDE_DIR}")

# Lemon
set(LEMON_INCLUDE_DIR "${PATH_THIRDPARTY_ROOT}/lemon/include")
set(LEMON_INCLUDE_DIRS "${LEMON_INCLUDE_DIR}")
find_library(LEMON_LIBRARY emon ${PATH_THIRDPARTY_ROOT}/lemon/lib)
set(LEMON_LIBRARIES "${LEMON_LIBRARY}")
message(STATUS "LEMON_INCLUDE_DIRS: ${LEMON_INCLUDE_DIRS}")
message(STATUS "LEMON_LIBRARIES: ${LEMON_LIBRARIES}")

# Cairo
find_package(Cairo)
message(STATUS "CAIRO_INCLUDE_DIRS: ${CAIRO_INCLUDE_DIRS}")
message(STATUS "CAIRO_LIBRARIES: ${CAIRO_LIBRARIES}")

# Get Torch info
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c 
  "import torch; print(torch.__path__[0]); print(int(torch.cuda.is_available())); print(torch.__version__);" 
  OUTPUT_VARIABLE TORCH_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "\n" ";" TORCH_OUTPUT_LIST ${TORCH_OUTPUT})
list(GET TORCH_OUTPUT_LIST 0 TORCH_INSTALL_PREFIX)
list(GET TORCH_OUTPUT_LIST 1 TORCH_ENABLE_CUDA)
list(GET TORCH_OUTPUT_LIST 2 TORCH_VERSION)
string(REPLACE "." ";" TORCH_VERSION_LIST ${TORCH_VERSION})
list(GET TORCH_VERSION_LIST 0 TORCH_VERSION_MAJOR)
list(GET TORCH_VERSION_LIST 1 TORCH_VERSION_MINOR)
message(STATUS TORCH_INSTALL_PREFIX=${TORCH_INSTALL_PREFIX})
message(STATUS TORCH_VERSION=${TORCH_VERSION_MAJOR}.${TORCH_VERSION_MINOR})

# find CUDA
if (TORCH_ENABLE_CUDA)
  find_package(CUDA 11.4)
  find_package(CUDAToolkit)
  if (NOT CUDA_FOUND)
    set(TORCH_ENABLE_CUDA 0 CACHE BOOL "Whether enable CUDA" FORCE)
    message(FATAL_ERROR "Xplace only supports CUDA mode, CMake will exit." )
  endif(NOT CUDA_FOUND)
endif()
message(STATUS TORCH_ENABLE_CUDA=${TORCH_ENABLE_CUDA})

# set cuda arch and nvcc flags
if (CUDA_FOUND)
  if (NOT CUDA_ARCH_LIST)
    set(CUDA_ARCH_LIST 6.0 6.1 7.0 7.5 8.0 8.6)
  endif(NOT CUDA_ARCH_LIST)
  # for cuda_add_library
  cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
  message(STATUS "CUDA_ARCH_FLAGS: ${CUDA_ARCH_FLAGS}")
  # set nvcc flags
  set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
  set(CMAKE_CUDA17_EXTENSION_COMPILE_OPTION "-std=c++17")
  list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS} --compiler-options;-fPIC;-std=c++17)
  list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS} --extended-lambda)
  list(APPEND TORCH_NVCC_FLAGS --expt-relaxed-constexpr)
  list(APPEND CUDA_NVCC_FLAGS ${TORCH_NVCC_FLAGS})
  message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
endif(CUDA_FOUND)

# set torch lib
add_library(torch STATIC IMPORTED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
find_library(TORCH_LIBRARY torch PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
find_library(C10_LIBRARY c10 PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
find_library(C10_CUDA_LIBRARY c10_cuda PATHS "${TORCH_INSTALL_PREFIX}/lib")
find_library(TORCH_CPU_LIBRARY torch_cpu PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
find_library(TORCH_CUDA_LIBRARY torch_cuda PATHS "${TORCH_INSTALL_PREFIX}/lib")

set(LINK_LIBS ${C10_LIBRARY} ${TORCH_CPU_LIBRARY})
if (TORCH_ENABLE_CUDA)
  set(LINK_LIBS ${LINK_LIBS}
    ${C10_CUDA_LIBRARY}
    ${TORCH_CUDA_LIBRARY})
endif()

# set torch include
if (EXISTS ${TORCH_INSTALL_PREFIX}/include)
  set(TORCH_HEADER_PREFIX ${TORCH_INSTALL_PREFIX}/include)
endif()
set(TORCH_INCLUDE_DIRS
  ${PYTHON_INCLUDE_DIRS} ${TORCH_HEADER_PREFIX} ${TORCH_HEADER_PREFIX}/torch/csrc/api/include)
message(STATUS TORCH_INCLUDE_DIRS=${TORCH_INCLUDE_DIRS})

# set torch target
# adapt from https://github.com/limbo018/DREAMPlace/blob/master/cmake/TorchExtension.cmake
set_target_properties(torch PROPERTIES
  IMPORTED_LOCATION "${TORCH_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}"
  INTERFACE_LINK_LIBRARIES "${LINK_LIBS}"
  INTERFACE_COMPILE_OPTIONS "-D_GLIBCXX_USE_CXX11_ABI=${CMAKE_CXX_ABI}"
  )

function(add_pytorch_extension target_name)
  set(multiValueArgs EXTRA_INCLUDE_DIRS EXTRA_LINK_LIBRARIES EXTRA_DEFINITIONS)
  cmake_parse_arguments(ARG "" "" "${multiValueArgs}" ${ARGN})
  if (TORCH_ENABLE_CUDA)
    set(CUDA_SRCS "${ARG_UNPARSED_ARGUMENTS}")
    list(FILTER CUDA_SRCS INCLUDE REGEX ".*cu$")
    if (CUDA_SRCS)
      cuda_add_library(${target_name}_cuda_tmp STATIC ${CUDA_SRCS})
      target_include_directories(${target_name}_cuda_tmp PRIVATE ${ARG_EXTRA_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
      target_link_libraries(${target_name}_cuda_tmp ${ARG_EXTRA_LINK_LIBRARIES} ${TORCH_LIBRARY})
      target_compile_definitions(${target_name}_cuda_tmp PRIVATE 
        TORCH_EXTENSION_NAME=${target_name}
        TORCH_VERSION_MAJOR=${TORCH_VERSION_MAJOR}
        TORCH_VERSION_MINOR=${TORCH_VERSION_MINOR}
        ENABLE_CUDA=${TORCH_ENABLE_CUDA}
        ${ARG_EXTRA_DEFINITIONS})
      set_target_properties(${target_name}_cuda_tmp PROPERTIES 
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden"
        )
    endif()
  endif()
  list(FILTER ARG_UNPARSED_ARGUMENTS EXCLUDE REGEX ".*cu$")
  pybind11_add_module(${target_name} MODULE ${ARG_UNPARSED_ARGUMENTS})
  target_include_directories(${target_name} PRIVATE ${ARG_EXTRA_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
  if (TORCH_ENABLE_CUDA AND CUDA_SRCS)
    target_link_libraries(${target_name} PRIVATE ${target_name}_cuda_tmp ${ARG_EXTRA_LINK_LIBRARIES} torch ${TORCH_PYTHON_LIBRARY})
  else()
    target_link_libraries(${target_name} PRIVATE ${ARG_EXTRA_LINK_LIBRARIES} torch ${TORCH_PYTHON_LIBRARY})
  endif()
  target_compile_definitions(${target_name} PRIVATE 
    TORCH_EXTENSION_NAME=${target_name}
    TORCH_VERSION_MAJOR=${TORCH_VERSION_MAJOR}
    TORCH_VERSION_MINOR=${TORCH_VERSION_MINOR}
    ENABLE_CUDA=${TORCH_ENABLE_CUDA}
    ${ARG_EXTRA_DEFINITIONS})
endfunction()

# add targets
add_subdirectory(cpp_to_py)