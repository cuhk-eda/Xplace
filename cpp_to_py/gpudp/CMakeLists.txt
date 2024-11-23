# Files
file(GLOB_RECURSE SRC_FILES_DP ${CMAKE_CURRENT_SOURCE_DIR}/check/*.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/dp/*.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/db/*.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/lg/*.cpp)
file(GLOB_RECURSE SRC_FILES_DP_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

# OpenMP
find_package(OpenMP REQUIRED)

# CUDA/CPP DP Kernel
add_library(dp STATIC ${CMAKE_CURRENT_SOURCE_DIR}/../io_parser/gp/GPDatabase.cpp
                      ${SRC_FILES_DP}
                      ${SRC_FILES_DP_CUDA})

set_target_properties(dp PROPERTIES
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON
                      POSITION_INDEPENDENT_CODE ON)

target_include_directories(dp PRIVATE ${TORCH_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/cpp_to_py)
target_link_libraries(dp PRIVATE torch ${TORCH_PYTHON_LIBRARY} xplace_common OpenMP::OpenMP_CXX pthread CUDA::curand)
target_compile_options(dp PRIVATE -fPIC)

pybind11_add_module(gpudp MODULE PyBindCppMain.cpp)

target_include_directories(gpudp PRIVATE ${TORCH_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/cpp_to_py)
target_link_libraries(gpudp PRIVATE torch ${TORCH_PYTHON_LIBRARY} xplace_common io_parser dp)
target_compile_definitions(gpudp PRIVATE 
                            TORCH_EXTENSION_NAME=gpudp
                            TORCH_VERSION_MAJOR=${TORCH_VERSION_MAJOR}
                            TORCH_VERSION_MINOR=${TORCH_VERSION_MINOR}
                            ENABLE_CUDA=${TORCH_ENABLE_CUDA})

install(TARGETS gpudp DESTINATION ${XPLACE_LIB_DIR})
