# Files
file(GLOB_RECURSE SRC_FILES_DP ${CMAKE_CURRENT_SOURCE_DIR}/check/*.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/dp/*.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/db/*.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/lg/*.cpp)
file(GLOB_RECURSE SRC_FILES_DP_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

# OpenMP
find_package(OpenMP REQUIRED)

# CUDA DP Kernel
cuda_add_library(dp_cuda_tmp STATIC ${SRC_FILES_DP_CUDA})

set_target_properties(dp_cuda_tmp PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(dp_cuda_tmp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(dp_cuda_tmp PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(dp_cuda_tmp PRIVATE ${PROJECT_SOURCE_DIR}/cpp_to_py ${TORCH_INCLUDE_DIRS} ${FLUTE_INCLUDE_DIR})
target_link_libraries(dp_cuda_tmp torch ${TORCH_PYTHON_LIBRARY} xplace_common flute OpenMP::OpenMP_CXX)

# CPU DP object
add_library(dp SHARED ${CMAKE_CURRENT_SOURCE_DIR}/../io_parser/gp/GPDatabase.cpp
                       ${SRC_FILES_DP}
                       ${SRC_FILES_DP_CUDA})

target_include_directories(dp PRIVATE ${PROJECT_SOURCE_DIR}/cpp_to_py ${TORCH_INCLUDE_DIRS} ${FLUTE_INCLUDE_DIR})
target_link_libraries(dp PRIVATE torch ${TORCH_PYTHON_LIBRARY} xplace_common flute dp_cuda_tmp pthread CUDA::curand)
target_compile_options(dp PRIVATE -fPIC)

install(TARGETS dp DESTINATION ${XPLACE_LIB_DIR})

# For Pybind
add_pytorch_extension(gpudp PyBindCppMain.cpp
    EXTRA_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/cpp_to_py ${FLUTE_INCLUDE_DIR}
    EXTRA_LINK_LIBRARIES xplace_common flute io_parser dp)

install(TARGETS gpudp DESTINATION ${XPLACE_LIB_DIR})
