# Files
file(GLOB_RECURSE SRC_FILES_GGR ${CMAKE_CURRENT_SOURCE_DIR}/db/*.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/gr/*.cpp)
file(GLOB_RECURSE SRC_FILES_GGR_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

# CUDA GGR Kernel
cuda_add_library(ggr_cuda_tmp STATIC ${SRC_FILES_GGR_CUDA})

set_target_properties(ggr_cuda_tmp PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(ggr_cuda_tmp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(ggr_cuda_tmp PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(ggr_cuda_tmp PRIVATE ${PROJECT_SOURCE_DIR}/cpp_to_py ${TORCH_INCLUDE_DIRS} ${FLUTE_INCLUDE_DIR})
target_link_libraries(ggr_cuda_tmp torch ${TORCH_PYTHON_LIBRARY} xplace_common flute)
# target_compile_options(ggr_cuda_tmp PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-use_fast_math>") # not work...


# CPU GGR object
add_library(ggr SHARED ${CMAKE_CURRENT_SOURCE_DIR}/../io_parser/gp/GPDatabase.cpp
                       ${SRC_FILES_GGR}
                       ${SRC_FILES_GGR_CUDA})

target_include_directories(ggr PRIVATE ${PROJECT_SOURCE_DIR}/cpp_to_py ${TORCH_INCLUDE_DIRS} ${FLUTE_INCLUDE_DIR})
target_link_libraries(ggr PRIVATE torch ${TORCH_PYTHON_LIBRARY} xplace_common flute ggr_cuda_tmp pthread)
target_compile_options(ggr PRIVATE -fPIC)

install(TARGETS ggr DESTINATION ${XPLACE_LIB_DIR})

# For Pybind
add_pytorch_extension(gpugr PyBindCppMain.cpp
    EXTRA_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/cpp_to_py ${FLUTE_INCLUDE_DIR}
    EXTRA_LINK_LIBRARIES xplace_common flute io_parser ggr)

install(TARGETS gpugr DESTINATION ${XPLACE_LIB_DIR})

################ For debug only ##################

# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA17_EXTENSION_COMPILE_OPTION "-std=c++17")
# set(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -arch sm_86 --extended-lambda --use_fast_math ")
# project(ggr LANGUAGES C CXX CUDA)

# file(GLOB_RECURSE SRC_FILES_GR2 ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# file(GLOB_RECURSE SRC_FILES_GR2_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

# add_executable(gpugr_cpp PyBindCppMain.cpp 
#             ${CMAKE_CURRENT_SOURCE_DIR}/../io_parser/gp/GPDatabase.cpp
#             ${SRC_FILES_GR2} 
#             ${SRC_FILES_GR2_CUDA})

# set_target_properties(gpugr_cpp PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# set_target_properties(gpugr_cpp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(gpugr_cpp PROPERTIES LINK_FLAGS "-Wl,--whole-archive -rdynamic -lpthread -Wl,--no-whole-archive")

# target_include_directories(
#     gpugr_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${PROJECT_SOURCE_DIR}/cpp_to_py ${FLUTE_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
# target_link_libraries(
#     gpugr_cpp PRIVATE torch ${TORCH_PYTHON_LIBRARY} flute xplace_common)

# target_compile_options(gpugr_cpp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#     -arch=sm_86
#     --use_fast_math
#     -std=c++17
#     >)

# install(TARGETS 
#     gpugr_cpp
#     DESTINATION ${XPLACE_LIB_DIR})