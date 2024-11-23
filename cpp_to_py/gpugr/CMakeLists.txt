# Files
file(GLOB_RECURSE SRC_FILES_GGR ${CMAKE_CURRENT_SOURCE_DIR}/db/*.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/gr/*.cpp)
file(GLOB_RECURSE SRC_FILES_GGR_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

# CUDA/CPP GGR Kernel
add_library(ggr STATIC ${CMAKE_CURRENT_SOURCE_DIR}/../io_parser/gp/GPDatabase.cpp
                       ${SRC_FILES_GGR}
                       ${SRC_FILES_GGR_CUDA})

set_target_properties(ggr PROPERTIES
                       CUDA_RESOLVE_DEVICE_SYMBOLS ON
                       POSITION_INDEPENDENT_CODE ON)

target_include_directories(ggr PRIVATE ${TORCH_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/cpp_to_py ${FLUTE_INCLUDE_DIR})
target_link_libraries(ggr PRIVATE torch ${TORCH_PYTHON_LIBRARY} xplace_common flute pthread)
target_compile_options(ggr PRIVATE -fPIC)
target_compile_options(ggr PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>")

pybind11_add_module(gpugr MODULE PyBindCppMain.cpp)

target_include_directories(gpugr PRIVATE ${TORCH_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/cpp_to_py ${FLUTE_INCLUDE_DIR})
target_link_libraries(gpugr PRIVATE torch ${TORCH_PYTHON_LIBRARY} xplace_common flute io_parser ggr)
target_compile_definitions(gpugr PRIVATE 
                            TORCH_EXTENSION_NAME=gpugr
                            TORCH_VERSION_MAJOR=${TORCH_VERSION_MAJOR}
                            TORCH_VERSION_MINOR=${TORCH_VERSION_MINOR}
                            ENABLE_CUDA=${TORCH_ENABLE_CUDA})


install(TARGETS gpugr DESTINATION ${XPLACE_LIB_DIR})
