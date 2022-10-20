file(GLOB_RECURSE SRC_FILES_IO_PARSER ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

pybind11_add_module(io_parser SHARED ${SRC_FILES_IO_PARSER})
target_include_directories(
    io_parser PUBLIC ${PROJECT_SOURCE_DIR}/cpp_to_py ${TORCH_INCLUDE_DIRS})
target_link_libraries(
    io_parser PRIVATE torch ${TORCH_PYTHON_LIBRARY} xplace_common)
target_compile_definitions(io_parser PRIVATE 
    TORCH_EXTENSION_NAME=io_parser
    TORCH_MAJOR_VERSION=${TORCH_MAJOR_VERSION}
    TORCH_MINOR_VERSION=${TORCH_MINOR_VERSION}
    ENABLE_CUDA=0)
target_compile_options(io_parser PRIVATE -fPIC)

install(TARGETS 
    io_parser
    DESTINATION ${XPLACE_LIB_DIR})