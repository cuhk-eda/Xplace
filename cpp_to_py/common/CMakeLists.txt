file(GLOB_RECURSE SRC_FILES_XPLACE_COMMON ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
find_library(LIBDEF def ${PATH_THIRDPARTY_ROOT}/lefdef/def58/lib)
find_library(LIBLEF lef ${PATH_THIRDPARTY_ROOT}/lefdef/lef58/lib)

add_library(xplace_common SHARED ${SRC_FILES_XPLACE_COMMON})

target_include_directories(
    xplace_common PRIVATE ${PROJECT_SOURCE_DIR}/cpp_to_py ${PATH_THIRDPARTY_ROOT}/lefdef ${TORCH_INCLUDE_DIRS})
target_link_libraries(
    xplace_common PRIVATE torch ${TORCH_PYTHON_LIBRARY} ${LIBDEF} ${LIBLEF})
target_compile_options(xplace_common PRIVATE -fPIC)

install(TARGETS 
    xplace_common
    DESTINATION ${XPLACE_LIB_DIR})
