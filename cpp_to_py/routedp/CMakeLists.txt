# Files
file(GLOB_RECURSE SRC_FILES_CPUDP ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# OpenMP
find_package(OpenMP REQUIRED)

# For Pybind
add_pytorch_extension(routedp PyBindCppMain.cpp
    EXTRA_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/cpp_to_py ${FLUTE_INCLUDE_DIR} ${LEMON_INCLUDE_DIRS}
    EXTRA_LINK_LIBRARIES xplace_common flute io_parser ${LEMON_LIBRARIES})

install(TARGETS routedp DESTINATION ${XPLACE_LIB_DIR})
