file(GLOB_RECURSE SRC_FILES_XPLACE_COMMON ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
find_library(LIBDEF def ${PATH_THIRDPARTY_ROOT}/lefdef/def58/lib)
find_library(LIBLEF lef ${PATH_THIRDPARTY_ROOT}/lefdef/lef58/lib)

# BISON/FLEX
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/io/verilog)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)
message(STATUS "BISON_EXECUTABLE=${BISON_EXECUTABLE}")
message(STATUS "FLEX_EXECUTABLE=${FLEX_EXECUTABLE}")

# Verilog scan/parse.
bison_target(verilog_parser ${CMAKE_CURRENT_SOURCE_DIR}/io/verilog/verilog_parser.yy ${CMAKE_CURRENT_BINARY_DIR}/verilog_parser.tab.cc)
flex_target(verilog_lexer ${CMAKE_CURRENT_SOURCE_DIR}/io/verilog/verilog_lexer.l ${CMAKE_CURRENT_BINARY_DIR}/verilog_lexer.yy.cc)
add_flex_bison_dependency(verilog_lexer verilog_parser)

add_library(xplace_common SHARED ${SRC_FILES_XPLACE_COMMON} ${FLEX_verilog_lexer_OUTPUTS} ${BISON_verilog_parser_OUTPUTS})

find_package(ZLIB)

target_include_directories(
    xplace_common PRIVATE ${PROJECT_SOURCE_DIR}/cpp_to_py ${PATH_THIRDPARTY_ROOT}/lefdef ${TORCH_INCLUDE_DIRS})
target_link_libraries(
    xplace_common PRIVATE torch stdc++fs ${TORCH_PYTHON_LIBRARY} ${LIBDEF} ${LIBLEF} ZLIB::ZLIB)
target_compile_options(xplace_common PRIVATE -fPIC)

install(TARGETS 
    xplace_common
    DESTINATION ${XPLACE_LIB_DIR})
