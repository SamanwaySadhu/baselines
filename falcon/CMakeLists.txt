cmake_minimum_required(VERSION 3.21)
project(lr_falcon LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Use an out-of-source build: cmake -S . -B build")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# emp-tool installs a CMake package (pass -DCMAKE_PREFIX_PATH=<prefix>); its target carries C++20,
# -march=native, OpenSSL and pthreads. Only its PRG (AES-CTR) and Hash (SHA-256) are used; networking is
# raw POSIX sockets (include/net.h). The upstream Falcon sources (scripts/fetch_upstream.sh) are reference
# material only and are not compiled.
find_package(emp-tool 1.0 CONFIG REQUIRED)

add_library(falcon_common INTERFACE)
target_include_directories(falcon_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../wan)
target_link_libraries(falcon_common INTERFACE emp-tool::emp-tool ${CMAKE_DL_LIBS})
target_compile_features(falcon_common INTERFACE cxx_std_20)
target_compile_options(falcon_common INTERFACE -Wall -Wextra)

add_executable(lr_falcon src/lr_falcon.cpp)
target_link_libraries(lr_falcon PRIVATE falcon_common)

add_executable(lr_falcon_selftest src/lr_falcon_selftest.cpp)
target_link_libraries(lr_falcon_selftest PRIVATE falcon_common)
