cmake_minimum_required(VERSION 3.21)
project(lr_spdz LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Use an out-of-source build: cmake -S . -B build")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# emp-tool and emp-ot install CMake packages (pass -DCMAKE_PREFIX_PATH=<prefix>); their targets carry
# C++20, -march=native, OpenSSL and pthreads. No other third-party code is used.
find_package(emp-tool 1.0 CONFIG REQUIRED)
find_package(emp-ot 1.0 CONFIG REQUIRED)

add_library(spz_common INTERFACE)
target_include_directories(spz_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(spz_common INTERFACE emp-ot::emp-ot emp-tool::emp-tool)
target_compile_features(spz_common INTERFACE cxx_std_20)
target_compile_options(spz_common INTERFACE -Wall -Wextra)

add_executable(lr_spdz src/lr_spdz.cpp)
target_link_libraries(lr_spdz PRIVATE spz_common)

add_executable(lr_spdz_selftest src/lr_spdz_selftest.cpp)
target_link_libraries(lr_spdz_selftest PRIVATE spz_common)
