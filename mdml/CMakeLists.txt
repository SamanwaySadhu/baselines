cmake_minimum_required(VERSION 3.16)
project(lr_mdml LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Use an out-of-source build: cmake -S . -B build")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# C++20 with GNU extensions: MD-ML needs __uint128_t to satisfy std::integral.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# MD-ML sources (scripts/fetch_deps.sh: pinned commit + peer-host patch).
set(MDML_DIR "${CMAKE_SOURCE_DIR}/third_party/MD-ML" CACHE PATH "MD-ML checkout")
if(NOT EXISTS "${MDML_DIR}/src/protocols/Gate.h")
  message(FATAL_ERROR "MD-ML not found at ${MDML_DIR}; run scripts/fetch_deps.sh")
endif()

# Eigen (header-only): system package if present, else third_party/eigen-3.4.0.
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  set(EIGEN_DIR "${CMAKE_SOURCE_DIR}/third_party/eigen-3.4.0")
  if(NOT EXISTS "${EIGEN_DIR}/Eigen/Core")
    message(FATAL_ERROR "Eigen not found; run scripts/fetch_deps.sh or install libeigen3-dev")
  endif()
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  target_include_directories(Eigen3::Eigen INTERFACE "${EIGEN_DIR}")
endif()

find_package(Boost 1.70 REQUIRED)   # header-only Boost.Asio
find_package(OpenSSL 1.1 REQUIRED)  # SHA-256 commitments, AES-CTR coins (MAC check)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)       # multi-core online phase (see --threads)

# MD-ML's FakeParty creates this directory; our programs pass absolute paths instead.
set(FAKE_OFFLINE_DIR "${CMAKE_BINARY_DIR}/fake-offline-data")

add_library(mdml STATIC
  ${MDML_DIR}/src/networking/Party.cpp
  ${MDML_DIR}/src/utils/Timer.cpp)
target_include_directories(mdml PUBLIC ${MDML_DIR}/src ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mdml SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
target_compile_definitions(mdml PUBLIC FAKE_OFFLINE_DIR="${FAKE_OFFLINE_DIR}")
target_link_libraries(mdml PUBLIC Eigen3::Eigen OpenSSL::Crypto Threads::Threads OpenMP::OpenMP_CXX)

foreach(prog lr_mdml lr_mdml_dealer lr_mdml_selftest)
  add_executable(${prog} src/${prog}.cpp)
  target_link_libraries(${prog} PRIVATE mdml)
endforeach()
