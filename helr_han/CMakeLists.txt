cmake_minimum_required(VERSION 3.16)
project(lr_helr_han LANGUAGES CXX)

# Han et al. (AAAI-19) logistic regression on HEAAN ciphertexts, as a client/server protocol.
# The HE library is the HEAAN snapshot with approximate bootstrapping (third_party/HEAAN, one small
# patch, see README), built from source on top of NTL + GMP.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Use an out-of-source build: cmake -S . -B build")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_library(NTL_LIBRARY ntl REQUIRED)
find_library(GMP_LIBRARY gmp REQUIRED)
find_package(OpenSSL 3.0 REQUIRED)   # SHA-256 for commitments, RAND_bytes for seeding
message(STATUS "NTL: ${NTL_LIBRARY}, GMP: ${GMP_LIBRARY}, OpenSSL ${OPENSSL_VERSION}")

set(HEAAN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/HEAAN/src)
file(GLOB HEAAN_SOURCES ${HEAAN_SRC}/*.cpp)
add_library(heaan STATIC ${HEAAN_SOURCES})
target_include_directories(heaan SYSTEM PUBLIC ${HEAAN_SRC})
target_compile_options(heaan PRIVATE -w -O3 -march=native)
target_link_libraries(heaan PUBLIC ${NTL_LIBRARY} ${GMP_LIBRARY} Threads::Threads)

add_library(han_common INTERFACE)
target_include_directories(han_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(han_common INTERFACE heaan OpenSSL::Crypto Threads::Threads)
target_compile_options(han_common INTERFACE -Wall -Wextra -Wno-unused-parameter)

add_executable(lr_helr_han src/lr_helr_han.cpp)
target_link_libraries(lr_helr_han PRIVATE han_common)

add_executable(lr_helr_han_selftest src/lr_helr_han_selftest.cpp)
target_link_libraries(lr_helr_han_selftest PRIVATE han_common)
