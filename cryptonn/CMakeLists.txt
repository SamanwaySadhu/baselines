cmake_minimum_required(VERSION 3.16)
project(lr_cryptonn LANGUAGES CXX)

# CryptoNN baseline (Xu, Joshi, Li, ICDCS 2019): logistic-regression training where
# the server holds the model in the clear and the data-touching steps use functional
# encryption -- FEIP (Abdalla et al., PKC 2015) and the paper's FEBO -- over NIST P-256
# (OpenSSL 3 libcrypto). Header-only library in include/; no third-party sources.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Use an out-of-source build: cmake -S . -B build")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenSSL 3.0 REQUIRED)  # libcrypto: EC_GROUP / EC_POINT / BIGNUM, RAND_bytes
find_package(Threads REQUIRED)      # --threads (parallel encryption / decryption)
message(STATUS "OpenSSL ${OPENSSL_VERSION}: ${OPENSSL_CRYPTO_LIBRARY}")

# Common settings for every target: headers, libcrypto, pthreads, -O3 -march=native.
add_library(cryptonn_common INTERFACE)
target_include_directories(cryptonn_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cryptonn_common INTERFACE OpenSSL::Crypto Threads::Threads)
target_compile_options(cryptonn_common INTERFACE -Wall -Wextra
                       $<$<CONFIG:Release>:-O3 -march=native>)

# The two-party program: lr_cryptonn <client|server> [options].
add_executable(lr_cryptonn src/lr_cryptonn.cpp)
target_link_libraries(lr_cryptonn PRIVATE cryptonn_common)

# Offline tests of the group wrapper, BSGS, FEIP, FEBO and the fixed-point protocol
# arithmetic (no networking).
add_executable(lr_cryptonn_selftest src/lr_cryptonn_selftest.cpp)
target_link_libraries(lr_cryptonn_selftest PRIVATE cryptonn_common)
