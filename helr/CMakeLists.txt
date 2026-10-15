cmake_minimum_required(VERSION 3.16)
project(lr_helr LANGUAGES CXX)

# HELR baseline (Kim, Song, Wang, Xia, Jiang; JMIR Med Inform 2018): logistic-regression
# training on CKKS/HEAAN ciphertexts, run as a client/server protocol with a decryption MAC
# and a verified decryption. The HE library is HELR's own HEAAN snapshot
# (third_party/HELR/src, fetched and patched by scripts/fetch_deps.sh), built here from
# source on top of NTL + GMP.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Use an out-of-source build: cmake -S . -B build")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(HELR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/HELR/src)
if(NOT EXISTS ${HELR_SRC}/Scheme.cpp)
  message(FATAL_ERROR "third_party/HELR is missing: run scripts/fetch_deps.sh first")
endif()
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/HELR/.patched)
  message(FATAL_ERROR "third_party/HELR is not patched: run scripts/fetch_deps.sh")
endif()

find_package(OpenSSL 3.0 REQUIRED)   # SHA-256, RAND_bytes
find_package(Threads REQUIRED)
find_library(NTL_LIBRARY ntl REQUIRED)
find_library(GMP_LIBRARY gmp REQUIRED)
message(STATUS "OpenSSL ${OPENSSL_VERSION}, NTL: ${NTL_LIBRARY}, GMP: ${GMP_LIBRARY}")

# HELR's HEAAN snapshot (2017/09), unmodified apart from patches/heaan.patch.
# Only the library sources; HEAAN.cpp (a demo main) and TestScheme.cpp are not built.
set(HEAAN_SOURCES
    CZZ.cpp CZZX.cpp Cipher.cpp EvaluatorUtils.cpp Message.cpp NumUtils.cpp Params.cpp PubKey.cpp
    Ring2Utils.cpp Scheme.cpp SchemeAlgo.cpp SchemeAux.cpp SecKey.cpp StringUtils.cpp TimeUtils.cpp)
list(TRANSFORM HEAAN_SOURCES PREPEND ${HELR_SRC}/)
add_library(heaan STATIC ${HEAAN_SOURCES})
target_include_directories(heaan PUBLIC ${HELR_SRC})
target_compile_options(heaan PRIVATE -w -O2)   # third-party research code: no warnings
target_link_libraries(heaan PUBLIC ${NTL_LIBRARY} ${GMP_LIBRARY} Threads::Threads)

add_library(helr_common INTERFACE)
target_include_directories(helr_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(helr_common INTERFACE heaan OpenSSL::Crypto Threads::Threads)
target_compile_options(helr_common INTERFACE -Wall -Wextra -Wno-unused-parameter $<$<CONFIG:Release>:-O2>)

# The two-party program: lr_helr <client|server> [options].
add_executable(lr_helr src/lr_helr.cpp)
target_link_libraries(lr_helr PRIVATE helr_common)

# Offline tests (no TCP): polynomial fit, serialization, HE circuit against the plaintext
# reference, the full protocol in-process, and the malicious-behaviour cases.
add_executable(lr_helr_selftest src/lr_helr_selftest.cpp)
target_link_libraries(lr_helr_selftest PRIVATE helr_common)
