cmake_minimum_required(VERSION 3.21)
project(lr_gc_ag2pc LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Use an out-of-source build: cmake -S . -B build")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# emp-tool and emp-ot install CMake packages (pass -DCMAKE_PREFIX_PATH=<prefix>
# if they are not under /usr/local). Their targets carry C++20, -march=native,
# OpenSSL and pthreads.
find_package(emp-tool 1.0 CONFIG REQUIRED)
find_package(emp-ot 1.0 CONFIG REQUIRED)

# emp-ag2pc is header-only. This project builds against its OWN copy in
# third_party/emp-ag2pc: upstream a245ca0 plus a fix to the input phase
# (third_party/emp-ag2pc/input-privacy.patch), without which every private input is
# sent to the other party in the clear. The copy is put first on the include path so
# it shadows any emp-ag2pc headers installed next to emp-tool.
set(EMP_AG2PC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/emp-ag2pc" CACHE PATH
    "directory containing emp-ag2pc/emp-ag2pc.h (default: the patched copy in third_party)")
if(NOT EXISTS "${EMP_AG2PC_DIR}/emp-ag2pc/emp-ag2pc.h")
  message(FATAL_ERROR "emp-ag2pc headers not found in ${EMP_AG2PC_DIR}")
endif()
set(EMP_AG2PC_INCLUDE_DIR "${EMP_AG2PC_DIR}")
message(STATUS "emp-ag2pc headers: ${EMP_AG2PC_INCLUDE_DIR}")

add_library(emp_ag2pc_headers INTERFACE)
target_include_directories(emp_ag2pc_headers BEFORE INTERFACE ${EMP_AG2PC_INCLUDE_DIR})
target_link_libraries(emp_ag2pc_headers INTERFACE emp-ot::emp-ot emp-tool::emp-tool)
target_compile_features(emp_ag2pc_headers INTERFACE cxx_std_20)

# The two-party program: party 1 = client (garbler), party 2 = server (evaluator).
add_executable(lr_gc_ag2pc src/lr_gc_ag2pc.cpp)
target_include_directories(lr_gc_ag2pc PRIVATE include)
target_link_libraries(lr_gc_ag2pc PRIVATE emp_ag2pc_headers)

# Offline correctness tests + gate counts (no networking).
add_executable(lr_gc_selftest src/lr_gc_selftest.cpp)
target_include_directories(lr_gc_selftest PRIVATE include)
target_link_libraries(lr_gc_selftest PRIVATE emp-tool::emp-tool)
target_compile_features(lr_gc_selftest PRIVATE cxx_std_20)

# Small test circuit: one dot product + one sigmoid (offline over ClearCtx, or over AG2PC).
add_executable(lr_gc_dot_sigmoid src/lr_gc_dot_sigmoid.cpp)
target_include_directories(lr_gc_dot_sigmoid PRIVATE include)
target_link_libraries(lr_gc_dot_sigmoid PRIVATE emp_ag2pc_headers)

# Test helper for scripts/check_input_privacy.sh (prints the client's input bits).
add_executable(lr_gc_client_inputs src/lr_gc_client_inputs.cpp)
target_include_directories(lr_gc_client_inputs PRIVATE include)
target_compile_features(lr_gc_client_inputs PRIVATE cxx_std_20)
