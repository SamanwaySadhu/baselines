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

# emp-ag2pc is header-only; at the pinned commit it installs no CMake package, so
# scripts/install_emp.sh copies its headers to <prefix>/include/emp-ag2pc. A source
# checkout can be used instead with -DEMP_AG2PC_DIR=/path/to/emp-ag2pc.
set(EMP_AG2PC_DIR "" CACHE PATH "emp-ag2pc checkout or include dir (contains emp-ag2pc/emp-ag2pc.h)")
find_path(EMP_AG2PC_INCLUDE_DIR emp-ag2pc/emp-ag2pc.h
  HINTS ${EMP_AG2PC_DIR} ${EMP_AG2PC_DIR}/include
  PATH_SUFFIXES include
  NO_CACHE)
if(NOT EMP_AG2PC_INCLUDE_DIR)
  message(FATAL_ERROR "emp-ag2pc headers not found. Run scripts/install_emp.sh, or pass "
                      "-DEMP_AG2PC_DIR=/path/to/emp-ag2pc (the git checkout).")
endif()
message(STATUS "emp-ag2pc headers: ${EMP_AG2PC_INCLUDE_DIR}")

add_library(emp_ag2pc_headers INTERFACE)
target_include_directories(emp_ag2pc_headers INTERFACE ${EMP_AG2PC_INCLUDE_DIR})
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
