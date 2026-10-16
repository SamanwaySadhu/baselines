cmake_minimum_required(VERSION 3.16)
project(lr_gc_agmpc C CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Use an out-of-source build: cmake -S . -B build")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# emp-agmpc needs the OLD emp-tool / emp-ot API. scripts/install_emp_legacy.sh installs the pinned versions
# (emp-tool 80cb3b4, emp-ot ff28eb5) into a separate prefix (default $HOME/emp-legacy), so the emp-tool 1.0 used by
# gc_ag2pc / secureml / falcon is untouched.
set(EMP_LEGACY_PREFIX "$ENV{HOME}/emp-legacy" CACHE PATH "prefix of the legacy emp-tool / emp-ot install")
if(NOT EXISTS "${EMP_LEGACY_PREFIX}/include/emp-tool/emp-tool.h" OR NOT EXISTS "${EMP_LEGACY_PREFIX}/include/emp-ot/emp-ot.h")
  message(FATAL_ERROR "legacy emp-tool / emp-ot not found in ${EMP_LEGACY_PREFIX}; run scripts/install_emp_legacy.sh")
endif()
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)
find_library(EMP_TOOL_LIB emp-tool PATHS "${EMP_LEGACY_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)

# The hardened emp-agmpc copy (see third_party/emp-agmpc/PATCHES.md). Refuse an unpatched copy.
set(AGMPC_DIR "${CMAKE_SOURCE_DIR}/third_party/emp-agmpc")
file(READ "${AGMPC_DIR}/emp-agmpc/cmpc_config.h" _cfg)
if(NOT _cfg MATCHES "EMP_AGMPC_HARDENED")
  message(FATAL_ERROR "${AGMPC_DIR} is not the hardened emp-agmpc copy")
endif()

# Party counts (clients + 1) compiled in; each one is its own translation unit.
set(AGMPC_NP_LIST "2;3;5" CACHE STRING "emp-agmpc party counts (batch + 1) to build")

add_library(agmpc_common INTERFACE)
target_include_directories(agmpc_common INTERFACE ${CMAKE_SOURCE_DIR}/include ${AGMPC_DIR})
target_include_directories(agmpc_common SYSTEM INTERFACE ${EMP_LEGACY_PREFIX}/include)
target_compile_features(agmpc_common INTERFACE cxx_std_20)
# legacy emp headers predate some libstdc++ hygiene; -march=native for AES-NI / SSE as emp-base.cmake does
target_compile_options(agmpc_common INTERFACE "SHELL:-include cstdint" "SHELL:-include memory" -march=native -maes -mrdseed
                       -Wno-deprecated-declarations -Wno-ignored-attributes)
target_compile_definitions(agmpc_common INTERFACE EMP_CIRCUIT_PATH=${EMP_LEGACY_PREFIX}/include/emp-tool/circuits/files/)
target_link_libraries(agmpc_common INTERFACE ${EMP_TOOL_LIB} OpenSSL::Crypto OpenSSL::SSL Threads::Threads)

set(_np_objs "")
set(_np_defs "")
foreach(np IN LISTS AGMPC_NP_LIST)
  add_library(runner_np${np} OBJECT src/runner_np.cpp)
  target_compile_definitions(runner_np${np} PRIVATE NP=${np})
  target_link_libraries(runner_np${np} PRIVATE agmpc_common)
  list(APPEND _np_objs $<TARGET_OBJECTS:runner_np${np}>)
  list(APPEND _np_defs AGMPC_HAVE_NP${np}=1)
endforeach()
string(REPLACE ";" "," _np_str "${AGMPC_NP_LIST}")

add_executable(lr_gc_agmpc src/lr_gc_agmpc.cpp ${_np_objs})
target_link_libraries(lr_gc_agmpc PRIVATE agmpc_common)
target_compile_definitions(lr_gc_agmpc PRIVATE ${_np_defs} AGMPC_NP_LIST="parties ${_np_str}")
target_compile_options(lr_gc_agmpc PRIVATE -Wall -Wextra)

# Circuit / segmentation / hash checks, no networking.
add_executable(lr_gc_agmpc_selftest src/lr_gc_agmpc_selftest.cpp)
target_link_libraries(lr_gc_agmpc_selftest PRIVATE agmpc_common)
target_compile_options(lr_gc_agmpc_selftest PRIVATE -Wall -Wextra)
