cmake_minimum_required(VERSION 3.21)
project(lr_secureml LANGUAGES CXX)

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

add_library(sml_common INTERFACE)
target_include_directories(sml_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../wan)
target_link_libraries(sml_common INTERFACE emp-ot::emp-ot emp-tool::emp-tool ${CMAKE_DL_LIBS})
target_compile_features(sml_common INTERFACE cxx_std_20)
target_compile_options(sml_common INTERFACE -Wall -Wextra)

add_executable(lr_secureml src/lr_secureml.cpp)
target_link_libraries(lr_secureml PRIVATE sml_common)

add_executable(lr_secureml_selftest src/lr_secureml_selftest.cpp)
target_link_libraries(lr_secureml_selftest PRIVATE sml_common)
