#!/usr/bin/env bash
# Configure + build lr_cryptonn and lr_cryptonn_selftest into ./build (or $BUILD_DIR).
# Needs a C++20 compiler, CMake >= 3.16 and OpenSSL 3 headers:
#   sudo apt install -y build-essential cmake libssl-dev
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR="${BUILD_DIR:-build}"
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j "$(nproc)"
echo "built: $BUILD_DIR/lr_cryptonn $BUILD_DIR/lr_cryptonn_selftest"
