#!/usr/bin/env bash
# Configure + build lr_helr and lr_helr_selftest into ./build (or $BUILD_DIR).
# Needs a C++17 compiler, CMake >= 3.16, NTL, GMP and OpenSSL 3 headers, and the HELR sources
# (scripts/fetch_deps.sh):
#   sudo apt install -y build-essential cmake git libntl-dev libgmp-dev libssl-dev
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR="${BUILD_DIR:-build}"
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j "$(nproc)"
echo "built: $BUILD_DIR/lr_helr $BUILD_DIR/lr_helr_selftest"
