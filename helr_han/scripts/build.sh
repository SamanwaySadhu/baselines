#!/usr/bin/env bash
# Configure + build into ./build.  Needs: g++ (C++17), cmake >= 3.16, libntl-dev, libgmp-dev, libssl-dev
#   sudo apt install -y build-essential cmake libntl-dev libgmp-dev libssl-dev
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"
echo "built: build/lr_helr_han build/lr_helr_han_selftest"
