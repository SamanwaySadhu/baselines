#!/usr/bin/env bash
# Configure + build lr_mdml, lr_mdml_dealer and lr_mdml_selftest into ./build.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f third_party/MD-ML/.patched ] || bash scripts/fetch_deps.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"
echo "built: build/lr_mdml build/lr_mdml_dealer build/lr_mdml_selftest"
