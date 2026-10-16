#!/usr/bin/env bash
# Configure + build lr_falcon and lr_falcon_selftest into ./build.
#   EMP_PREFIX=$HOME/emp scripts/build.sh   (default prefix: $HOME/emp; install EMP with ../gc_ag2pc/scripts/install_emp.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
EMP_PREFIX="${EMP_PREFIX:-$HOME/emp}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$EMP_PREFIX"
cmake --build build -j "$(nproc)"
echo "built: build/lr_falcon build/lr_falcon_selftest"
