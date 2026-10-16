#!/usr/bin/env bash
# Configure + build lr_gc_agmpc and lr_gc_agmpc_selftest into ./build.
#   EMP_LEGACY_PREFIX=$HOME/emp-legacy NP_LIST="2;3;5" scripts/build.sh
# (install the legacy emp first with scripts/install_emp_legacy.sh). Each party count (batch + 1) in NP_LIST is
# its own translation unit; the build is memory-hungry, so -j is limited.
set -euo pipefail
cd "$(dirname "$0")/.."
EMP_LEGACY_PREFIX="${EMP_LEGACY_PREFIX:-$HOME/emp-legacy}"
NP_LIST="${NP_LIST:-2;3;5}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DEMP_LEGACY_PREFIX="$EMP_LEGACY_PREFIX" -DAGMPC_NP_LIST="$NP_LIST"
cmake --build build -j "${JOBS:-3}"
echo "built: build/lr_gc_agmpc build/lr_gc_agmpc_selftest (party counts $NP_LIST)"
