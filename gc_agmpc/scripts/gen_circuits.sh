#!/usr/bin/env bash
# Generate the float32 iteration circuits (needs emp-tool 1.0, the one gc_ag2pc uses) into ./circuits:
#   scripts/gen_circuits.sh 5 12 14 31        # dimensions (default: 5 12 14 31); CHECK=1 compares with reference_train
set -euo pipefail
cd "$(dirname "$0")/.."
EMP1="${EMP1_PREFIX:-$HOME/emp}"
cmake -S gen -B gen/build -DCMAKE_PREFIX_PATH="$EMP1" >/dev/null
cmake --build gen/build -j4 >/dev/null
DS=("$@"); [ ${#DS[@]} -eq 0 ] && DS=(5 12 14 31)
for d in "${DS[@]}"; do
  ./gen/build/gen_circuit --d "$d" --out circuits ${CHECK:+--check}
done
