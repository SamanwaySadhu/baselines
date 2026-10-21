#!/usr/bin/env bash
# Train on the real datasets (iris, wine, titanic, breast_cancer) with both parties locally and
# print the accuracy of the trained weights.
#   ML_DIR=/mnt/c/ml_training PY=<python with pandas+seaborn> BUILD=~/gc_build scripts/run_real.sh [datasets...]
set -uo pipefail
cd "$(dirname "$0")/.."
ML_DIR="${ML_DIR:-/mnt/c/ml_training}"
PY="${PY:-$ML_DIR/training_env/bin/python}"
BUILD="${BUILD:-build}"
ITERS="${ITERS:-20}"
SETS=("$@"); [ ${#SETS[@]} -eq 0 ] && SETS=(iris wine titanic breast_cancer)
mkdir -p results
for ds in "${SETS[@]}"; do
  f="results/data_$ds.txt"
  "$PY" scripts/prep_dataset.py "$ds" "$ML_DIR" "$f" ${SEED:-42} || exit 1
  d=$(( $(head -1 "$f" | cut -d' ' -f2) + 1 ))
  port=$(( (RANDOM % 20000) + 20000 ))
  "$BUILD/lr_gc_ag2pc" server --port "$port" --d "$d" --iters "$ITERS" --eval "$f" --quiet > "results/real_${ds}_server.log" 2>&1 &
  spid=$!
  "$BUILD/lr_gc_ag2pc" client --port "$port" --d "$d" --iters "$ITERS" --data "$f" --quiet > "results/real_${ds}_client.log" 2>&1
  crc=$?; wait "$spid"; src=$?
  echo "== $ds (d=$d, client exit $crc, server exit $src)"
  grep -E "trained weights|--eval|ACCURACY" "results/real_${ds}_server.log"
done
