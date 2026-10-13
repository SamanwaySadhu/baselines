#!/usr/bin/env bash
# Sweep the dimension d (REPS runs each) on localhost; one CSV row per run.
#   D_LIST="5 10 50 100" REPS=3 scripts/bench_sweep.sh [extra flags]
set -uo pipefail
cd "$(dirname "$0")/.."
D_LIST="${D_LIST:-5 10 20 50 100}"
REPS="${REPS:-3}"
OUT="${OUT:-results/sweep.csv}"
mkdir -p results
for d in $D_LIST; do
  for r in $(seq 1 "$REPS"); do
    PORT=$(( 30000 + RANDOM % 20000 ))
    build/lr_gc_ag2pc server --port $PORT --quiet --d "$d" --csv "$OUT" "$@" > results/sweep_server.log 2>&1 &
    SPID=$!
    build/lr_gc_ag2pc client --port $PORT --quiet --d "$d" "$@" > results/sweep_client.log 2>&1
    wait $SPID || echo "run d=$d rep=$r failed (see results/sweep_server.log)"
    echo "d=$d rep=$r: $(grep -m1 'communication:' results/sweep_server.log)"
  done
done
echo "CSV: $OUT"
