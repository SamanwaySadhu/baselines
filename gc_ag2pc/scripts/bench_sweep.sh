#!/usr/bin/env bash
# Sweep the dimension d (REPS runs each) on localhost; one CSV row per run.
#   D_LIST="5 10 50 100" REPS=3 scripts/bench_sweep.sh [extra flags]    # -> results/sweep_v2.csv
# VERIFY=1 (default) makes the server check every run bit for bit against the plaintext
# float reference (the 'verified' CSV column and a PASS/FAIL per run); VERIFY=0 skips it.
set -uo pipefail
cd "$(dirname "$0")/.."
D_LIST="${D_LIST:-5 10 20 50 100}"
REPS="${REPS:-3}"
OUT="${OUT:-results/sweep_v2.csv}"  # new column layout; older results/sweep.csv runs used the leaky library
V=""; [ "${VERIFY:-1}" = "1" ] && V="--verify"
mkdir -p results
for d in $D_LIST; do
  for r in $(seq 1 "$REPS"); do
    PORT=$(( 30000 + RANDOM % 20000 ))
    build/lr_gc_ag2pc server --port $PORT --quiet $V --d "$d" --csv "$OUT" "$@" > results/sweep_server.log 2>&1 &
    SPID=$!
    build/lr_gc_ag2pc client --port $PORT --quiet --d "$d" "$@" > results/sweep_client.log 2>&1
    wait $SPID || echo "run d=$d rep=$r FAILED (a failed --verify also exits non-zero; see results/sweep_server.log)"
    echo "d=$d rep=$r: $(grep -m1 -o -E 'verify: (PASS|FAIL)' results/sweep_server.log || echo 'verify: skipped') | $(grep -m1 'COMMUNICATION' results/sweep_server.log)"
  done
done
echo "CSV: $OUT"
