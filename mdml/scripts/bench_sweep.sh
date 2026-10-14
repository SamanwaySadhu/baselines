#!/usr/bin/env bash
# Sweep the dimension d (REPS runs each) on localhost; one CSV row per run.
#   D_LIST="5 10 50 100" REPS=3 scripts/bench_sweep.sh [extra training flags]
set -uo pipefail
cd "$(dirname "$0")/.."
D_LIST="${D_LIST:-5 10 20 50 100}"
REPS="${REPS:-3}"
OUT="${OUT:-results/sweep.csv}"
OFF="${OFFLINE_DIR:-results/offline}"
mkdir -p results
for d in $D_LIST; do
  for r in $(seq 1 "$REPS"); do
    PORT=$(( 30000 + RANDOM % 20000 ))
    JOB="sweep_d${d}_r${r}_$$"
    build/lr_mdml_dealer --offline-dir "$OFF" --job "$JOB" --d "$d" "$@" > results/sweep_dealer.log 2>&1 \
      || { echo "dealer failed for d=$d"; cat results/sweep_dealer.log; continue; }
    build/lr_mdml server ${THREADS:+--threads $THREADS} --port $PORT --offline-dir "$OFF" --job "$JOB" --quiet --d "$d" --csv "$OUT" "$@" \
      > results/sweep_server.log 2>&1 &
    SPID=$!
    build/lr_mdml client ${THREADS:+--threads $THREADS} --port $PORT --offline-dir "$OFF" --job "$JOB" --quiet --d "$d" "$@" > results/sweep_client.log 2>&1
    wait $SPID || echo "run d=$d rep=$r failed (see results/sweep_server.log)"
    rm -f "$OFF/$JOB"-party-0.txt "$OFF/$JOB"-party-1.txt "$OFF/$JOB"-verify.txt
    echo "d=$d rep=$r: $(grep -m1 'TOTAL online' results/sweep_server.log | awk -F'|' '{print "online server ms:" $3}') $(grep -m1 'communication:' results/sweep_server.log)"
  done
done
echo "CSV: $OUT"
