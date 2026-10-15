#!/usr/bin/env bash
# Sweep the feature dimension d (one CSV row per run, written by the client).
#   D_LIST="5 16 64" REPS=1 THREADS=6 OUT=results/sweep.csv scripts/bench_sweep.sh [extra flags]
# Both parties run on this machine, so THREADS defaults to half of the hardware threads each. Memory: each party
# holds the evaluation keys (about 1.4 GB at the default parameters) plus the encrypted data.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=build/lr_helr_han
D_LIST="${D_LIST:-5 16 64}"
REPS="${REPS:-1}"
THREADS="${THREADS:-$(( ($(nproc) + 1) / 2 ))}"
OUT="${OUT:-results/sweep.csv}"
mkdir -p "$(dirname "$OUT")" results
for d in $D_LIST; do
  for rep in $(seq "$REPS"); do
    PORT=$(( (RANDOM % 15000) + 20000 ))
    "$BIN" server --port "$PORT" --d "$d" --threads "$THREADS" --quiet "$@" > results/sweep_server.log 2>&1 &
    SPID=$!
    "$BIN" client --port "$PORT" --d "$d" --threads "$THREADS" --quiet --csv "$OUT" "$@" > results/sweep_client.log 2>&1
    CRC=$?
    wait "$SPID"; SRC=$?
    echo "d=$d rep=$rep: exit client $CRC server $SRC; $(grep -h 'verify:' results/sweep_client.log | cut -c1-110)"
  done
done
echo "wrote $OUT"
