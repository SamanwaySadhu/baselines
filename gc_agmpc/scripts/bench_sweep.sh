#!/usr/bin/env bash
# Sweep the feature dimension d, one CSV row per run (written by the server). All parties run on this machine.
#   D_LIST="5 10 50" REPS=2 BATCH=4 OUT=results/sweep.csv scripts/bench_sweep.sh [extra lr_gc_agmpc flags]
# Each run is checked against the plaintext reference with --verify (VERIFY=0 to skip).
set -uo pipefail
cd "$(dirname "$0")/.."
D_LIST="${D_LIST:-5 10 50 100}"
REPS="${REPS:-1}"
BATCH="${BATCH:-4}"
OUT="${OUT:-results/sweep.csv}"
BIN=build/lr_gc_agmpc
V=""; [ "${VERIFY:-1}" = "1" ] && V="--verify"
mkdir -p "$(dirname "$OUT")" results
for d in $D_LIST; do
  for rep in $(seq "$REPS"); do
    PORT=$(( (RANDOM % 9000) + 20000 ))
    COMMON=(--port "$PORT" --timeout 600 --quiet --d "$d" --batch "$BATCH")
    PIDS=()
    for ((k = 1; k <= BATCH; ++k)); do
      "$BIN" client --id "$k" "${COMMON[@]}" "$@" > "results/sweep_client$k.log" 2>&1 &
      PIDS+=($!)
    done
    "$BIN" server "${COMMON[@]}" $V --csv "$OUT" "$@" > results/sweep_server.log 2>&1
    SRC=$?
    CODES=""
    for p in "${PIDS[@]}"; do wait "$p"; CODES="$CODES $?"; done
    echo "d=$d rep=$rep: exit server $SRC clients$CODES; $(grep -h 'verify:' results/sweep_server.log | cut -c1-60)"
  done
done
echo "wrote $OUT"
