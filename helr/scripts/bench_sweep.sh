#!/usr/bin/env bash
# Sweep the dimension d (REPS runs each) on localhost; one CSV row per run.
#   D_LIST="2 3 5" REPS=1 OUT=results/sweep.csv scripts/bench_sweep.sh [extra training / HE flags]
# THREADS=N passes --threads N to both parties. BASE_PORT=P uses ports P, P+1, ... (one per
# run; default: a random port per run). VERIFY=1 (default) makes the server compare the
# trained weights with the plaintext polynomial training (the 'dev_poly' / 'verified' CSV
# columns); VERIFY=0 skips it. BUILD_DIR overrides ./build.
# A run at the default parameters (N = 2^17, 20 iterations) takes minutes to hours per d and a few GB
# of RAM per party; see the README for the cost model and for what fits on which machine.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN="${BUILD_DIR:-build}/lr_helr"
[ -x "$BIN" ] || { echo "$BIN not found; run scripts/build.sh first"; exit 1; }
D_LIST="${D_LIST:-2 3 5}"
REPS="${REPS:-1}"
OUT="${OUT:-results/sweep.csv}"
T=""; [ -n "${THREADS:-}" ] && T="--threads $THREADS"
V=""; [ "${VERIFY:-1}" = "1" ] && V="--verify"
mkdir -p results "$(dirname "$OUT")"
FAILS=0
RUN=0
for d in $D_LIST; do
  for r in $(seq 1 "$REPS"); do
    if [ -n "${BASE_PORT:-}" ]; then PORT=$(( BASE_PORT + RUN )); else PORT=$(( 30000 + RANDOM % 20000 )); fi
    RUN=$((RUN + 1))
    "$BIN" server --port $PORT --quiet $T $V --d "$d" --csv "$OUT" "$@" > results/sweep_server.log 2>&1 &
    SPID=$!
    "$BIN" client --host 127.0.0.1 --port $PORT --quiet $T --d "$d" "$@" > results/sweep_client.log 2>&1
    CRC=$?
    wait $SPID; SRC=$?
    if [ $CRC -ne 0 ] || [ $SRC -ne 0 ]; then
      FAILS=$((FAILS + 1))
      echo "run d=$d rep=$r FAILED (client exit $CRC, server exit $SRC; see results/sweep_{server,client}.log)"
    fi
    echo "d=$d rep=$r: $(grep -m1 -o -E 'verify: (PASS|FAIL)' results/sweep_server.log || echo 'verify: skipped') | $(grep -m1 -i 'communication' results/sweep_server.log)"
  done
done
echo "CSV: $OUT ($FAILS failed runs)"
exit $(( FAILS > 0 ))
