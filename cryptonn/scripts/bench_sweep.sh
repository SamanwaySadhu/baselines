#!/usr/bin/env bash
# Sweep the dimension d (REPS runs each) on localhost; one CSV row per run.
#   D_LIST="5 10 50 100" REPS=3 OUT=results/sweep.csv scripts/bench_sweep.sh [extra training flags]
# THREADS=N passes --threads N to both parties. VERIFY=1 (default) makes the server check
# every run bit for bit against the plaintext fixed-point reference (the 'verified' CSV
# column and a PASS/FAIL per run); VERIFY=0 skips it. BUILD_DIR overrides ./build.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN="${BUILD_DIR:-build}/lr_cryptonn"
[ -x "$BIN" ] || { echo "$BIN not found; run scripts/build.sh first"; exit 1; }
D_LIST="${D_LIST:-5 10 50 100}"
REPS="${REPS:-3}"
OUT="${OUT:-results/sweep.csv}"
T=""; [ -n "${THREADS:-}" ] && T="--threads $THREADS"
V=""; [ "${VERIFY:-1}" = "1" ] && V="--verify"
mkdir -p results "$(dirname "$OUT")"
FAILS=0
for d in $D_LIST; do
  for r in $(seq 1 "$REPS"); do
    PORT=$(( 30000 + RANDOM % 20000 ))
    "$BIN" server --port $PORT --quiet $T $V --d "$d" --csv "$OUT" "$@" > results/sweep_server.log 2>&1 &
    SPID=$!
    "$BIN" client --host 127.0.0.1 --port $PORT --quiet $T --d "$d" "$@" > results/sweep_client.log 2>&1
    CRC=$?
    wait $SPID; SRC=$?
    if [ $CRC -ne 0 ] || [ $SRC -ne 0 ]; then
      FAILS=$((FAILS + 1))
      echo "run d=$d rep=$r FAILED (client exit $CRC, server exit $SRC; a failed --verify also exits non-zero; see results/sweep_{server,client}.log)"
    fi
    echo "d=$d rep=$r: $(grep -m1 -o -E 'verify: (PASS|FAIL)' results/sweep_server.log || echo 'verify: skipped') | $(grep -m1 -i 'communication' results/sweep_server.log)"
  done
done
echo "CSV: $OUT ($FAILS failed runs)"
exit $(( FAILS > 0 ))
