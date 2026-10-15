#!/usr/bin/env bash
# Sweep the feature dimension d, in both security modes, one CSV row per run (written by the server).
#   D_LIST="5 50 500" MODES="malicious semi-honest" REPS=2 THREADS=6 OUT=results/sweep.csv scripts/bench_sweep.sh
# Both parties run on this machine, so THREADS defaults to half of the hardware threads. Each run is
# checked against the plaintext reference with --verify (VERIFY=0 to skip).
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=build/lr_secureml
D_LIST="${D_LIST:-5 10 50 100 500 1000}"
MODES="${MODES:-malicious semi-honest}"
REPS="${REPS:-1}"
THREADS="${THREADS:-$(( ($(nproc) + 1) / 2 ))}"
OUT="${OUT:-results/sweep.csv}"
V=""; [ "${VERIFY:-1}" = "1" ] && V="--verify"
mkdir -p "$(dirname "$OUT")" results
for d in $D_LIST; do
  for mode in $MODES; do
    flag=""; [ "$mode" = "semi-honest" ] && flag="--semi-honest"
    for rep in $(seq "$REPS"); do
      PORT=$(( (RANDOM % 15000) + 20000 ))
      "$BIN" server --port "$PORT" --d "$d" --threads "$THREADS" $flag $V --quiet --csv "$OUT" "$@" > results/sweep_server.log 2>&1 &
      SPID=$!
      "$BIN" client --port "$PORT" --d "$d" --threads "$THREADS" $flag --quiet "$@" > results/sweep_client.log 2>&1
      CRC=$?
      wait "$SPID"; SRC=$?
      echo "d=$d mode=$mode rep=$rep: exit client $CRC server $SRC; $(grep -h 'verify:' results/sweep_server.log | cut -c1-70)"
    done
  done
done
echo "wrote $OUT"
