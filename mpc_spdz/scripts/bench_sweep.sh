#!/usr/bin/env bash
# Sweep the feature dimension d, in both security modes, one CSV row per run (written by the server).
#   D_LIST="5 50 500" MODES="malicious semi-honest" REPS=2 THREADS=1 OUT=results/sweep.csv scripts/bench_sweep.sh
# All 5 parties run on this machine. Each run is checked against the plaintext reference with --verify
# (VERIFY=0 to skip). Extra arguments are passed to every party (e.g. --iters 20 --batch 4).
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=build/lr_spdz
D_LIST="${D_LIST:-5 10 50 100 500 1000}"
MODES="${MODES:-malicious semi-honest}"
REPS="${REPS:-1}"
THREADS="${THREADS:-1}"
OUT="${OUT:-results/sweep.csv}"
V=""; [ "${VERIFY:-1}" = "1" ] && V="--verify"
B=4
args=("$@")
for ((i = 0; i < ${#args[@]}; ++i)); do [ "${args[$i]}" = "--batch" ] && B="${args[$((i + 1))]}"; done
mkdir -p "$(dirname "$OUT")" results
for d in $D_LIST; do
  for mode in $MODES; do
    flag=""; [ "$mode" = "semi-honest" ] && flag="--semi-honest"
    for rep in $(seq "$REPS"); do
      PORT=$(( (RANDOM % 9000) + 20000 ))
      PIDS=()
      for ((k = 1; k <= B; ++k)); do
        "$BIN" client --id "$k" --port "$PORT" --d "$d" --threads "$THREADS" $flag --quiet "$@" > "results/sweep_client$k.log" 2>&1 &
        PIDS+=($!)
      done
      "$BIN" server --port "$PORT" --d "$d" --threads "$THREADS" $flag $V --quiet --csv "$OUT" "$@" > results/sweep_server.log 2>&1
      SRC=$?
      CC=0; for p in "${PIDS[@]}"; do wait "$p"; c=$?; [ "$c" -gt "$CC" ] && CC=$c; done
      echo "d=$d mode=$mode rep=$rep: exit server $SRC, clients(max) $CC; $(grep -h 'verify:' results/sweep_server.log | cut -c1-70)"
    done
  done
done
echo "wrote $OUT"
