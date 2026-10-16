#!/usr/bin/env bash
# Sweep the feature dimension d in both security modes, one CSV row per run (written by the client).
#   D_LIST="5 50 500" MODES="malicious semi-honest" REPS=2 OUT=results/sweep.csv scripts/bench_sweep.sh
# All four parties run on this machine. Each run is checked against the bit-exact plaintext reference with
# --verify (VERIFY=0 to skip).
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=build/lr_falcon
D_LIST="${D_LIST:-5 10 50 100 500 1000}"
MODES="${MODES:-malicious semi-honest}"
REPS="${REPS:-1}"
OUT="${OUT:-results/sweep.csv}"
V=""; [ "${VERIFY:-1}" = "1" ] && V="--verify"
mkdir -p "$(dirname "$OUT")" results
for d in $D_LIST; do
  for mode in $MODES; do
    flag=""; [ "$mode" = "semi-honest" ] && flag="--semi-honest"
    for rep in $(seq "$REPS"); do
      PORT=$(( (RANDOM % 12000) + 20000 )); RUN_ID=$(( (RANDOM << 15) | RANDOM ))
      COMMON=(--port "$PORT" --run-id "$RUN_ID" --timeout 60 --connect-timeout 60 --quiet --d "$d" $flag)
      PIDS=()
      for i in 0 1 2; do
        "$BIN" server --id "$i" "${COMMON[@]}" "$@" > "results/sweep_server$i.log" 2>&1 &
        PIDS+=($!)
      done
      "$BIN" client "${COMMON[@]}" $V --csv "$OUT" "$@" > results/sweep_client.log 2>&1
      CRC=$?
      SRC=""
      for p in "${PIDS[@]}"; do wait "$p"; SRC="$SRC $?"; done
      echo "d=$d mode=$mode rep=$rep: exit client $CRC servers$SRC; $(grep -h 'verify:' results/sweep_client.log | cut -c1-70)"
    done
  done
done
echo "wrote $OUT"
