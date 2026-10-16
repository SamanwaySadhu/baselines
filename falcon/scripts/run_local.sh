#!/usr/bin/env bash
# Run the three servers (in the background) and the client (in the foreground) on this machine and print
# the client's report.
#   scripts/run_local.sh --d 5                      # any lr_falcon training/protocol flags
#   PORT=12345 VERIFY=0 scripts/run_local.sh --d 1000 --semi-honest
# Logs: results/client.log, results/server{0,1,2}.log; a CSV row is appended to results/results.csv and
# all metrics of the run go to results/last_run.json. Exit code: the client's if nonzero, else the largest
# of the servers'.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=build/lr_falcon
PORT="${PORT:-$(( (RANDOM % 12000) + 20000 ))}"
RUN_ID="${RUN_ID:-$(( (RANDOM << 15) | RANDOM ))}"
TIMEOUT="${TIMEOUT:-60}"
mkdir -p results
V=""; [ "${VERIFY:-1}" = "1" ] && V="--verify"
COMMON=(--port "$PORT" --run-id "$RUN_ID" --timeout "$TIMEOUT" --connect-timeout "$TIMEOUT")
PIDS=()
for i in 0 1 2; do
  "$BIN" server --id "$i" "${COMMON[@]}" "$@" > "results/server$i.log" 2>&1 &
  PIDS+=($!)
done
"$BIN" client "${COMMON[@]}" $V --csv results/results.csv --json results/last_run.json "$@" > results/client.log 2>&1
CRC=$?
SRC=()
for p in "${PIDS[@]}"; do wait "$p"; SRC+=($?); done
cat results/client.log
echo "(exit codes: client $CRC, S0 ${SRC[0]}, S1 ${SRC[1]}, S2 ${SRC[2]}; logs: results/client.log, results/server{0,1,2}.log)"
rc=$CRC
if [ "$rc" = 0 ]; then for s in "${SRC[@]}"; do [ "$s" -gt "$rc" ] && rc=$s; done; fi
exit $rc
