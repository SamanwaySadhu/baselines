#!/usr/bin/env bash
# Run the server and the B clients (B = --batch, default 4) as processes on this machine and print the server's
# report.
#   scripts/run_local.sh --d 5                      # any lr_spdz training/protocol flags
#   PORT=23456 VERIFY=0 scripts/run_local.sh --d 100 --semi-honest
# Logs: results/server.log, results/client{1..B}.log; a CSV row is appended to results/results.csv and all metrics
# go to results/last_run.json. Exit code: the server's if nonzero, else the largest of the clients'.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=build/lr_spdz
PORT="${PORT:-$(( (RANDOM % 9000) + 20000 ))}"
mkdir -p results
B=4
args=("$@")
for ((i = 0; i < ${#args[@]}; ++i)); do [ "${args[$i]}" = "--batch" ] && B="${args[$((i + 1))]}"; done
V=""; [ "${VERIFY:-1}" = "1" ] && V="--verify"
PIDS=()
for ((k = 1; k <= B; ++k)); do
  "$BIN" client --id "$k" --port "$PORT" "$@" > "results/client$k.log" 2>&1 &
  PIDS+=($!)
done
"$BIN" server --port "$PORT" $V --csv results/results.csv --json results/last_run.json "$@" > results/server.log 2>&1
SRC=$?
CODES=()
for p in "${PIDS[@]}"; do wait "$p"; CODES+=($?); done
cat results/server.log
echo "(exit codes: server $SRC, clients ${CODES[*]}; logs: results/server.log, results/client{1..$B}.log)"
rc=$SRC
if [ "$rc" = 0 ]; then for c in "${CODES[@]}"; do [ "$c" -gt "$rc" ] && rc=$c; done; fi
exit $rc
