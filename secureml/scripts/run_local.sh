#!/usr/bin/env bash
# Run both parties on this machine (server in the background, client in the foreground) and print the
# server's report.
#   scripts/run_local.sh --d 5                      # any lr_secureml training/protocol flags
#   PORT=12345 VERIFY=0 scripts/run_local.sh --d 100 --semi-honest
# Logs: results/server.log, results/client.log; a CSV row is appended to results/results.csv
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=build/lr_secureml
PORT="${PORT:-$(( (RANDOM % 15000) + 20000 ))}"
mkdir -p results
V=""; [ "${VERIFY:-1}" = "1" ] && V="--verify"
# both parties share this machine: half of the hardware threads each (override with --threads N)
TH="--threads ${THREADS:-$(( ($(nproc) + 1) / 2 ))}"
"$BIN" server --port "$PORT" $TH $V --csv results/results.csv --json results/last_run.json "$@" > results/server.log 2>&1 &
SPID=$!
"$BIN" client --port "$PORT" $TH "$@" > results/client.log 2>&1
CRC=$?
wait "$SPID"; SRC=$?
cat results/server.log
echo "(client exit $CRC, server exit $SRC; client log: results/client.log)"
exit $(( CRC | SRC ))
