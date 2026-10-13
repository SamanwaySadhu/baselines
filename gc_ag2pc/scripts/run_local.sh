#!/usr/bin/env bash
# Run both parties on this machine (server in the background, client in the
# foreground) and print the server's report.
#   scripts/run_local.sh --d 5                 # any lr_gc_ag2pc training/protocol flags
#   PORT=12345 VERIFY=0 scripts/run_local.sh --d 100
# Logs: results/server.log, results/client.log; CSV row appended to results/results.csv
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=build/lr_gc_ag2pc
PORT="${PORT:-$(( (RANDOM % 20000) + 20000 ))}"
mkdir -p results
V=""; [ "${VERIFY:-1}" = "1" ] && V="--verify"
"$BIN" server --port "$PORT" $V --csv results/results.csv --json results/last_run.json "$@" > results/server.log 2>&1 &
SPID=$!
"$BIN" client --port "$PORT" "$@" > results/client.log 2>&1
CRC=$?
wait "$SPID"; SRC=$?
cat results/server.log
echo "(client exit $CRC, server exit $SRC; client log: results/client.log)"
exit $(( CRC | SRC ))
