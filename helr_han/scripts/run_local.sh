#!/usr/bin/env bash
# Run both parties on this machine (server in the background, client in the foreground); the client prints the
# report (verification, per-phase tables, stage totals, SUMMARY).
#   scripts/run_local.sh --d 5                           # any lr_helr_han flags
#   scripts/run_local.sh --d 5 --iters 4 --logn 12       # tiny INSECURE ring, runs in seconds
#   THREADS=6 scripts/run_local.sh --d 100
# Both parties share this machine: THREADS defaults to half of the hardware threads each.
# Logs: results/server.log, results/client.log; one CSV row is appended to results/results.csv
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=build/lr_helr_han
PORT="${PORT:-$(( (RANDOM % 15000) + 20000 ))}"
mkdir -p results
TH="--threads ${THREADS:-$(( ($(nproc) + 1) / 2 ))}"
"$BIN" server --port "$PORT" $TH "$@" > results/server.log 2>&1 &
SPID=$!
"$BIN" client --port "$PORT" $TH --csv results/results.csv --json results/last_run.json "$@" > results/client.log 2>&1
CRC=$?
wait "$SPID"; SRC=$?
cat results/client.log
echo "(client exit $CRC, server exit $SRC; server log: results/server.log)"
exit $(( CRC | SRC ))
