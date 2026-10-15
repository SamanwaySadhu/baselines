#!/usr/bin/env bash
# One training run on this machine: the server in the background, the client in the
# foreground; prints the server's report.
#   scripts/run_local.sh --d 3 --iters 20                 # any lr_helr flags (both parties)
#   PORT=12345 THREADS=4 VERIFY=0 scripts/run_local.sh --d 5
# VERIFY=1 (default): the server checks the trained weights against the plaintext polynomial
# training (test only: it regenerates the client's data from --seed).
# Logs: results/server.log, results/client.log; JSON: results/last_run.json; a CSV row is
# appended to results/results.csv. BUILD_DIR overrides ./build.
# NOTE: at the default parameters (N = 2^17) a run takes tens of minutes to hours and
# several GB of RAM per party: use THREADS=2..4 only if memory allows (~1 GB per concurrent
# multiplication), and try --logn 12 --insecure --iters 3 first to see the whole flow in seconds.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN="${BUILD_DIR:-build}/lr_helr"
[ -x "$BIN" ] || { echo "$BIN not found; run scripts/build.sh first"; exit 1; }
PORT="${PORT:-$(( (RANDOM % 20000) + 20000 ))}"
mkdir -p results
T=""; [ -n "${THREADS:-}" ] && T="--threads $THREADS"
V=""; [ "${VERIFY:-1}" = "1" ] && V="--verify"
"$BIN" server --port "$PORT" $T $V --csv results/results.csv --json results/last_run.json "$@" \
  > results/server.log 2>&1 &
SPID=$!
"$BIN" client --host 127.0.0.1 --port "$PORT" $T "$@" > results/client.log 2>&1
CRC=$?
wait "$SPID"; SRC=$?
cat results/server.log
echo "(client exit $CRC, server exit $SRC; client log: results/client.log)"
exit $(( CRC | SRC ))
