#!/usr/bin/env bash
# One training run on this machine: the server in the background, the client in the
# foreground; prints the server's report.
#   scripts/run_local.sh --d 5                    # any lr_cryptonn training flags (both parties)
#   PORT=12345 THREADS=4 VERIFY=0 scripts/run_local.sh --d 100 --iters 20
# VERIFY=1 (default): the server checks the trained weights bit for bit against the
# plaintext fixed-point reference (test only: it regenerates the client's data from --seed).
# Logs: results/server.log, results/client.log; JSON: results/last_run.json;
# CSV row appended to results/results.csv. BUILD_DIR overrides ./build.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN="${BUILD_DIR:-build}/lr_cryptonn"
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
