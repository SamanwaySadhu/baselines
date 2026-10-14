#!/usr/bin/env bash
# One training run on this machine: dealer (fresh preprocessing), then the server in
# the background and the client in the foreground; prints the server's report.
#   scripts/run_local.sh --d 5                 # any training flags (passed to all three)
#   PORT=12345 VERIFY=0 scripts/run_local.sh --d 100
# Logs: results/{dealer,server,client}.log; CSV row appended to results/results.csv
set -uo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-$(( (RANDOM % 20000) + 20000 ))}"
JOB="${JOB:-lr_$$}"
OFF="${OFFLINE_DIR:-results/offline}"
mkdir -p results
V=""; [ "${VERIFY:-1}" = "1" ] && V="--verify"
build/lr_mdml_dealer --offline-dir "$OFF" --job "$JOB" --json results/last_dealer.json "$@" > results/dealer.log 2>&1 \
  || { cat results/dealer.log; exit 1; }
cat results/dealer.log
build/lr_mdml server --port "$PORT" --offline-dir "$OFF" --job "$JOB" $V \
  --csv results/results.csv --json results/last_run.json "$@" > results/server.log 2>&1 &
SPID=$!
build/lr_mdml client --port "$PORT" --offline-dir "$OFF" --job "$JOB" ${TAMPER:+--tamper "$TAMPER"} "$@" \
  > results/client.log 2>&1
CRC=$?
wait "$SPID"; SRC=$?
cat results/server.log
echo "(client exit $CRC, server exit $SRC; client log: results/client.log)"
[ "${KEEP_OFFLINE:-0}" = "1" ] || rm -f "$OFF/$JOB"-party-0.txt "$OFF/$JOB"-party-1.txt "$OFF/$JOB"-verify.txt
exit $(( CRC | SRC ))
