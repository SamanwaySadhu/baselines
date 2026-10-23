#!/usr/bin/env bash
# One training run on this machine: dealer (fresh preprocessing), then the server in
# the background and the client in the foreground; prints the server's report.
#   scripts/run_local.sh --d 5                 # any training flags (passed to all three)
#   PORT=12345 VERIFY=0 scripts/run_local.sh --d 100
# WAN simulation (see ../wan/wan.sh): WAN_RTT_MS=40 WAN_BW_MBPS=100 scripts/run_local.sh --d 5
#   routes the parties' connections through ../wan/wanproxy (RTT/2 one-way delay, bandwidth cap per link and
#   direction; MD-ML's Party opens one socket per direction) and prints a NETWORK WAIT table.
#   WAN_RTT_MS=0 = loopback with wait accounting only (LAN reference).
# Logs: results/{dealer,server,client}.log; CSV row appended to results/results.csv
set -uo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-$(( (RANDOM % 20000) + 20000 ))}"
JOB="${JOB:-lr_$$}"
OFF="${OFFLINE_DIR:-results/offline}"
mkdir -p results
# THREADS = OpenMP threads per party (default: all cores; both parties share this machine)
T=""; [ -n "${THREADS:-}" ] && T="--threads $THREADS"
V=""; [ "${VERIFY:-1}" = "1" ] && V="--verify"
build/lr_mdml_dealer --offline-dir "$OFF" --job "$JOB" --json results/last_dealer.json "$@" > results/dealer.log 2>&1 \
  || { cat results/dealer.log; exit 1; }
cat results/dealer.log
source ../wan/wan.sh
trap wan_stop EXIT
FRONT=$(( PORT + 1000 ))
# MD-ML's Party uses ports PORT+0..3 (one TCP connection per direction); each is its own link
wan_start "$FRONT:$PORT:1" "$((FRONT + 1)):$((PORT + 1)):1" "$((FRONT + 2)):$((PORT + 2)):1" "$((FRONT + 3)):$((PORT + 3)):1" || exit 1
[ "$WAN_ON" = 1 ] && export MDML_PEER_PORT="$FRONT"
[ -n "$WAN_PRELOAD" ] && export LD_PRELOAD="$WAN_PRELOAD"
build/lr_mdml server $T --port "$PORT" --offline-dir "$OFF" --job "$JOB" $V \
  --csv results/results.csv --json results/last_run.json "$@" > results/server.log 2>&1 &
SPID=$!
build/lr_mdml client $T --port "$PORT" --offline-dir "$OFF" --job "$JOB" ${TAMPER:+--tamper "$TAMPER"} "$@" \
  > results/client.log 2>&1
CRC=$?
wait "$SPID"; SRC=$?
unset LD_PRELOAD
wan_stop
cat results/server.log
echo "(client exit $CRC, server exit $SRC; client log: results/client.log)"
[ "${KEEP_OFFLINE:-0}" = "1" ] || rm -f "$OFF/$JOB"-party-0.txt "$OFF/$JOB"-party-1.txt "$OFF/$JOB"-verify.txt
exit $(( CRC | SRC ))
