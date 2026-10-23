#!/usr/bin/env bash
# Run both parties on this machine (server in the background, client in the foreground) and print the
# server's report.
#   scripts/run_local.sh --d 5                      # any lr_secureml training/protocol flags
#   PORT=12345 VERIFY=0 scripts/run_local.sh --d 100 --semi-honest
# WAN simulation (see ../wan/wan.sh): WAN_RTT_MS=40 WAN_BW_MBPS=100 scripts/run_local.sh --d 5
#   routes the client's connections through ../wan/wanproxy (RTT/2 one-way delay, one shared bandwidth pipe per
#   direction) and prints a NETWORK WAIT table. WAN_RTT_MS=0 = loopback with wait accounting only (LAN reference).
# Logs: results/server.log, results/client.log; a CSV row is appended to results/results.csv
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=build/lr_secureml
PORT="${PORT:-$(( (RANDOM % 15000) + 20000 ))}"
mkdir -p results
V=""; [ "${VERIFY:-1}" = "1" ] && V="--verify"
# both parties share this machine: half of the hardware threads each (override with --threads N)
TH="--threads ${THREADS:-$(( ($(nproc) + 1) / 2 ))}"
source ../wan/wan.sh
trap wan_stop EXIT
FRONT=$(( PORT + 1000 ))
NTH="${THREADS:-$(( ($(nproc) + 1) / 2 ))}"
for ((i = 1; i <= $#; i++)); do [ "${!i}" = "--threads" ] && { j=$((i + 1)); NTH="${!j}"; }; done
wan_start "$FRONT:$PORT:$(( NTH + 1 ))" || exit 1
CP=(); [ "$WAN_ON" = 1 ] && CP=(--connect-port "$FRONT")
[ -n "$WAN_PRELOAD" ] && export LD_PRELOAD="$WAN_PRELOAD"
"$BIN" server --port "$PORT" $TH $V --csv results/results.csv --json results/last_run.json "$@" > results/server.log 2>&1 &
SPID=$!
"$BIN" client --port "$PORT" "${CP[@]}" $TH "$@" > results/client.log 2>&1
CRC=$?
wait "$SPID"; SRC=$?
unset LD_PRELOAD
wan_stop
cat results/server.log
echo "(client exit $CRC, server exit $SRC; client log: results/client.log)"
exit $(( CRC | SRC ))
