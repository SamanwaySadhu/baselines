#!/usr/bin/env bash
# Runs Falcon, SecureML and MD-ML over the emulated WAN and prints each one's NETWORK WAIT report.
#   wan/run_all.sh [--rtt-ms 40] [--bw-mbps 100] [--lan] -- <flags for every baseline, e.g. --d 5>
# --lan = loopback with wait accounting only (the LAN reference). Training flags go to all three programs.
# SecureML uses THREADS (default 8 here; its own default is half the cores) OT worker sockets per party.
set -uo pipefail
cd "$(dirname "$0")/.."
RTT=40; BW=100
while [ $# -gt 0 ]; do
  case "$1" in
    --rtt-ms) RTT="$2"; shift 2;;
    --bw-mbps) BW="$2"; shift 2;;
    --lan) RTT=0; BW=""; shift;;
    --) shift; break;;
    *) break;;
  esac
done
export WAN_RTT_MS="$RTT"
[ -n "$BW" ] && export WAN_BW_MBPS="$BW" || unset WAN_BW_MBPS
export THREADS="${THREADS:-8}"
echo "### WAN: RTT ${RTT} ms (one-way $(awk "BEGIN{print $RTT/2}") ms), bandwidth ${BW:-unlimited} Mbit/s per link and direction"
for b in falcon secureml mdml; do
  echo; echo "################ $b"
  ( cd "$b" && VERIFY=0 bash scripts/run_local.sh "$@" 2>&1 | sed -n '/NETWORK WAIT/,/^SUMMARY\|^communication\|^flights/p' | grep -v '^SUMMARY\|^communication\|^flights' )
done
