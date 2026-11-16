#!/usr/bin/env bash
# Run the benchmark: a LAN-like run (RTT 0) and a WAN run (default RTT 50 ms) of one configuration, same data.
#   scripts/run_local.sh --n 1000 --d 10                 # variant 1 (paper's v1), 6 iterations
#   VARIANT=2 scripts/run_local.sh --n 1000 --d 10        # variant 2 (comparison-based sigmoid)
#   WAN_RTT_MS=80 WAN_BW_MBPS=100 scripts/run_local.sh --n 5000 --d 20
# Extra flags go to lr_fss (see ./lr_fss --help). JSON per run: results/<tag>_{lan,wan}.json
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./lr_fss
[ -x "$BIN" ] || { echo "build first: scripts/build.sh" >&2; exit 1; }
mkdir -p results
V="${VARIANT:-1}"
RTT="${WAN_RTT_MS:-50}"
BW="${WAN_BW_MBPS:-0}"
TAG="v${V}_$(date +%H%M%S)"
echo "### LAN (RTT 0)"
"$BIN" --variant "$V" --rtt-ms 0 --json "results/${TAG}_lan.json" "$@"
echo
echo "### WAN (RTT ${RTT} ms, bandwidth ${BW} Mbit/s, 0 = unlimited)"
"$BIN" --variant "$V" --rtt-ms "$RTT" --bw-mbps "$BW" --json "results/${TAG}_wan.json" "$@"
