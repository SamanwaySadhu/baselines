#!/usr/bin/env bash
# d in {5,12,14,31}, real parameters (N=2^16), WAN 50 ms RTT, no verification. Output: results/wan50/d<D>.{log,json}
set -uo pipefail
cd "$(dirname "$0")/.."
for d in ${DIMS:-5 12 14 31}; do
  WAN_RTT_MS="${WAN_RTT_MS:-50}" bash scripts/run_local.sh --d "$d" > "results/wan50/d$d.out" 2>&1
  cp results/client.log "results/wan50/d$d.client.log"; cp results/server.log "results/wan50/d$d.server.log"; cp results/last_run.json "results/wan50/d$d.json"
done
