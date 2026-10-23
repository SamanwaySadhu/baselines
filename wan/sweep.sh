#!/usr/bin/env bash
# Sweep WAN settings for one baseline: wan/sweep.sh <falcon|secureml|mdml> <outdir>
# Writes <outdir>/<baseline>_d<D>_rtt<R>_bw<B>.log for every setting.
set -uo pipefail
B="$1"; OUT="$2"; mkdir -p "$OUT"
cd "$(dirname "$0")/../$B"
for D in 5 100; do
  for S in "0:" "10:1000" "40:100" "80:50" "150:20"; do
    R="${S%%:*}"; W="${S##*:}"
    f="$OUT/${B}_d${D}_rtt${R}_bw${W:-inf}.log"
    echo "[$(date +%T)] $B d=$D rtt=$R bw=${W:-inf}"
    env WAN_RTT_MS="$R" ${W:+WAN_BW_MBPS=$W} THREADS=8 VERIFY=0 bash scripts/run_local.sh --d "$D" > "$f" 2>&1
  done
done
echo "[$(date +%T)] $B done"
