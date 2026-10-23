# wan.sh -- sourced by the baselines' scripts/run_local.sh. Starts a WAN-emulating proxy (wanproxy) and arms the
# wait-time shim (libwanshim.so) for the party processes.
#
#   WAN_RTT_MS     unset: nothing changes (plain loopback run, no wait accounting)
#                  0 and no WAN_BW_MBPS: loopback run, but the shim measures network wait (the "LAN" reference)
#                  otherwise: all traffic goes through wanproxy with one-way latency RTT/2 and bandwidth WAN_BW_MBPS
#   WAN_BW_MBPS    link bandwidth per direction in Mbit/s (default: unlimited)
#   WAN_STATS      optional file for the proxy's traffic totals (JSON)
#
# Usage in a script:  wan_start FRONT:BACK[:COUNT] ...   (sets WAN_ON=1 if the proxy runs)
#                     ... run parties; each needs  ${WAN_PRELOAD:+env LD_PRELOAD=$WAN_PRELOAD}  ...
#                     wan_stop
WAN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAN_ON=0
WAN_PRELOAD=""
WAN_PID=""

wan_build() {
  [ -x "$WAN_DIR/wanproxy" ] || g++ -O2 -std=c++17 -pthread -o "$WAN_DIR/wanproxy" "$WAN_DIR/wanproxy.cpp"
  [ -f "$WAN_DIR/libwanshim.so" ] || gcc -O2 -fPIC -shared -o "$WAN_DIR/libwanshim.so" "$WAN_DIR/wanshim.c" -ldl
}

wan_start() {
  [ -n "${WAN_RTT_MS:-}" ] || return 0
  wan_build
  WAN_PRELOAD="$WAN_DIR/libwanshim.so"
  if [ "$WAN_RTT_MS" = "0" ] && [ -z "${WAN_BW_MBPS:-}" ]; then return 0; fi
  local maps=() m
  for m in "$@"; do maps+=(--map "$m"); done
  local rdy; rdy="$(mktemp)"
  "$WAN_DIR/wanproxy" --latency-ms "$(awk "BEGIN{print $WAN_RTT_MS/2}")" --bw-mbps "${WAN_BW_MBPS:-0}" \
      ${WAN_STATS:+--stats "$WAN_STATS"} "${maps[@]}" > "$rdy" &
  WAN_PID=$!
  local i
  for i in $(seq 1 200); do
    grep -q ready "$rdy" 2>/dev/null && break
    kill -0 "$WAN_PID" 2>/dev/null || { echo "wanproxy failed to start (port in use?)" >&2; rm -f "$rdy"; return 1; }
    sleep 0.05
  done
  rm -f "$rdy"
  WAN_ON=1
}

wan_stop() {
  [ -n "$WAN_PID" ] && { kill -TERM "$WAN_PID" 2>/dev/null; wait "$WAN_PID" 2>/dev/null; }
  WAN_PID=""
  return 0
}
