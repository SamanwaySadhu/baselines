#!/usr/bin/env bash
# Fault-injection test of the malicious security. For every deviation kind and every cheating party (the server and
# each client), the run must ABORT:
#   * the server never prints "trained weights";
#   * every honest party exits 3 (it detected the deviation) or 4 (a peer aborted / closed);
#   * at least one honest party exits 3 with an "ABORT:" line (it detected the deviation itself);
#   * the cheater logged "TAMPER fired" (the deviation really happened).
#   scripts/test_tamper.sh [lr_gc_agmpc flags]        e.g. --d 3 --iters 2
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=build/lr_gc_agmpc
mkdir -p results
B=4
args=("$@")
for ((i = 0; i < ${#args[@]}; ++i)); do [ "${args[$i]}" = "--batch" ] && B="${args[$((i + 1))]}"; done
# kind -> parties that can deviate in it (1 = server, 2.. = clients); the server garbles nothing and owns no input
declare -A WHO=(
  [gt-row]="clients" [label]="clients" [in-mask]="all" [input-equivocate]="clients" [chain-share]="all"
  [xy-share]="all" [d-bit]="all" [e-bit]="all" [abit-equivocate]="all" [out-label]="server" [out-mask]="clients" [echo]="all")
KINDS="${KINDS:-gt-row label in-mask input-equivocate chain-share xy-share d-bit e-bit abit-equivocate out-label out-mask echo}"
fail=0
for kind in $KINDS; do
  case "${WHO[$kind]}" in
    clients) cheaters=$(seq 2 $((B + 1))) ;;
    server) cheaters=1 ;;
    *) cheaters=$(seq 1 $((B + 1))) ;;
  esac
  for ch in $cheaters; do
    PORT=$(( (RANDOM % 9000) + 20000 ))
    COMMON=(--port "$PORT" --timeout 60 --connect-timeout 30 --quiet --tamper "$kind" --cheater "$ch")
    PIDS=()
    for ((k = 1; k <= B; ++k)); do
      timeout 300 "$BIN" client --id "$k" "${COMMON[@]}" "$@" > "results/t_client$k.log" 2>&1 &
      PIDS+=($!)
    done
    timeout 300 "$BIN" server "${COMMON[@]}" "$@" > results/t_server.log 2>&1
    CODES=($?)
    for p in "${PIDS[@]}"; do wait "$p"; CODES+=($?); done
    ok=1; detected=0
    grep -q "trained weights" results/t_server.log && ok=0
    logs=(results/t_server.log); for ((k = 1; k <= B; ++k)); do logs+=("results/t_client$k.log"); done
    grep -q "TAMPER fired" "${logs[$((ch - 1))]}" || ok=0
    for ((p = 1; p <= B + 1; ++p)); do
      c=${CODES[$((p - 1))]}
      [ "$p" = "$ch" ] && continue
      if [ "$c" != 3 ] && [ "$c" != 4 ]; then ok=0; fi
      if [ "$c" = 3 ] && grep -q "ABORT:" "${logs[$((p - 1))]}"; then detected=1; fi
    done
    [ "$detected" = 1 ] || ok=0
    who=$([ "$ch" = 1 ] && echo server || echo "client$((ch - 1))")
    if [ "$ok" = 1 ]; then
      msg=$(grep -h "ABORT:" "${logs[@]}" | grep -v "peer " | head -1 | cut -c1-90)
      printf "%-17s cheater=%-8s -> aborted (exit codes %s) %s\n" "$kind" "$who" "${CODES[*]}" "$msg"
    else
      printf "%-17s cheater=%-8s -> NOT DETECTED / WRONG (exit codes %s)\n" "$kind" "$who" "${CODES[*]}"
      fail=1
    fi
  done
done
[ "$fail" = 0 ] && echo "tamper test: PASS (every deviation was detected, no weights were output)" || echo "tamper test: FAIL"
exit $fail
