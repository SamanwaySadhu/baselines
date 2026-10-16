#!/usr/bin/env bash
# Fault-injection test (TEST ONLY flags --tamper KIND --cheater I; server I deviates once, +1 on one element).
# Malicious leg, for every kind and every cheating server, the run must abort:
#   - the client exits 3 or 4 and no log contains "trained weights",
#   - every honest server exits 3 or 4,
#   - the cheater exits 3 or 4 and logged "TAMPER fired",
#   - some honest server (the client for "output") logged the kind's detection message.
# Semi-honest leg (the same deviations, minus open-hash and check, which exist only in malicious mode, and
# tree-noncanonical, which is a parse error and aborts in both modes): all four exit 0, nobody logs ABORT.
#   scripts/test_tamper.sh [extra lr_falcon flags]     e.g. --d 5
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=build/lr_falcon
mkdir -p results
declare -A FRAG=(
  [open-value]="checked opening mismatch"
  [open-hash]="checked opening mismatch"
  [rec3]="final check failed|checked opening mismatch"
  [tree]="PC check failed|checked opening mismatch"
  [tree-noncanonical]="non-canonical field element"
  [mask]="PC check failed|checked opening mismatch"
  [dot]="final check failed|checked opening mismatch"
  [check]="checked opening mismatch"
  [output]="output components disagree"
)
MAL_KINDS="open-value open-hash rec3 tree tree-noncanonical mask dot check output"
SEMI_KINDS="open-value rec3 tree mask dot output"

run_case() {  # $1 = "" or --semi-honest, $2 = kind, $3 = cheater; sets CRC and SRC[0..2]
  local mode=$1 kind=$2 ch=$3 i
  shift 3
  local port=$(( (RANDOM % 12000) + 20000 )) rid=$(( (RANDOM << 15) | RANDOM ))
  local common=(--port "$port" --run-id "$rid" --timeout 30 --connect-timeout 30 --quiet --tamper "$kind" --cheater "$ch")
  local pids=()
  for i in 0 1 2; do
    "$BIN" server --id "$i" "${common[@]}" $mode "$@" > "results/t_server$i.log" 2>&1 &
    pids+=($!)
  done
  "$BIN" client "${common[@]}" $mode "$@" > results/t_client.log 2>&1
  CRC=$?
  SRC=()
  for i in 0 1 2; do wait "${pids[$i]}"; SRC+=($?); done
}

fail=0
echo "== malicious: every deviation must abort =="
for kind in $MAL_KINDS; do
  for ch in 0 1 2; do
    run_case "" "$kind" "$ch" "$@"
    ok=1; why=""
    case "$CRC" in 3|4) ;; *) ok=0; why="$why client exit $CRC;";; esac
    if grep -q "trained weights" results/t_client.log results/t_server*.log; then ok=0; why="$why weights printed;"; fi
    for i in 0 1 2; do
      case "${SRC[$i]}" in 3|4) ;; *) ok=0; why="$why S$i exit ${SRC[$i]};";; esac
    done
    grep -q "TAMPER fired" "results/t_server$ch.log" || { ok=0; why="$why tamper did not fire;"; }
    det=""
    if [ "$kind" = output ]; then
      det=$(grep -hE "ABORT: (${FRAG[$kind]})" results/t_client.log | head -1)
    else
      for i in 0 1 2; do
        [ "$i" = "$ch" ] && continue
        [ -z "$det" ] && det=$(grep -hE "ABORT: (${FRAG[$kind]})" "results/t_server$i.log" | head -1)
      done
    fi
    [ -n "$det" ] || { ok=0; why="$why no honest party logged '${FRAG[$kind]}';"; }
    if [ "$ok" = 1 ]; then
      printf "%-18s cheater=S%s -> aborted (exits C/S0/S1/S2 = %s/%s/%s/%s): %s\n" "$kind" "$ch" "$CRC" "${SRC[0]}" "${SRC[1]}" "${SRC[2]}" "$(echo "$det" | cut -c1-80)"
    else
      printf "%-18s cheater=S%s -> FAIL:%s\n" "$kind" "$ch" "$why"
      fail=1
    fi
  done
done
echo "== semi-honest: the same deviations are not detected (no checks), nobody aborts =="
for kind in $SEMI_KINDS; do
  for ch in 0 1 2; do
    run_case --semi-honest "$kind" "$ch" "$@"
    if [ "$CRC" = 0 ] && [ "${SRC[0]}" = 0 ] && [ "${SRC[1]}" = 0 ] && [ "${SRC[2]}" = 0 ] && ! grep -q ABORT results/t_client.log results/t_server*.log; then
      printf "%-18s cheater=S%s -> no abort (as expected without checks)\n" "$kind" "$ch"
    else
      printf "%-18s cheater=S%s -> FAIL (exits C/S0/S1/S2 = %s/%s/%s/%s)\n" "$kind" "$ch" "$CRC" "${SRC[0]}" "${SRC[1]}" "${SRC[2]}"
      fail=1
    fi
  done
done
[ "$fail" = 0 ] && echo "tamper test: PASS (malicious: every deviation detected; semi-honest: no aborts)" || echo "tamper test: FAIL"
exit $fail
