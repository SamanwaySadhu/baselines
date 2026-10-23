#!/usr/bin/env bash
# Fault-injection test of the malicious security with the real binaries: for every kind of deviation and several
# cheating parties, the run must ABORT and must not produce a result.
#   scripts/test_tamper.sh [extra lr_spdz flags]     e.g. --d 3 --iters 2
#   CHEATERS="0 1 4" KINDS="triple matrix" TIMEOUT=60 scripts/test_tamper.sh
# Pass criteria for one run (malicious mode): the server prints no trained weights; EVERY party exits non-zero (3 =
# it detected the deviation, 1 = a peer had already aborted and closed the connection); at least one party exits 3
# with an ABORT line. A deviation in --semi-honest mode must not abort (that mode has no checks).
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=build/lr_spdz
KINDS="${KINDS:-open-pre open-online open-equiv triple matrix ot-send ot-choose mac-value mac-sigma input-equiv}"
CHEATERS="${CHEATERS:-0 1 4}"
TIMEOUT="${TIMEOUT:-120}"
mkdir -p results
B=4
args=("$@")
for ((i = 0; i < ${#args[@]}; ++i)); do [ "${args[$i]}" = "--batch" ] && B="${args[$((i + 1))]}"; done
fail=0
run_once() {  # kind cheater [extra flags]; sets CODES, SRVLOG
  local kind=$1 cheater=$2; shift 2
  local port=$(( (RANDOM % 9000) + 20000 )) pids=()
  for ((k = 1; k <= B; ++k)); do
    timeout "$TIMEOUT" "$BIN" client --id "$k" --port "$port" --quiet --tamper "$kind" --cheater "$cheater" "${args[@]}" "$@" > "results/t_client$k.log" 2>&1 &
    pids+=($!)
  done
  timeout "$TIMEOUT" "$BIN" server --port "$port" --quiet --tamper "$kind" --cheater "$cheater" "${args[@]}" "$@" > results/t_server.log 2>&1
  CODES=($?)
  for p in "${pids[@]}"; do wait "$p"; CODES+=($?); done
}
for kind in $KINDS; do
  for cheater in $CHEATERS; do
    # kinds that only a client can commit
    case "$kind" in ot-send|input-equiv) [ "$cheater" = 0 ] && continue ;; esac
    [ "$cheater" -gt "$B" ] && continue
    run_once "$kind" "$cheater"
    ok=1; any3=0
    for c in "${CODES[@]}"; do [ "$c" = 0 ] && ok=0; [ "$c" = 3 ] && any3=1; done
    grep -q "trained weights" results/t_server.log && ok=0
    [ "$any3" = 1 ] || ok=0
    msg=$(grep -h "ABORT" results/t_server.log results/t_client*.log | head -1 | cut -c1-100)
    if [ "$ok" = 1 ]; then
      printf "%-12s cheater=%d -> aborted (exit codes %s) %s\n" "$kind" "$cheater" "${CODES[*]}" "$msg"
    else
      printf "%-12s cheater=%d -> NOT DETECTED / hung (exit codes %s)\n" "$kind" "$cheater" "${CODES[*]}"
      fail=1
    fi
  done
done
# semi-honest: no checks, so the deviation goes through (and the run must not hang or abort)
run_once open-online 1 --semi-honest
bad=0; for c in "${CODES[@]}"; do [ "$c" != 0 ] && bad=1; done
[ "$bad" = 0 ] && echo "semi-honest, open-online by client 1 -> not aborted (as designed: no checks)" || { echo "semi-honest run aborted unexpectedly: ${CODES[*]}"; fail=1; }
[ "$fail" = 0 ] && echo "tamper test: PASS (every deviation was detected)" || echo "tamper test: FAIL"
exit $fail
