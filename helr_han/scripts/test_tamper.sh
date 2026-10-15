#!/usr/bin/env bash
# Fault-injection test of the malicious-security checks, with the real two-process binary on a tiny INSECURE ring
# (so each case takes seconds): every deviation by the server or the client must make the other party ABORT
# (exit code 3) and, for the server's deviations, the client must not accept or print any weights.
#   scripts/test_tamper.sh [extra lr_helr_han flags]
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=build/lr_helr_han
mkdir -p results
COMMON="--d 5 --iters 4 --logn 12 --threads 4 --quiet"
fail=0
run_case() {  # <tamper-role> <kind> <detector-role> <expected message fragment>
  local role=$1 kind=$2 det=$3 frag=$4
  local PORT=$(( (RANDOM % 15000) + 20000 )) S="" C=""
  [ "$role" = server ] && S="--tamper $kind"
  [ "$role" = client ] && C="--tamper $kind"
  "$BIN" server --port "$PORT" $COMMON $S "$@" > results/t_server.log 2>&1 &
  local SPID=$!
  "$BIN" client --port "$PORT" $COMMON $C "$@" > results/t_client.log 2>&1
  local CRC=$?
  wait "$SPID"; local SRC=$?
  local drc=$CRC dlog=results/t_client.log
  [ "$det" = server ] && { drc=$SRC; dlog=results/t_server.log; }
  if [ "$drc" = 3 ] && grep -q "$frag" "$dlog" && ! grep -q "trained weights" results/t_client.log; then
    printf "%-7s %-12s -> %s aborted: %s\n" "$role" "$kind" "$det" "$(grep -h ABORT "$dlog" | head -1 | cut -c1-120)"
  else
    printf "%-7s %-12s -> NOT DETECTED as expected (client exit %s, server exit %s)\n" "$role" "$kind" "$CRC" "$SRC"
    fail=1
  fi
}
shift 0
run_case server result       client "differ from the client's own plaintext training" "$@"
run_case server oracle       client "differ from the client's own plaintext training" "$@"
run_case server level        client "unexpected shape or level" "$@"
run_case client range        server "coefficient out of range" "$@"
run_case client shape        server "unexpected shape or level" "$@"
run_case client key          server "unreduced residues" "$@"
run_case client missing-key  server "wrong number of evaluation keys" "$@"
[ "$fail" = 0 ] && echo "tamper test: PASS (every deviation was detected)" || echo "tamper test: FAIL"
exit $fail
