#!/usr/bin/env bash
# Fault-injection test of the malicious security: for every kind of deviation and each cheating party,
# the run must ABORT (exit 3 at the party that detects it) and must not produce a result.
#   scripts/test_tamper.sh [extra lr_secureml flags]     e.g. --d 5 --threads 4
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=build/lr_secureml
mkdir -p results
fail=0
for kind in open-pre open-online triple matrix ot-input mac-value mac-sigma; do
  for cheater in client server; do
    PORT=$(( (RANDOM % 15000) + 20000 ))
    "$BIN" server --port "$PORT" --quiet --tamper "$kind" --cheater "$cheater" "$@" > results/t_server.log 2>&1 &
    SPID=$!
    "$BIN" client --port "$PORT" --quiet --tamper "$kind" --cheater "$cheater" "$@" > results/t_client.log 2>&1
    CRC=$?
    wait "$SPID"; SRC=$?
    if [ "$CRC" = 3 ] && [ "$SRC" = 3 ] && ! grep -q "trained weights" results/t_server.log; then
      printf "%-12s cheater=%-6s -> aborted (client %s: %s)\n" "$kind" "$cheater" "$CRC" "$(grep -h ABORT results/t_client.log | head -1 | cut -c1-110)"
    else
      printf "%-12s cheater=%-6s -> NOT DETECTED (client exit %s, server exit %s)\n" "$kind" "$cheater" "$CRC" "$SRC"
      fail=1
    fi
  done
done
[ "$fail" = 0 ] && echo "tamper test: PASS (every deviation was detected)" || echo "tamper test: FAIL"
exit $fail
