#!/usr/bin/env bash
# End-to-end multi-process runs over several configurations (batch = number of clients, d, segment size); each must
# give weights equal to the plaintext reference bit for bit (--verify) with every party exiting 0.
#   scripts/test_e2e.sh            (about a minute)
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
run() {
  local out
  out=$(VERIFY=1 bash scripts/run_local.sh --quiet "$@" 2>&1)
  local rc=$?
  if [ "$rc" = 0 ] && grep -q "verify: EXACT MATCH" <<<"$out"; then
    printf "%-55s ok   %s\n" "$*" "$(grep -o '[0-9]* ANDs in [0-9]* segments' <<<"$out")"
  else
    printf "%-55s FAIL (exit %s)\n" "$*" "$rc"; fail=1
  fi
}
run --d 5 --iters 20
run --d 1 --iters 3
run --d 7 --iters 4 --no-bias --zero-init --seed 3
run --d 16 --iters 3 --seed 4
run --d 5 --iters 3 --seg-ands 500          # many chained circuits per iteration
run --d 4 --iters 5 --batch 2 --lr 1.0 --seed 5
run --d 3 --iters 4 --batch 1 --lr 0.25 --seed 6 --seg-ands 300
run --d 6 --iters 3 --ssp 60
[ "$fail" = 0 ] && echo "e2e: PASS" || echo "e2e: FAIL"
exit $fail
