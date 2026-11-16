#!/usr/bin/env bash
# Clone google/fss_machine_learning (the Agarwal et al. EuroS&P'24 code), pin it, and drop our driver into it.
# Result: deps/fss_machine_learning, ready for `bazel build //applications/lr_baseline:lr_fss`.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=deps/fss_machine_learning
mkdir -p deps
[ -d "$REPO/.git" ] || { git clone https://github.com/google/fss_machine_learning "$REPO"; git -C "$REPO" checkout 4e369f04648c427b9eb73ec62135e7f5bc079ec3; }
mkdir -p "$REPO/applications/lr_baseline"
cp src/lr_fss.cc src/BUILD.lr_baseline "$REPO/applications/lr_baseline/"
mv "$REPO/applications/lr_baseline/BUILD.lr_baseline" "$REPO/applications/lr_baseline/BUILD"
echo "ok: $REPO ($(git -C $REPO rev-parse --short HEAD))"
