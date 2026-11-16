#!/usr/bin/env bash
# Needs bazelisk as `bazel` on PATH. The repo's WORKSPACE (2022, gRPC/abseil era) needs an old Bazel.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/bin:$PATH" USE_BAZEL_VERSION="${USE_BAZEL_VERSION:-5.4.1}"
scripts/fetch_deps.sh
cd deps/fss_machine_learning
bazel build --copt=-Wno-error --conlyopt=-Wno-error --host_copt=-Wno-error --per_file_copt=external/boringssl/.*@-Wno-error,-Wno-dangling-pointer "${@:-//applications/lr_baseline:lr_fss}"
cp -f bazel-bin/applications/lr_baseline/lr_fss ../../lr_fss
echo "built fss_lr/lr_fss"
