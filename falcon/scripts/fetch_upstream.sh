#!/usr/bin/env bash
# Fetch the upstream Falcon sources (pinned commit) into third_party/falcon-public, for reference only:
# include/falcon.h cites them as file:line (src/Functionalities.cpp, src/tools.{h,cpp}, src/Precompute.cpp,
# src/FCLayer.cpp). Nothing from third_party/ is compiled.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=https://github.com/snwagh/falcon-public.git
COMMIT=e11808c
mkdir -p third_party
if [ ! -d third_party/falcon-public/.git ]; then
  git clone --quiet "$REPO" third_party/falcon-public
fi
git -C third_party/falcon-public fetch --quiet origin
git -C third_party/falcon-public checkout --quiet "$COMMIT"
echo "third_party/falcon-public at $(git -C third_party/falcon-public rev-parse --short HEAD)"
