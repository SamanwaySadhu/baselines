#!/usr/bin/env bash
# Install the OLD emp-tool / emp-ot API that emp-agmpc needs, pinned to the commits that were HEAD when emp-agmpc's
# last code change (43e4953, 2021-07-20) landed, into a SEPARATE prefix so that the emp-tool 1.0 used by gc_ag2pc,
# secureml and falcon ($HOME/emp) is untouched.
#   PREFIX=$HOME/emp-legacy SRC=$HOME/emp-legacy-src scripts/install_emp_legacy.sh
set -euo pipefail
PREFIX="${PREFIX:-$HOME/emp-legacy}"
SRC="${SRC:-$HOME/emp-legacy-src}"
TOOL_COMMIT=80cb3b4bdef30c43feaed35b104e30691fc9fd9a   # emp-tool, 2021-07-20
OT_COMMIT=ff28eb5d9368783f57fe40bcf1158f3d34537948     # emp-ot,   2021-07-20
# The legacy headers predate some libstdc++ hygiene: GCC >= 13 needs these forced includes.
FLAGS="-include cstdint -include memory"
mkdir -p "$SRC"
fetch() {  # repo dir commit
  if [ ! -d "$SRC/$2/.git" ]; then git clone -q "https://github.com/emp-toolkit/$1.git" "$SRC/$2"; fi
  git -C "$SRC/$2" fetch -q origin || true
  git -C "$SRC/$2" checkout -q "$3"
}
fetch emp-tool emp-tool "$TOOL_COMMIT"
fetch emp-ot emp-ot "$OT_COMMIT"
for p in emp-tool emp-ot; do
  cmake -S "$SRC/$p" -B "$SRC/$p/build-legacy" -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="$FLAGS"
  cmake --build "$SRC/$p/build-legacy" -j "$(nproc)"
  cmake --install "$SRC/$p/build-legacy"
done
echo "installed legacy emp-tool ($TOOL_COMMIT) and emp-ot ($OT_COMMIT) into $PREFIX"
