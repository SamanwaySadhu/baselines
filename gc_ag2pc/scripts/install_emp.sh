#!/usr/bin/env bash
# Install the pinned, mutually compatible EMP snapshot (emp-tool + emp-ot + emp-ag2pc).
#
#   scripts/install_emp.sh                 # installs into $HOME/emp (no sudo needed)
#   PREFIX=/usr/local scripts/install_emp.sh   # system-wide (uses sudo for the install step)
#
# Why pinned: emp-ag2pc main (a245ca0, 2026-06-22) does not compile against later
# emp-tool commits (>= 39e513c redefines joinNclean / bit0_mask). These three
# commits were released together on 2026-06-22 and pass emp-ag2pc's test suite.
set -euo pipefail
PREFIX="${PREFIX:-$HOME/emp}"
SRC="${SRC:-$HOME/emp-src}"
JOBS="${JOBS:-$(nproc)}"
EMP_TOOL_COMMIT=2ea73a3
EMP_OT_COMMIT=842da83
EMP_AG2PC_COMMIT=a245ca0

SUDO=""
if [ ! -w "$(dirname "$PREFIX")" ] && [ ! -w "$PREFIX" ]; then SUDO="sudo"; fi

mkdir -p "$SRC"
fetch() {  # name commit
  if [ ! -d "$SRC/$1/.git" ]; then git clone https://github.com/emp-toolkit/$1.git "$SRC/$1"; fi
  git -C "$SRC/$1" fetch --quiet origin || true
  git -C "$SRC/$1" checkout --quiet "$2"
  echo "$1 @ $(git -C "$SRC/$1" rev-parse --short HEAD)"
}
build() {  # name extra-cmake-args...
  local n="$1"; shift
  rm -rf "$SRC/$n/build"
  cmake -S "$SRC/$n" -B "$SRC/$n/build" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX" "$@"
  cmake --build "$SRC/$n/build" -j "$JOBS"
}

fetch emp-tool  "$EMP_TOOL_COMMIT"
fetch emp-ot    "$EMP_OT_COMMIT"
fetch emp-ag2pc "$EMP_AG2PC_COMMIT"

build emp-tool -DEMP_TOOL_BUILD_TESTS=OFF -DEMP_TOOL_BUILD_BENCHMARKS=OFF
$SUDO cmake --install "$SRC/emp-tool/build"
build emp-ot -DEMP_OT_BUILD_TESTS=OFF
$SUDO cmake --install "$SRC/emp-ot/build"

# emp-ag2pc is header-only and has no install rules at this commit: build its own
# tests (sanity check) and copy the headers into the prefix.
build emp-ag2pc -DEMP_AG2PC_BUILD_TESTS=ON -DEMP_AG2PC_BUILD_EXAMPLES=OFF
(cd "$SRC/emp-ag2pc" && ctest --test-dir build --output-on-failure)
$SUDO rm -rf "$PREFIX/include/emp-ag2pc"
$SUDO mkdir -p "$PREFIX/include"
$SUDO cp -r "$SRC/emp-ag2pc/emp-ag2pc" "$PREFIX/include/emp-ag2pc"

echo
echo "EMP installed into $PREFIX"
echo "Build the baseline with:  EMP_PREFIX=$PREFIX scripts/build.sh"
