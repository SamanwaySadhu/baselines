#!/usr/bin/env bash
# Fetch the pinned HELR sources (Kim, Song, Wang, Xia, Jiang; MIT license) into
# ./third_party/HELR and apply the small patch (see patches/heaan.patch): an empty PubKey constructor so the program can build a PubKey
# without generating every key, and move constructors for Cipher and Message. Only HELR's src/*.cpp
# (its HEAAN snapshot of 2017/09) are compiled; the prebuilt .o / .a files that the
# upstream repository checks in are never used. System packages needed:
#   sudo apt install -y build-essential cmake git libntl-dev libgmp-dev libssl-dev
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
TP="$ROOT/third_party"
HELR_COMMIT=c94951f2691d55defc82f95d3144c831eb6c8796   # K-miran/HELR master (2019)
mkdir -p "$TP"

if [ ! -f "$TP/HELR/.patched" ]; then
  rm -rf "$TP/HELR"
  git clone -q https://github.com/K-miran/HELR.git "$TP/HELR"
  git -C "$TP/HELR" -c advice.detachedHead=false checkout -q "$HELR_COMMIT"
  git -C "$TP/HELR" apply --ignore-whitespace "$ROOT/patches/heaan.patch"
  touch "$TP/HELR/.patched"
  echo "HELR $HELR_COMMIT -> $TP/HELR (patched)"
else
  echo "HELR already present: $TP/HELR"
fi
