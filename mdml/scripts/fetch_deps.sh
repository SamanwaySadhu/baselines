#!/usr/bin/env bash
# Fetch the pinned MD-ML sources (+ a small networking patch: peer host, TCP_NODELAY) and the
# header-only Eigen into ./third_party. No sudo needed; Boost (headers only) and
# OpenSSL (libcrypto) come from the system:
#   sudo apt install -y build-essential cmake git libboost-dev libssl-dev
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
TP="$ROOT/third_party"
MDML_COMMIT=65ad2d2c8d14586c0c707123f50d51fac17e86e3   # NemoYuan2008/MD-ML main, 2024
EIGEN_VER=3.4.0
mkdir -p "$TP"

if [ ! -f "$TP/MD-ML/.patched" ]; then
  rm -rf "$TP/MD-ML"
  git clone -q https://github.com/NemoYuan2008/MD-ML.git "$TP/MD-ML"
  git -C "$TP/MD-ML" -c advice.detachedHead=false checkout -q "$MDML_COMMIT"
  git -C "$TP/MD-ML" apply --ignore-whitespace "$ROOT/patches/mdml-network.patch"
  touch "$TP/MD-ML/.patched"
  echo "MD-ML $MDML_COMMIT -> $TP/MD-ML (patched)"
else
  echo "MD-ML already present: $TP/MD-ML"
fi

if [ ! -f "$TP/eigen-$EIGEN_VER/Eigen/Core" ]; then
  tmp="$(mktemp -d)"
  curl -fsSL "https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VER/eigen-$EIGEN_VER.tar.gz" -o "$tmp/eigen.tar.gz"
  tar -xzf "$tmp/eigen.tar.gz" -C "$TP"
  rm -rf "$tmp"
  echo "Eigen $EIGEN_VER -> $TP/eigen-$EIGEN_VER"
else
  echo "Eigen already present: $TP/eigen-$EIGEN_VER"
fi
