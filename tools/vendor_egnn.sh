#!/usr/bin/env bash
# Vendors EGNN code from the on-disk clone into third_party/egnn_clean/.
# Idempotent: rerunning overwrites the destination.
set -euo pipefail

SRC=/home/aoxu/projects/egnn/models/egnn_clean
DST="$(cd "$(dirname "$0")/.." && pwd)/third_party/egnn_clean"

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: EGNN source not found at $SRC" >&2
  exit 1
fi

mkdir -p "$DST"
cp "$SRC/egnn_clean.py" "$DST/egnn_clean.py"
cp "$SRC/__init__.py" "$DST/__init__.py"
cp /home/aoxu/projects/egnn/LICENSE "$DST/LICENSE"

echo "EGNN vendored to $DST"
ls -la "$DST"
