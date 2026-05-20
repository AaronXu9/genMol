#!/usr/bin/env bash
# QM9 is downloaded by PyG on first DataModule.prepare_data().
# This script exists as a manual trigger.
set -euo pipefail
cd "$(dirname "$0")/../.."
python scripts/prepare_qm9.py
