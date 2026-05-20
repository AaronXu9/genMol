#!/usr/bin/env bash
# CrossDocked2020 (pocket10 subset) — ~50 GB Google Drive download.
# Reference: https://github.com/gnina/models/tree/master/data/CrossDocked2020
#
# Manual steps:
# 1. Go to https://bits.csb.pitt.edu/files/crossdock2020/
# 2. Download CrossDocked2020_v1.3.tgz (~50 GB) and types files.
# 3. Extract under data/raw/crossdocked/.
# 4. Use the split index from the DiffSBDD repo (split_by_name.pt) — see
#    third_party/diffsbdd_ref/data/ after vendoring.
set -euo pipefail
echo "MANUAL: see this file's header for download instructions."
exit 1
