"""Triggers QM9 download via PyG into data/raw/qm9/."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from genmol.data.qm9 import QM9DataModule


def main():
    dm = QM9DataModule(root="data/raw/qm9")
    dm.prepare_data()
    print("QM9 ready at data/raw/qm9/")


if __name__ == "__main__":
    main()
