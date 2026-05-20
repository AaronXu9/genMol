"""Symlink PLINDER v2 from /mnt/katritch_lab2/aoxu/2024-06/v2/ → data/raw/plinder/."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PLINDER_SOURCE = Path("/mnt/katritch_lab2/aoxu/2024-06/v2")
DEST = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "plinder"


def main():
    if not PLINDER_SOURCE.exists():
        print(f"ERROR: {PLINDER_SOURCE} not found", file=sys.stderr)
        sys.exit(1)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() or DEST.is_symlink():
        print(f"{DEST} already exists; not overwriting")
        return
    os.symlink(PLINDER_SOURCE, DEST)
    print(f"Symlinked {DEST} -> {PLINDER_SOURCE}")


if __name__ == "__main__":
    main()
