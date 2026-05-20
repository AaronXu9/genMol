"""File I/O helpers: SDF writers, PDB pocket loaders, content hashing."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def sha1_of_file(path: Path | str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sdf(mols: Iterable, path: Path | str) -> None:
    """Write an iterable of RDKit Mol objects to a single SDF file."""
    from rdkit import Chem

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(path)) as w:
        for m in mols:
            if m is not None:
                w.write(m)
