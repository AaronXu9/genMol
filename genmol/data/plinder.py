"""PLINDER LightningDataModule — scaffolded; activated when M3+ uses real binding data.

The user has PLINDER v2 at /mnt/katritch_lab2/aoxu/2024-06/v2/.
Run `tools/plinder/stage_from_mount.py` to symlink it into `data/raw/plinder/`.
"""
from __future__ import annotations

import lightning as L


class PlinderDataModule(L.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "Scaffolded for later: parse `data/raw/plinder/systems/*` into "
            "PocketLigandDataset entries. Initial parser plan: filter to "
            "drug-like ligands (MW 200-500, atoms in {C,N,O,F,P,S,Cl,Br,I,H}), "
            "extract 10 Å pocket around each ligand."
        )
