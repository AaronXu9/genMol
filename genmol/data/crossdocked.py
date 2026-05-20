"""CrossDocked2020 LightningDataModule — implemented in M3."""
from __future__ import annotations

import lightning as L


class CrossDockedDataModule(L.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "M3: implement CrossDocked2020 DataModule. Use tools/crossdocked/"
            "{download.sh, extract_pockets.py} to materialize the pocket-ligand pairs."
        )
