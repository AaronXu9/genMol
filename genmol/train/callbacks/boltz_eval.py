"""Nightly Boltz-2 evaluation hook. M4+; typically run via scripts/nightly_boltz.py."""
from __future__ import annotations

import lightning as L


class BoltzEvalCallback(L.Callback):
    def __init__(self, every_n_epochs: int = 1, n_top_k: int = 50):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.n_top_k = n_top_k

    def on_validation_epoch_end(self, trainer, pl_module):
        # M4: pick top-K by Vina, run Boltz-2 in subprocess.
        pass
