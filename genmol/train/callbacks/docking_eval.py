"""Vina docking evaluation via subprocess into conda env 'unidock2'. M3+."""
from __future__ import annotations

import lightning as L


class DockingEvalCallback(L.Callback):
    def __init__(self, every_n_steps: int = 10_000, n_samples: int = 100, receptor_pdb: str = ""):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.n_samples = n_samples
        self.receptor_pdb = receptor_pdb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # M3+: sample, write SDF, invoke VinaOracle via subprocess.
        pass
