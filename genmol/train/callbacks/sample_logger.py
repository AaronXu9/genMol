"""Logs generated molecules to W&B as 3D objects."""
from __future__ import annotations

import lightning as L


class SampleLoggerCallback(L.Callback):
    def __init__(self, every_n_steps: int = 5000, n_samples: int = 8):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.n_samples = n_samples

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # M1+: sample, decode to SDF blob, log to wandb.Molecule.
        pass
