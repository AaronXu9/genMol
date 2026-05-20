"""Periodic PoseBusters 3D-geometry evaluation. Stub for M4; minimal hook now."""
from __future__ import annotations

import lightning as L


class PoseBustersEvalCallback(L.Callback):
    def __init__(self, every_n_steps: int = 5000, n_samples: int = 50):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.n_samples = n_samples

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == 0 or trainer.global_step % self.every_n_steps != 0:
            return
        # M4: sample, decode, run posebusters, log pass rate.
        pass
