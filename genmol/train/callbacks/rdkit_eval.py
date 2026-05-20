"""Periodic RDKit-metric evaluation during training."""
from __future__ import annotations

import lightning as L
import torch

from genmol.eval.metrics_2d import rdkit_metrics
from genmol.sample.decode import decode_batch
from genmol.sample.sampler_impl import DiffusionSampler, FlowSampler


class RDKitEvalCallback(L.Callback):
    def __init__(
        self,
        every_n_steps: int = 1000,
        n_samples: int = 64,
        n_atoms: int = 19,
        num_sampling_steps: int = 100,
    ):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.n_samples = n_samples
        self.n_atoms = n_atoms
        self.num_sampling_steps = num_sampling_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return
        self._run_eval(trainer, pl_module)

    def _run_eval(self, trainer, pl_module):
        # Pick sampler matching the prediction type.
        process = pl_module.process
        backbone = pl_module.backbone
        atom_feat_dim = getattr(backbone, "atom_feat_dim")
        if process.prediction_type == "noise":
            sampler = DiffusionSampler(backbone, process, atom_feat_dim, self.num_sampling_steps)
        else:
            sampler = FlowSampler(backbone, process, atom_feat_dim, self.num_sampling_steps)

        backbone.eval()
        try:
            with torch.no_grad():
                out = sampler.sample(self.n_samples, self.n_atoms, cond=None)
            mols = decode_batch(out["x"], out["h"], out["mask"])
            metrics = rdkit_metrics(mols)
            for k, v in metrics.items():
                pl_module.log(f"rdkit/{k}", float(v), prog_bar=False, on_step=True)
        finally:
            backbone.train()
