"""Periodic RDKit-metric evaluation during training.

Includes a regression guard: if `validity` is exactly 0 for more than
`zero_validity_warn_after` consecutive evaluations past the warmup, log a
loud warning. The decoder bug we shipped initially (43a6322) produced exactly
0.0 validity for an entire 200k-step run without the run failing. This guard
makes that class of bug surface within a few thousand steps.
"""
from __future__ import annotations

import lightning as L
import torch

from genmol.eval.metrics_2d import rdkit_metrics
from genmol.sample.decode import decode_batch
from genmol.sample.sampler_impl import DiffusionSampler, FlowSampler
from genmol.utils.logging import get_logger

log = get_logger("genmol.callbacks.rdkit_eval")


class RDKitEvalCallback(L.Callback):
    def __init__(
        self,
        every_n_steps: int = 1000,
        n_samples: int = 64,
        n_atoms: int = 19,
        num_sampling_steps: int = 100,
        zero_validity_warn_after: int = 5,
        zero_validity_warmup_steps: int = 5_000,
    ):
        """
        Args:
            every_n_steps:              run the eval every N training steps.
            n_samples:                  molecules per eval.
            n_atoms:                    fixed atom count for sampling
                                        (matches QM9 average; sites that want
                                        variable size should call sample()
                                        directly with a per-sample tensor).
            num_sampling_steps:         reverse-process steps per sample.
            zero_validity_warn_after:   if validity is exactly 0 for this many
                                        consecutive evals past the warmup,
                                        log a warning.
            zero_validity_warmup_steps: don't warn before this global step
                                        (an untrained model legitimately
                                        produces ~0% validity).
        """
        super().__init__()
        self.every_n_steps = every_n_steps
        self.n_samples = n_samples
        self.n_atoms = n_atoms
        self.num_sampling_steps = num_sampling_steps
        self.zero_validity_warn_after = zero_validity_warn_after
        self.zero_validity_warmup_steps = zero_validity_warmup_steps
        self._consecutive_zero = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return
        self._run_eval(trainer, pl_module)

    def _run_eval(self, trainer, pl_module):
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
            self._maybe_warn_zero_validity(trainer.global_step, metrics["validity"])
        finally:
            backbone.train()

    def _maybe_warn_zero_validity(self, step: int, validity: float) -> None:
        if validity > 0.0:
            self._consecutive_zero = 0
            return
        self._consecutive_zero += 1
        if step < self.zero_validity_warmup_steps:
            return
        if self._consecutive_zero >= self.zero_validity_warn_after:
            log.warning(
                "RDKit validity has been 0.0 for %d consecutive evaluations "
                "(step %d). This is the failure mode of the decoder bug "
                "fixed in 43a6322. Check decode.py for a silently-swallowed "
                "exception, sample with a known-good ckpt, or run "
                "tests/unit/test_decode_known_mol.py.",
                self._consecutive_zero, step,
            )
