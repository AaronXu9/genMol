"""EDMProcess: variance-preserving diffusion on (x, h) with COM-zeroed noise.

Follows Hoogeboom et al. 2022 (E(3)-Equivariant Diffusion). Key choices:
- COM-zeroed Gaussian noise on coordinates → translation invariant.
- Joint noising of (x, h) on the same schedule.
- eps-prediction loss.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

from genmol.data.base import ProcessBatch
from genmol.process.base import Process
from genmol.process.schedules import SCHEDULES, NoiseSchedule
from genmol.utils.geometry import remove_mean, sample_com_zero_noise

if TYPE_CHECKING:
    from genmol.nn.backbone import Backbone
    from genmol.sample.sampler import GuidanceFn


class EDMProcess(Process):
    def __init__(self, schedule: str = "polynomial", schedule_kwargs: Optional[dict] = None):
        if schedule not in SCHEDULES:
            raise ValueError(f"Unknown schedule: {schedule}")
        self.schedule: NoiseSchedule = SCHEDULES[schedule](**(schedule_kwargs or {}))

    @property
    def prediction_type(self) -> str:
        return "noise"

    def sample_t(self, batch_size: int, device: torch.device) -> Tensor:
        # Uniform in (0, 1). Avoid exactly 0 and 1.
        return torch.rand(batch_size, device=device).clamp(1e-3, 1 - 1e-3)

    def corrupt(self, batch: ProcessBatch, t: Tensor) -> ProcessBatch:
        a = self.schedule.alpha(t).view(-1, 1, 1)
        s = self.schedule.sigma(t).view(-1, 1, 1)

        eps_x = sample_com_zero_noise(batch.x.shape, batch.mask, batch.x.device)
        eps_h = torch.randn_like(batch.h) * batch.mask.unsqueeze(-1).float()

        x_t = a * batch.x + s * eps_x
        h_t = a * batch.h + s * eps_h

        return ProcessBatch(
            x=x_t,
            h=h_t,
            mask=batch.mask,
            cond=batch.cond,
            aux={"eps_x": eps_x, "eps_h": eps_h, "t": t},
        )

    def loss(self, backbone: "Backbone", batch: ProcessBatch) -> Tensor:
        t = self.sample_t(batch.batch_size, batch.x.device)
        corrupted = self.corrupt(batch, t)
        pred_x, pred_h = backbone(
            corrupted.x, corrupted.h, t, corrupted.mask, corrupted.cond
        )
        # MSE on noise prediction, masked to real atoms.
        m = corrupted.mask.unsqueeze(-1).float()
        n = m.sum().clamp(min=1.0)
        loss_x = ((pred_x - corrupted.aux["eps_x"]) ** 2 * m).sum() / (n * 3)
        loss_h = ((pred_h - corrupted.aux["eps_h"]) ** 2 * m).sum() / (n * pred_h.shape[-1])
        return loss_x + loss_h

    def step(
        self,
        backbone: "Backbone",
        x_t: ProcessBatch,
        t: Tensor,
        dt: Tensor,
        guidance_fn: Optional["GuidanceFn"] = None,
        guidance_scale: float = 0.0,
    ) -> ProcessBatch:
        """DDIM-like reverse step from t to t-dt."""
        a_t = self.schedule.alpha(t).view(-1, 1, 1)
        s_t = self.schedule.sigma(t).view(-1, 1, 1)
        t_next = (t - dt).clamp(min=0.0)
        a_next = self.schedule.alpha(t_next).view(-1, 1, 1)
        s_next = self.schedule.sigma(t_next).view(-1, 1, 1)

        pred_eps_x, pred_eps_h = backbone(x_t.x, x_t.h, t, x_t.mask, x_t.cond)

        # Optional guidance: add a gradient term to the predicted noise.
        if guidance_fn is not None and guidance_scale != 0.0:
            g = guidance_fn(x_t.x, float(t[0].item()), x_t.cond)
            pred_eps_x = pred_eps_x - guidance_scale * g

        # Predict x_0 from x_t and eps
        x0_x = (x_t.x - s_t * pred_eps_x) / a_t.clamp(min=1e-8)
        x0_h = (x_t.h - s_t * pred_eps_h) / a_t.clamp(min=1e-8)

        # Compose x_{t-dt}
        x_new = a_next * x0_x + s_next * pred_eps_x
        h_new = a_next * x0_h + s_next * pred_eps_h
        x_new = remove_mean(x_new, x_t.mask)

        return ProcessBatch(x=x_new, h=h_new, mask=x_t.mask, cond=x_t.cond)

    def init_noise(
        self,
        n_atoms: Tensor,
        max_atoms: int,
        feature_dim: int,
        cond: Optional[dict],
        device: torch.device,
    ) -> ProcessBatch:
        B = int(n_atoms.shape[0])
        mask = torch.zeros(B, max_atoms, device=device)
        for b in range(B):
            mask[b, : int(n_atoms[b].item())] = 1

        x = sample_com_zero_noise((B, max_atoms, 3), mask, device)
        h = torch.randn(B, max_atoms, feature_dim, device=device) * mask.unsqueeze(-1)
        return ProcessBatch(x=x, h=h, mask=mask, cond=cond)

    def t_schedule(self, n_steps: int, device: torch.device) -> Tensor:
        # From t=1 down to t=0 (corruption → clean).
        return torch.linspace(1.0, 0.0, n_steps + 1, device=device)
