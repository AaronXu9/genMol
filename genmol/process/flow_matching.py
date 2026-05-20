"""FlowMatchingProcess: conditional-OT flow matching on (x, h).

Lipman 2023 / Liu 2022. Direct velocity-prediction loss along a linear path.

Base distribution for x: COM-zeroed Gaussian (matching EDM convention so the
backbone can be swapped 1-for-1). Base distribution for h: standard Gaussian.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

from genmol.data.base import ProcessBatch
from genmol.process.base import Process
from genmol.process.flow_paths import PATHS, InterpolantPath
from genmol.utils.geometry import remove_mean, sample_com_zero_noise

if TYPE_CHECKING:
    from genmol.nn.backbone import Backbone
    from genmol.sample.sampler import GuidanceFn


class FlowMatchingProcess(Process):
    def __init__(self, path: str = "linear"):
        if path not in PATHS:
            raise ValueError(f"Unknown path: {path}")
        self.path: InterpolantPath = PATHS[path]()

    @property
    def prediction_type(self) -> str:
        return "velocity"

    def sample_t(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.rand(batch_size, device=device).clamp(1e-3, 1 - 1e-3)

    def corrupt(self, batch: ProcessBatch, t: Tensor) -> ProcessBatch:
        # x_0 is base noise, x_1 is data. Path goes 0 → data at t=1.
        x0_x = sample_com_zero_noise(batch.x.shape, batch.mask, batch.x.device)
        x0_h = torch.randn_like(batch.h) * batch.mask.unsqueeze(-1).float()

        x_t = self.path.interp(x0_x, batch.x, t)
        h_t = self.path.interp(x0_h, batch.h, t)
        v_target_x = self.path.target_velocity(x0_x, batch.x, t)
        v_target_h = self.path.target_velocity(x0_h, batch.h, t)

        return ProcessBatch(
            x=x_t * batch.mask.unsqueeze(-1).float(),
            h=h_t * batch.mask.unsqueeze(-1).float(),
            mask=batch.mask,
            cond=batch.cond,
            aux={"v_x": v_target_x, "v_h": v_target_h, "t": t},
        )

    def loss(self, backbone: "Backbone", batch: ProcessBatch) -> Tensor:
        t = self.sample_t(batch.batch_size, batch.x.device)
        corrupted = self.corrupt(batch, t)
        pred_x, pred_h = backbone(
            corrupted.x, corrupted.h, t, corrupted.mask, corrupted.cond
        )
        m = corrupted.mask.unsqueeze(-1).float()
        n = m.sum().clamp(min=1.0)
        loss_x = ((pred_x - corrupted.aux["v_x"]) ** 2 * m).sum() / (n * 3)
        loss_h = ((pred_h - corrupted.aux["v_h"]) ** 2 * m).sum() / (n * pred_h.shape[-1])
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
        """Forward Euler step along the velocity field from t to t+dt."""
        pred_v_x, pred_v_h = backbone(x_t.x, x_t.h, t, x_t.mask, x_t.cond)

        if guidance_fn is not None and guidance_scale != 0.0:
            g = guidance_fn(x_t.x, float(t[0].item()), x_t.cond)
            pred_v_x = pred_v_x + guidance_scale * g

        x_new = x_t.x + pred_v_x * dt.view(-1, 1, 1)
        h_new = x_t.h + pred_v_h * dt.view(-1, 1, 1)
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
        # From t=0 (noise) up to t=1 (data).
        return torch.linspace(0.0, 1.0, n_steps + 1, device=device)
