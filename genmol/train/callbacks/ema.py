"""Exponential moving average of weights — critical for diffusion/FM stability."""
from __future__ import annotations

import copy

import lightning as L
import torch
from torch import nn


class EMACallback(L.Callback):
    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self._shadow: dict[str, torch.Tensor] = {}
        self._initialized = False

    def _module_to_average(self, pl_module: L.LightningModule) -> nn.Module:
        # We average the backbone parameters; the rest are frozen/derived.
        return pl_module.backbone

    def on_train_start(self, trainer, pl_module):
        target = self._module_to_average(pl_module)
        self._shadow = {k: v.detach().clone() for k, v in target.state_dict().items()}
        self._initialized = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._initialized:
            return
        target = self._module_to_average(pl_module)
        with torch.no_grad():
            for k, v in target.state_dict().items():
                if v.dtype.is_floating_point:
                    self._shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
                else:
                    self._shadow[k] = v.detach().clone()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["ema_shadow"] = self._shadow
