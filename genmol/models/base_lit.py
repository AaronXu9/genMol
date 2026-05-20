"""BaseGenLitModule: shared LightningModule for all generative models.

Owns a `Process` and a `Backbone` via composition. The training step is just:
    loss = self.process.loss(self.backbone, batch)
"""
from __future__ import annotations

import math
from typing import Optional

import lightning as L
import torch
from torch import Tensor

from genmol.data.base import ProcessBatch
from genmol.nn.backbone import Backbone
from genmol.process.base import Process


class BaseGenLitModule(L.LightningModule):
    """LightningModule glue. Subclasses for QM9 / SBDD set the model-config
    defaults but the body is identical."""

    def __init__(
        self,
        backbone: Backbone,
        process: Process,
        lr: float = 1e-4,
        weight_decay: float = 1e-12,
        warmup_steps: int = 1000,
        max_steps: int = 200_000,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        # Avoid storing the backbone/process in hparams (not picklable in general).
        self.save_hyperparameters(ignore=["backbone", "process"])
        self.backbone = backbone
        self.process = process
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.ema_decay = ema_decay

        # Sanity: backbone.prediction_type must match the process.
        if getattr(backbone, "prediction_type", None) != process.prediction_type:
            raise ValueError(
                f"backbone.prediction_type={backbone.prediction_type!r} but "
                f"process.prediction_type={process.prediction_type!r} — these must agree"
            )

    # ===== Lightning hooks =====

    def training_step(self, batch_dict: dict, batch_idx: int) -> Tensor:
        batch = self._to_process_batch(batch_dict)
        loss = self.process.loss(self.backbone, batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch_dict: dict, batch_idx: int) -> Tensor:
        batch = self._to_process_batch(batch_dict)
        loss = self.process.loss(self.backbone, batch)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.backbone.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return [opt], [{"scheduler": sched, "interval": "step"}]

    # ===== Helpers =====

    def _to_process_batch(self, batch_dict: dict) -> ProcessBatch:
        return ProcessBatch(
            x=batch_dict["x"], h=batch_dict["h"], mask=batch_dict["mask"],
            cond=batch_dict.get("cond"),
        )
