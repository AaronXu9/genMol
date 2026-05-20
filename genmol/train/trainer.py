"""Trainer construction from Hydra config."""
from __future__ import annotations

import lightning as L
from omegaconf import DictConfig


def build_trainer(cfg: DictConfig) -> L.Trainer:
    """cfg is the resolved Hydra trainer config block."""
    from hydra.utils import instantiate

    # Instantiate callbacks and logger via Hydra's `_target_` mechanism.
    callbacks = []
    if "callbacks" in cfg and cfg.callbacks:
        for _, cb_cfg in cfg.callbacks.items():
            callbacks.append(instantiate(cb_cfg))

    logger = None
    if cfg.get("logger") is not None:
        logger = instantiate(cfg.logger)

    return L.Trainer(
        max_steps=int(cfg.max_steps),
        accelerator=cfg.get("accelerator", "auto"),
        devices=cfg.get("devices", 1),
        precision=cfg.get("precision", 32),
        log_every_n_steps=cfg.get("log_every_n_steps", 50),
        gradient_clip_val=cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 1),
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=cfg.get("enable_progress_bar", True),
        enable_checkpointing=cfg.get("enable_checkpointing", True),
        default_root_dir=cfg.get("default_root_dir", "logs/"),
        check_val_every_n_epoch=cfg.get("check_val_every_n_epoch", None),
        val_check_interval=cfg.get("val_check_interval", 1.0),
    )
