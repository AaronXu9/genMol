"""Hydra entrypoint for training.

Usage:
    python -m scripts.train experiment=qm9_edm
    python -m scripts.train experiment=qm9_flow
    python -m scripts.train experiment=smoke   # CPU, 10 steps
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# Allow `python scripts/train.py ...` from the repo root.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from genmol.utils.logging import get_logger
from genmol.utils.seeding import set_global_seed

log = get_logger("genmol.train")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    set_global_seed(int(cfg.seed))

    # Instantiate components.
    data = instantiate(cfg.data)
    backbone = instantiate(cfg.backbone)
    process = instantiate(cfg.process)
    model = instantiate(cfg.model, backbone=backbone, process=process)

    # Build callbacks + logger from their config blocks.
    callbacks = []
    if cfg.get("callbacks"):
        for _, cb_cfg in cfg.callbacks.items():
            if cb_cfg is None:
                continue
            callbacks.append(instantiate(cb_cfg))

    logger = instantiate(cfg.logger) if cfg.get("logger") else None

    trainer = L.Trainer(
        max_steps=int(cfg.trainer.max_steps),
        accelerator=cfg.trainer.get("accelerator", "auto"),
        devices=cfg.trainer.get("devices", 1),
        precision=cfg.trainer.get("precision", 32),
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 50),
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 1),
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=cfg.trainer.get("enable_progress_bar", True),
        enable_checkpointing=cfg.trainer.get("enable_checkpointing", True),
        default_root_dir=cfg.trainer.get("default_root_dir", cfg.output_dir),
        val_check_interval=cfg.trainer.get("val_check_interval", 1.0),
        num_sanity_val_steps=cfg.trainer.get("num_sanity_val_steps", 0),
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
