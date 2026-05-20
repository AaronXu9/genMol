"""Phase-2 seam: reference-model loading.

For DPO and PPO/GRPO, we need a frozen copy of the original generator.
This is a small utility so the seam can be exercised in Phase 1 tests.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def load_reference_model(ckpt: str | Path) -> nn.Module:
    """Load a Lightning checkpoint and return the underlying nn.Module with
    requires_grad=False on every parameter.
    """
    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in state:
        raise ValueError(f"Not a Lightning checkpoint: {ckpt_path}")

    # The caller is responsible for instantiating the LitModule and loading
    # the state_dict. This helper only provides the freezing convention.
    raise NotImplementedError(
        "Phase 2: caller-specific. Pattern:\n"
        "    model = MyLitModule.load_from_checkpoint(ckpt)\n"
        "    for p in model.parameters(): p.requires_grad_(False)\n"
        "    model.eval()\n"
        "    return model"
    )


def freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    return module
