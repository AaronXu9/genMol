"""Pocket-conditioned EGNN backbone — M3 milestone.

Stub for now: raises NotImplementedError. Will be implemented in M3 after
vendoring DiffSBDD as a reference (`third_party/diffsbdd_ref/`). The plan:
- Two node types: ligand atoms (denoised) and pocket atoms (frozen).
- Pocket atoms participate in the graph but their coords/types are not updated.
- Use a node-type embedding and a pocket-mask in the message passing.
"""
from __future__ import annotations

from typing import Optional

from torch import Tensor

from genmol.nn.backbone import Backbone


class PocketEGNNBackbone(Backbone):
    """Pocket-conditioned variant. Implemented in M3."""

    def __init__(self, **kwargs):
        super().__init__()
        self.prediction_type = kwargs.get("prediction_type", "noise")
        raise NotImplementedError(
            "M3: implement pocket-conditioned EGNN. Reference DiffSBDD's "
            "equivariant_diffusion/dynamics.py once vendored."
        )

    def forward(self, x, h, t, mask, cond: Optional[dict] = None):
        raise NotImplementedError("M3")
