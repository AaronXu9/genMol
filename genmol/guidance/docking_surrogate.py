"""Phase-2 stub: a learned docking-gradient surrogate."""
from __future__ import annotations


class DockingGradientSurrogate:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "Phase 2: train a small SE(3)-equivariant net on (pocket, ligand) → Vina,"
            " then use its grad as a guidance term."
        )
