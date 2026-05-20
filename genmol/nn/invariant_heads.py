"""MLP heads producing invariant features (atom-type logits, charges)."""
from __future__ import annotations

from torch import Tensor, nn


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)
