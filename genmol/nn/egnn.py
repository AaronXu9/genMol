"""EGNNBackbone: thin wrapper over the vendored `third_party/egnn_clean.EGNN`.

The vendored EGNN expects flat per-node tensors (h: (M, F), x: (M, 3)) plus an
edge_index of shape (2, E). We wrap it to accept the project's batched
(B, N, 3) / (B, N, F) / (B, N) mask convention used everywhere else.

For QM9 the molecule is treated as fully-connected within each sample
(standard EDM convention). This is wasteful for >100 atoms (CrossDocked
pocket conditioning), but acceptable for QM9 (≤29 atoms with H).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn

# Make the vendored EGNN importable.
_THIRD_PARTY = Path(__file__).resolve().parent.parent.parent / "third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

from egnn_clean.egnn_clean import EGNN  # type: ignore  # noqa: E402

from genmol.nn.backbone import Backbone
from genmol.nn.time_embed import SinusoidalTimeEmbedding


class EGNNBackbone(Backbone):
    """Equivariant backbone for joint (x, h) prediction over a batched graph.

    Args:
        atom_feat_dim:   F. Input/output feature dim (atom-type one-hot).
        hidden_dim:      EGNN hidden width.
        n_layers:        Number of E_GCL layers.
        time_embed_dim:  Width of the sinusoidal time embedding.
        prediction_type: 'noise' or 'velocity'. Both have the same head shape;
                         only the loss interpretation differs.
        attention:       Whether E_GCL uses attention on edges.
    """

    def __init__(
        self,
        atom_feat_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 9,
        time_embed_dim: int = 32,
        prediction_type: str = "noise",
        attention: bool = True,
    ):
        super().__init__()
        if prediction_type not in {"noise", "velocity", "x0"}:
            raise ValueError(f"Bad prediction_type: {prediction_type}")
        self.prediction_type = prediction_type
        self.atom_feat_dim = atom_feat_dim
        self.hidden_dim = hidden_dim

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # The vendored EGNN takes (h, x, edges, edge_attr=None). We feed it
        # node features = concat(atom_feats, time_embed_broadcast).
        in_node_nf = atom_feat_dim + time_embed_dim
        self.egnn = EGNN(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_dim,
            out_node_nf=atom_feat_dim,
            in_edge_nf=0,
            n_layers=n_layers,
            attention=attention,
            normalize=True,
            tanh=False,
        )

    def forward(
        self,
        x: Tensor,
        h: Tensor,
        t: Tensor,
        mask: Tensor,
        cond: Optional[dict] = None,
    ) -> tuple[Tensor, Tensor]:
        B, N, _ = x.shape
        device = x.device

        # Time embedding broadcast over N
        t_emb = self.time_embed(t)              # (B, T)
        t_emb_b = t_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, T)
        h_in = torch.cat([h, t_emb_b], dim=-1)

        # Flatten batch into one big graph with fully-connected per-sample edges.
        h_flat = h_in.reshape(B * N, -1)
        x_flat = x.reshape(B * N, 3)
        mask_flat = mask.reshape(B * N).bool()

        # Build per-sample fully connected edges, then offset by sample index.
        edges = self._build_edges(mask, device)

        # Run EGNN
        h_out_flat, x_out_flat = self.egnn(h_flat, x_flat, edges, edge_attr=None)

        # Predicted coordinate update: EGNN returns absolute positions.
        # For noise/velocity prediction, we want the delta from the input.
        pred_x_flat = x_out_flat - x_flat
        pred_x = pred_x_flat.reshape(B, N, 3) * mask.unsqueeze(-1).float()

        # h prediction (atom-type logits or noise on features)
        pred_h = h_out_flat.reshape(B, N, self.atom_feat_dim) * mask.unsqueeze(-1).float()

        return pred_x, pred_h

    @staticmethod
    def _build_edges(mask: Tensor, device) -> list[Tensor]:
        """Per-sample fully-connected edges (excluding self-loops), then
        offset to live in a single flat graph of size B*N. Returns the EGNN's
        expected [rows, cols] format."""
        B, N = mask.shape
        rows, cols = [], []
        for b in range(B):
            n_real = int(mask[b].sum().item())
            if n_real < 2:
                continue
            idx = torch.arange(n_real, device=device) + b * N
            r = idx.unsqueeze(1).expand(-1, n_real).reshape(-1)
            c = idx.unsqueeze(0).expand(n_real, -1).reshape(-1)
            keep = r != c
            rows.append(r[keep])
            cols.append(c[keep])
        if rows:
            return [torch.cat(rows), torch.cat(cols)]
        return [torch.zeros(0, dtype=torch.long, device=device)] * 2
