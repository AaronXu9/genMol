"""Convenience constructor wrapping EGNNBackbone with the EDM/FM conventions.

This file exists so that the backbone factory signature is decoupled from the
specific (`prediction_type`) of the calling Process — Hydra instantiates the
backbone with `prediction_type` set from the matching `process` config.
"""
from __future__ import annotations

from genmol.nn.egnn import EGNNBackbone


def build_egnn_dynamics(
    atom_feat_dim: int,
    hidden_dim: int = 256,
    n_layers: int = 9,
    time_embed_dim: int = 32,
    prediction_type: str = "noise",
    attention: bool = True,
) -> EGNNBackbone:
    return EGNNBackbone(
        atom_feat_dim=atom_feat_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        time_embed_dim=time_embed_dim,
        prediction_type=prediction_type,
        attention=attention,
    )
