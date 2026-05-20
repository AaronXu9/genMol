"""Generate N samples from a checkpoint and write them to an SDF file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from genmol.sample.decode import decode_batch
from genmol.sample.sampler_impl import DiffusionSampler, FlowSampler
from genmol.utils.io import write_sdf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--n-atoms", type=int, default=19)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--out", type=str, default="samples.sdf")
    args = p.parse_args()

    # Caller is responsible for instantiating the right LitModule subclass via Hydra
    # before invoking us. M1+ glue: load Lit module, derive sampler kind by process.
    raise NotImplementedError(
        "scripts/sample.py: load Lit module from ckpt, pick DiffusionSampler vs "
        "FlowSampler from process.prediction_type, then sample()."
    )


if __name__ == "__main__":
    main()
