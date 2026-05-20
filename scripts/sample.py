"""Generate N samples from a checkpoint and write them to an SDF file.

Usage:
    python scripts/sample.py --ckpt logs/.../epoch=102-step=200000.ckpt \\
        --model edm --n 64 --num-steps 250 --out samples.sdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from genmol.nn.egnn import EGNNBackbone
from genmol.process.edm import EDMProcess
from genmol.process.flow_matching import FlowMatchingProcess
from genmol.sample.decode import decode_batch
from genmol.sample.sampler_impl import DiffusionSampler, FlowSampler
from genmol.utils.io import write_sdf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", choices=["edm", "flow"], required=True)
    p.add_argument("--n", type=int, default=64, help="number of molecules to sample")
    p.add_argument("--n-atoms-min", type=int, default=11)
    p.add_argument("--n-atoms-max", type=int, default=22)
    p.add_argument("--num-steps", type=int, default=100, help="reverse-process steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out", default="samples.sdf")
    p.add_argument(
        "--fm-path", default="linear", choices=["linear", "vp"],
        help="flow-matching path (only used if --model=flow)",
    )
    args = p.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    pred_type = "noise" if args.model == "edm" else "velocity"
    backbone = EGNNBackbone(
        atom_feat_dim=5, hidden_dim=256, n_layers=9,
        time_embed_dim=32, prediction_type=pred_type, attention=True,
    ).to(device)
    backbone.load_state_dict({
        k[len("backbone."):]: v for k, v in state["state_dict"].items()
        if k.startswith("backbone.")
    })
    backbone.eval()

    if args.model == "edm":
        proc = EDMProcess(schedule="polynomial")
        sampler = DiffusionSampler(backbone, proc, atom_feat_dim=5, num_steps=args.num_steps)
    else:
        proc = FlowMatchingProcess(path=args.fm_path)
        sampler = FlowSampler(backbone, proc, atom_feat_dim=5, num_steps=args.num_steps)

    torch.manual_seed(args.seed)
    n_atoms = torch.randint(args.n_atoms_min, args.n_atoms_max + 1, (args.n,), device=device)
    with torch.no_grad():
        out = sampler.sample(n_samples=args.n, n_atoms=n_atoms, seed=args.seed)

    mols = decode_batch(out["x"], out["h"], out["mask"])
    n_valid = sum(m is not None for m in mols)
    print(f"sampled {args.n} molecules, {n_valid} valid ({n_valid/args.n:.1%})")

    write_sdf((m for m in mols if m is not None), args.out)
    print(f"wrote valid mols to {args.out}")


if __name__ == "__main__":
    main()
