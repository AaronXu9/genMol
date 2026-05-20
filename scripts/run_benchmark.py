"""End-to-end evaluation harness for a trained checkpoint.

Loads a Lightning checkpoint, samples N molecules, computes RDKit 2D metrics
(validity / uniqueness / novelty / QED) and 3D geometry stats (bond-length
mean/std). Writes report.json + report.md under reports/<run_name>/.

Usage:
    python scripts/run_benchmark.py \\
        --ckpt logs/qm9_edm-.../checkpoints/epoch=102-step=200000.ckpt \\
        --model edm --n-samples 512 --num-steps 250 --output-dir reports/m1_edm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

# Silence RDKit's noisy valence-error stderr; we count them via metrics.
RDLogger.DisableLog("rdApp.*")

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from genmol.eval.metrics_2d import rdkit_metrics
from genmol.eval.metrics_3d import bond_length_stats
from genmol.nn.egnn import EGNNBackbone
from genmol.process.edm import EDMProcess
from genmol.process.flow_matching import FlowMatchingProcess
from genmol.sample.decode import decode_batch
from genmol.sample.sampler_impl import DiffusionSampler, FlowSampler


def load_train_smiles(qm9_cache: Path) -> set[str]:
    """Load a pre-built pickle of canonical training SMILES (RemoveHs applied,
    matching what the decoder produces). Build it once with
    `scripts/build_train_smiles.py`.
    """
    pkl = qm9_cache.parent / "train_smiles.pkl"
    if pkl.exists():
        import pickle
        with open(pkl, "rb") as f:
            s = pickle.load(f)
        print(f"  loaded {len(s)} canonical train SMILES from {pkl}", flush=True)
        return s
    print(
        f"  WARNING: {pkl} not found — novelty will equal uniqueness. "
        f"Run `python scripts/build_train_smiles.py` to fix.",
        flush=True,
    )
    return set()


def build_model_and_sampler(ckpt_path: Path, model_kind: str, num_sampling_steps: int):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"  ckpt epoch={state.get('epoch')}  global_step={state.get('global_step')}")

    pred_type = "noise" if model_kind == "edm" else "velocity"
    backbone = EGNNBackbone(
        atom_feat_dim=5, hidden_dim=256, n_layers=9,
        time_embed_dim=32, prediction_type=pred_type, attention=True,
    )
    backbone.load_state_dict(
        {k[len("backbone."):]: v for k, v in state["state_dict"].items()
         if k.startswith("backbone.")}
    )
    backbone.eval()

    if model_kind == "edm":
        proc = EDMProcess(schedule="polynomial")
        sampler = DiffusionSampler(backbone, proc, atom_feat_dim=5, num_steps=num_sampling_steps)
    elif model_kind == "flow":
        proc = FlowMatchingProcess(path="linear")
        sampler = FlowSampler(backbone, proc, atom_feat_dim=5, num_steps=num_sampling_steps)
    else:
        raise ValueError(model_kind)
    return backbone, sampler


def sample_in_batches(sampler, n_total: int, n_atoms_min: int, n_atoms_max: int,
                      batch_size: int, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    x_list, h_list, mask_list = [], [], []
    done = 0
    while done < n_total:
        b = min(batch_size, n_total - done)
        n_atoms = torch.randint(n_atoms_min, n_atoms_max + 1, (b,), generator=gen)
        with torch.no_grad():
            out = sampler.sample(n_samples=b, n_atoms=n_atoms, seed=seed + done)
        x_list.append(out["x"])
        h_list.append(out["h"])
        mask_list.append(out["mask"])
        done += b
        print(f"    sampled {done}/{n_total}", flush=True)
    # Pad to a common N for stacking (use max N seen)
    max_n = max(x.shape[1] for x in x_list)
    def pad_to(t, target):
        if t.shape[1] >= target:
            return t
        pad = torch.zeros(t.shape[0], target - t.shape[1], *t.shape[2:], dtype=t.dtype)
        return torch.cat([t, pad], dim=1)
    x = torch.cat([pad_to(t, max_n) for t in x_list], dim=0)
    h = torch.cat([pad_to(t, max_n) for t in h_list], dim=0)
    mask = torch.cat([pad_to(t, max_n) for t in mask_list], dim=0)
    return {"x": x, "h": h, "mask": mask}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", choices=["edm", "flow"], required=True)
    p.add_argument("--n-samples", type=int, default=512)
    p.add_argument("--num-steps", type=int, default=250)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--n-atoms-min", type=int, default=11)
    p.add_argument("--n-atoms-max", type=int, default=22)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--qm9-cache", type=str,
                   default="data/raw/qm9/processed/qm9_genmol.pt",
                   help="Path to gdb9 processed cache (its sibling csv has train SMILES).")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Benchmark: {args.model.upper()} from {args.ckpt}")
    print(f"    {args.n_samples} samples, {args.num_steps} sampling steps")

    # 1) Build model + sampler
    print("\n[1/4] Building model + sampler")
    _, sampler = build_model_and_sampler(Path(args.ckpt), args.model, args.num_steps)

    # 2) Sample N molecules
    print(f"\n[2/4] Sampling {args.n_samples} molecules")
    t0 = time.time()
    samples = sample_in_batches(
        sampler,
        n_total=args.n_samples,
        n_atoms_min=args.n_atoms_min,
        n_atoms_max=args.n_atoms_max,
        batch_size=args.batch,
        seed=args.seed,
    )
    t_sample = time.time() - t0
    print(f"    total sampling time: {t_sample:.1f}s ({args.n_samples / t_sample:.2f} mol/s)")

    # 3) Decode + load train SMILES + compute metrics
    print("\n[3/4] Decoding + computing metrics")
    mols = decode_batch(samples["x"], samples["h"], samples["mask"])
    train_smiles = load_train_smiles(Path(args.qm9_cache))
    metrics_2d = rdkit_metrics(mols, train_smiles=train_smiles)
    metrics_3d = bond_length_stats(mols)

    # 4) Write report
    print("\n[4/4] Writing report")
    report = {
        "ckpt": args.ckpt,
        "model": args.model,
        "n_samples_requested": args.n_samples,
        "num_sampling_steps": args.num_steps,
        "n_atoms_range": [args.n_atoms_min, args.n_atoms_max],
        "seed": args.seed,
        "sampling_time_seconds": t_sample,
        "samples_per_second": args.n_samples / t_sample,
        **metrics_2d,
        **metrics_3d,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    md = [
        f"# Benchmark report: {args.model.upper()}",
        f"- **Checkpoint**: `{args.ckpt}`",
        f"- **N samples**: {args.n_samples}",
        f"- **Sampling steps**: {args.num_steps}",
        f"- **N atoms**: random uniform [{args.n_atoms_min}, {args.n_atoms_max}]",
        f"- **Sampling time**: {t_sample:.1f}s ({args.n_samples / t_sample:.2f} mol/s)",
        "",
        "## RDKit 2D metrics",
    ]
    for k in ("total", "valid", "validity", "uniqueness", "novelty", "mean_qed"):
        md.append(f"- **{k}**: {metrics_2d.get(k):.4f}" if isinstance(metrics_2d.get(k), float) else f"- **{k}**: {metrics_2d.get(k)}")
    md.append("")
    md.append("## 3D geometry")
    for k, v in metrics_3d.items():
        md.append(f"- **{k}**: {v:.4f}" if isinstance(v, float) else f"- **{k}**: {v}")
    md.append("")
    md.append("## Example valid SMILES")
    valid_smi = [Chem.MolToSmiles(m) for m in mols if m is not None]
    for s in valid_smi[:20]:
        md.append(f"- `{s}`")

    (out_dir / "report.md").write_text("\n".join(md))
    print(f"\nReport written to {out_dir}/report.{{json,md}}")
    print(f"\nKey numbers:")
    print(f"  validity:   {metrics_2d['validity']:.3f}")
    print(f"  uniqueness: {metrics_2d['uniqueness']:.3f}")
    print(f"  novelty:    {metrics_2d['novelty']:.3f}")
    print(f"  mean QED:   {metrics_2d['mean_qed']:.3f}")
    print(f"  mean bond:  {metrics_3d['mean_bond_length']:.3f} Å (std {metrics_3d['std_bond_length']:.3f})")


if __name__ == "__main__":
    main()
