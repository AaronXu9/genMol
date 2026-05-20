"""Concrete sampler implementations for diffusion and flow matching.

Lives in a separate module from `sampler.py` (which defines the ABC) so that
the ABC has no concrete dependencies, keeping import graphs clean for tests.
"""
from __future__ import annotations

import math
from typing import Optional, Union

import torch
from torch import Tensor

from genmol.data.base import ProcessBatch
from genmol.process.base import Process
from genmol.sample.logprob import gaussian_log_prob
from genmol.sample.sampler import GuidanceFn, Sampler
from genmol.utils.geometry import remove_mean


def _n_atoms_to_tensor(n_atoms: Union[int, Tensor], n_samples: int, device) -> Tensor:
    if isinstance(n_atoms, int):
        return torch.full((n_samples,), n_atoms, dtype=torch.long, device=device)
    return n_atoms.to(device).long()


class DiffusionSampler(Sampler):
    def __init__(
        self,
        model,
        process: Process,
        atom_feat_dim: int,
        num_steps: int = 250,
    ):
        self.model = model
        self.process = process
        self.atom_feat_dim = atom_feat_dim
        self.num_steps = num_steps

    def sample(
        self,
        n_samples,
        n_atoms,
        cond=None,
        guidance_fn: Optional[GuidanceFn] = None,
        guidance_scale: float = 0.0,
        return_log_prob: bool = False,
        return_trajectory: bool = False,
        seed: Optional[int] = None,
        temperature: float = 1.0,
    ) -> dict:
        device = next(self.model.parameters()).device
        if seed is not None:
            torch.manual_seed(seed)

        n_atoms_t = _n_atoms_to_tensor(n_atoms, n_samples, device)
        max_atoms = int(n_atoms_t.max().item())
        state = self.process.init_noise(
            n_atoms_t, max_atoms, self.atom_feat_dim, cond, device
        )

        ts = self.process.t_schedule(self.num_steps, device)
        trajectory = [state] if return_trajectory else None
        log_prob = torch.zeros(n_samples, device=device) if return_log_prob else None

        for i in range(self.num_steps):
            t = ts[i].expand(n_samples)
            dt = (ts[i] - ts[i + 1]).expand(n_samples)
            next_state = self.process.step(
                self.model, state, t, dt,
                guidance_fn=guidance_fn, guidance_scale=guidance_scale,
            )
            if return_log_prob:
                # Deterministic step → log-prob is a constant; for true RL we
                # use stochastic_sample. This branch exists for completeness.
                pass
            state = next_state
            if return_trajectory:
                trajectory.append(state)

        return {
            "x": state.x,
            "h": state.h,
            "mask": state.mask,
            "log_prob": log_prob,
            "trajectory": trajectory,
        }

    def stochastic_sample(
        self,
        n_samples,
        n_atoms,
        cond=None,
        guidance_fn: Optional[GuidanceFn] = None,
        guidance_scale: float = 0.0,
        return_log_prob: bool = False,
        return_trajectory: bool = False,
        seed: Optional[int] = None,
        temperature: float = 1.0,
    ) -> dict:
        """Ancestral DDPM sampling with per-step noise injection.

        For Phase-2 DDPO: returns per-sample log_prob with grad.
        """
        device = next(self.model.parameters()).device
        if seed is not None:
            torch.manual_seed(seed)

        n_atoms_t = _n_atoms_to_tensor(n_atoms, n_samples, device)
        max_atoms = int(n_atoms_t.max().item())
        state = self.process.init_noise(
            n_atoms_t, max_atoms, self.atom_feat_dim, cond, device
        )

        ts = self.process.t_schedule(self.num_steps, device)
        trajectory = [state] if return_trajectory else None
        log_prob = (
            torch.zeros(n_samples, device=device, requires_grad=False)
            if return_log_prob
            else None
        )

        for i in range(self.num_steps):
            t = ts[i].expand(n_samples)
            dt = (ts[i] - ts[i + 1]).expand(n_samples)
            mean_state = self.process.step(
                self.model, state, t, dt,
                guidance_fn=guidance_fn, guidance_scale=guidance_scale,
            )
            # Inject stochastic noise (small) — placeholder; full DDPM-style
            # noise level should be derived from the schedule.
            noise_std = temperature * 0.01
            x_noisy = mean_state.x + noise_std * torch.randn_like(mean_state.x)
            x_noisy = remove_mean(x_noisy, state.mask)

            if return_log_prob:
                step_lp = gaussian_log_prob(x_noisy, mean_state.x, noise_std, state.mask)
                log_prob = log_prob + step_lp

            state = ProcessBatch(
                x=x_noisy, h=mean_state.h, mask=state.mask, cond=state.cond
            )
            if return_trajectory:
                trajectory.append(state)

        return {
            "x": state.x,
            "h": state.h,
            "mask": state.mask,
            "log_prob": log_prob,
            "trajectory": trajectory,
        }


class FlowSampler(Sampler):
    def __init__(self, model, process: Process, atom_feat_dim: int, num_steps: int = 100):
        self.model = model
        self.process = process
        self.atom_feat_dim = atom_feat_dim
        self.num_steps = num_steps

    def sample(
        self,
        n_samples,
        n_atoms,
        cond=None,
        guidance_fn: Optional[GuidanceFn] = None,
        guidance_scale: float = 0.0,
        return_log_prob: bool = False,
        return_trajectory: bool = False,
        seed: Optional[int] = None,
        temperature: float = 1.0,
    ) -> dict:
        device = next(self.model.parameters()).device
        if seed is not None:
            torch.manual_seed(seed)

        n_atoms_t = _n_atoms_to_tensor(n_atoms, n_samples, device)
        max_atoms = int(n_atoms_t.max().item())
        state = self.process.init_noise(
            n_atoms_t, max_atoms, self.atom_feat_dim, cond, device
        )

        ts = self.process.t_schedule(self.num_steps, device)
        trajectory = [state] if return_trajectory else None

        for i in range(self.num_steps):
            t = ts[i].expand(n_samples)
            dt = (ts[i + 1] - ts[i]).expand(n_samples)
            state = self.process.step(
                self.model, state, t, dt,
                guidance_fn=guidance_fn, guidance_scale=guidance_scale,
            )
            if return_trajectory:
                trajectory.append(state)

        return {
            "x": state.x,
            "h": state.h,
            "mask": state.mask,
            "log_prob": None,
            "trajectory": trajectory,
        }

    def stochastic_sample(
        self,
        n_samples,
        n_atoms,
        cond=None,
        guidance_fn: Optional[GuidanceFn] = None,
        guidance_scale: float = 0.0,
        return_log_prob: bool = False,
        return_trajectory: bool = False,
        seed: Optional[int] = None,
        temperature: float = 1.0,
    ) -> dict:
        """ODE→SDE conversion à la FlowGRPO. Adds Gaussian noise at each step
        while preserving marginals (Liu 2025). For Phase-1 we provide a
        simplified version: Euler step + small isotropic noise."""
        device = next(self.model.parameters()).device
        if seed is not None:
            torch.manual_seed(seed)

        n_atoms_t = _n_atoms_to_tensor(n_atoms, n_samples, device)
        max_atoms = int(n_atoms_t.max().item())
        state = self.process.init_noise(
            n_atoms_t, max_atoms, self.atom_feat_dim, cond, device
        )

        ts = self.process.t_schedule(self.num_steps, device)
        trajectory = [state] if return_trajectory else None
        log_prob = (
            torch.zeros(n_samples, device=device) if return_log_prob else None
        )

        for i in range(self.num_steps):
            t = ts[i].expand(n_samples)
            dt = (ts[i + 1] - ts[i]).expand(n_samples)
            mean_state = self.process.step(
                self.model, state, t, dt,
                guidance_fn=guidance_fn, guidance_scale=guidance_scale,
            )
            noise_std = temperature * 0.01
            x_noisy = mean_state.x + noise_std * torch.randn_like(mean_state.x)
            x_noisy = remove_mean(x_noisy, state.mask)

            if return_log_prob:
                step_lp = gaussian_log_prob(x_noisy, mean_state.x, noise_std, state.mask)
                log_prob = log_prob + step_lp

            state = ProcessBatch(
                x=x_noisy, h=mean_state.h, mask=state.mask, cond=state.cond
            )
            if return_trajectory:
                trajectory.append(state)

        return {
            "x": state.x,
            "h": state.h,
            "mask": state.mask,
            "log_prob": log_prob,
            "trajectory": trajectory,
        }
