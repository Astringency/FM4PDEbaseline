from __future__ import annotations

import time
import warnings

import torch
import torch.nn.functional as F

from baselines.common.data_adapter import PDEBatch
from baselines.common.metrics import physics_loss_metric

from .base import BaselineModel
from .pinn_sparse import _physics_weight_metadata, _select_physics_loss, observation_loss_from_batch


class Var4DBaseline(BaselineModel):
    name = "var4d"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        self.optimized_numel = _optimized_state_numel(data_spec)
        return self

    def parameter_count(self) -> int:
        return int(self.optimized_numel)

    def fit(self, train_loader, val_loader=None):
        return {"status": "per_instance_4dvar"}

    def predict(self, batch: PDEBatch):
        start = time.perf_counter()
        if not batch.metadata.get("supports_trajectory", False):
            warnings.warn("4D-Var requested for data without trajectory metadata; using state reconstruction surrogate.", RuntimeWarning, stacklevel=2)
        state0, output_view, background, dyn_meta = _initial_trajectory(batch)
        state = torch.nn.Parameter(state0.detach().clone())
        steps = int(self.config.get("steps", 3))
        lr = float(self.config.get("lr", 2e-2))
        lam_b = float(self.config.get("lambda_background", 0.1))
        lam_o = float(self.config.get("lambda_obs", 1.0))
        opt = torch.optim.Adam([state], lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            output = output_view(state)
            obs = observation_loss_from_batch(output, batch)
            bg = F.mse_loss(_background_view(state, batch), background)
            dyn_meta = {**dyn_meta, **_physics_weight_metadata(self.config)}
            dyn = _select_physics_loss(physics_loss_metric(state, batch.pde_name, dyn_meta), self.config, state)
            if not torch.isfinite(dyn):
                dyn = torch.tensor(0.0, device=state.device)
            loss = lam_o * obs + lam_b * bg + dyn
            loss.backward()
            opt.step()
        batch.metadata["inference_optimization_time"] = time.perf_counter() - start
        return output_view(state).detach()


def _smoothness_surrogate(x: torch.Tensor) -> torch.Tensor:
    return (x[..., 1:, :] - x[..., :-1, :]).pow(2).mean() + (x[..., :, 1:] - x[..., :, :-1]).pow(2).mean()


def _optimized_state_numel(data_spec: dict) -> int:
    metadata = data_spec.get("metadata", {})
    full_shape = tuple(metadata.get("full_shape", ()))
    if len(full_shape) == 5:
        channels = int(full_shape[1])
        steps = int(full_shape[2])
        input_idx = int(metadata.get("input_time_index", 0))
        optimized_steps = max(steps - input_idx, 2)
        return int(channels * optimized_steps * full_shape[3] * full_shape[4])
    target_shape = tuple(data_spec.get("target_shape", ()))
    return int(torch.tensor(target_shape[1:]).prod().item()) if len(target_shape) > 1 else 0


def _initial_trajectory(batch: PDEBatch):
    pde = batch.pde_name.lower()
    full = batch.full_tensor.detach()
    guess = batch.metadata.get("voronoi_grid", batch.input_fields).detach()
    meta = {"full_tensor": full, "input_fields": batch.input_fields, "task": "trajectory", **batch.metadata}
    if pde == "nsnonbounded":
        initial = full[:, :, :1]
        target_guess = guess.reshape(guess.shape[0], 1, -1, guess.shape[-2], guess.shape[-1])
        state0 = torch.cat([initial.to(guess.device, guess.dtype), target_guess], dim=2)
        background = initial[:, :, 0].to(guess.device, guess.dtype)

        def output_view(state):
            return state[:, :, 1:].reshape(state.shape[0], -1, state.shape[-2], state.shape[-1])

        return state0, output_view, background, {**meta, "final_time": float(batch.metadata.get("final_time", 1.0))}
    if pde in {"reaction_diffusion", "shallow_water"} and full.ndim == 5:
        input_idx = int(batch.metadata.get("input_time_index", 0))
        initial = full[:, :, input_idx].to(guess.device, guess.dtype)
        steps = max(full.shape[2] - input_idx, 2)
        final_guess = guess[:, : initial.shape[1]]
        alpha = torch.linspace(0.0, 1.0, steps, device=guess.device, dtype=guess.dtype).view(1, 1, steps, 1, 1)
        state0 = initial[:, :, None] * (1.0 - alpha) + final_guess[:, :, None] * alpha
        total_t = float(batch.metadata.get("final_time", 1.0 if pde == "shallow_water" else 5.0))
        segment_t = total_t * (steps - 1) / max(full.shape[2] - 1, 1)

        def output_view(state):
            return state[:, :, -1]

        return state0, output_view, initial, {**meta, "final_time": segment_t, "input_time_index": 0}

    state0 = guess

    def output_view(state):
        return state

    return state0, output_view, batch.input_fields.detach(), meta


def _background_view(state: torch.Tensor, batch: PDEBatch) -> torch.Tensor:
    if state.ndim == 5:
        return state[:, :, 0]
    return state[:, : batch.input_fields.shape[1]]


def _obs_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred, target)
    local_mask = mask[: pred.shape[1]].unsqueeze(0).to(pred.device, pred.dtype)
    return (((pred - target) ** 2) * local_mask).sum() / local_mask.sum().clamp_min(1.0)
