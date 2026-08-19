from __future__ import annotations

import time

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
        pde = str(data_spec.get("pde", "")).lower()
        if pde != "burger":
            raise ValueError("The audited Var4D adapter is scoped only to Burgers sparse trajectory reconstruction")
        self.optimized_numel = _optimized_state_numel(data_spec)
        self.set_backend(
            "var4d_burgers_weak_constraint",
            "4dvar_mathematical_reference",
            implementation_mode_effective="adapted",
            implementation_source="local_burgers_weak_constraint_trajectory_optimization",
            official_import_success=False,
            official_reimplementation_success=False,
            official_alignment_level="local",
            official_alignment_notes=(
                "Optimizes every Burgers trajectory value with Adam under observation, background, and soft PDE-residual "
                "losses. This is not canonical strong-constraint 4D-Var, which optimizes an initial/control state through "
                "a dynamical propagator."
            ),
            adapter_status="var4d_style_burgers_weak_constraint_adapted",
        )
        return self

    def parameter_count(self) -> int:
        return int(self.optimized_numel)

    def fit(self, train_loader, val_loader=None):
        return {"status": "per_instance_4dvar"}

    def predict(self, batch: PDEBatch):
        start = time.perf_counter()
        if batch.pde_name.lower() != "burger" or not batch.metadata.get("supports_trajectory", False):
            raise ValueError("Var4D requires a loaded Burgers T x X trajectory")
        state0, output_view, background, dyn_meta = _initial_trajectory(batch)
        batch.metadata["assimilation_mode"] = str(dyn_meta.get("assimilation_mode", _assimilation_mode(batch)))
        batch.metadata["assimilation_background_source"] = str(dyn_meta.get("assimilation_background_source", "task_background"))
        batch.metadata["assimilation_uses_hidden_truth"] = bool(dyn_meta.get("assimilation_uses_hidden_truth", False))
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


def _optimized_state_numel(data_spec: dict) -> int:
    target_shape = tuple(data_spec.get("target_shape", ()))
    return int(torch.tensor(target_shape[1:]).prod().item()) if len(target_shape) > 1 else 0


def _initial_trajectory(batch: PDEBatch):
    pde = batch.pde_name.lower()
    if pde != "burger" or batch.target_fields.ndim != 4:
        raise ValueError("Var4D/VIVID trajectory initialization supports only Burgers [B,C,T,X]")
    guess = batch.metadata.get("voronoi_grid", batch.input_fields).detach()
    mode = _assimilation_mode(batch)
    meta = {"input_fields": batch.input_fields, "task": "trajectory", **batch.metadata, "assimilation_mode": mode}
    expected_shape = tuple(batch.target_fields.shape)
    if guess.ndim != 4 or tuple(guess.shape) != expected_shape:
        raise ValueError(
            "Burgers Var4D requires an observation-derived full T x X background "
            f"with shape {expected_shape}; got {tuple(guess.shape)}"
        )
    # The complete state, including t=0, must be inferred from the sparse
    # observations. Never seed it from full_tensor/initial_1d.
    state0 = guess.clone()
    background = state0.detach().clone()

    def output_view(state):
        return state

    hidden_truth_keys = {
        "full_tensor",
        "full_trajectory",
        "original_input_fields",
        "observation_source_fields",
        "observed_solution_fields",
        "background_fields",
        "solution_fields",
        "source_fields",
        "coeff_fields",
        "initial_1d",
    }
    observation_meta = {key: value for key, value in meta.items() if key not in hidden_truth_keys}
    observation_meta.update(
        {
            "input_fields": background,
            "initial_1d": background[:, 0, 0, :],
            "final_time": float(batch.metadata.get("final_time", 1.0)),
            "input_time_index": 0,
            "assimilation_background_source": "voronoi_grid_from_sparse_observations",
            "assimilation_uses_hidden_truth": False,
        }
    )
    return state0, output_view, background, observation_meta


def _background_view(state: torch.Tensor, batch: PDEBatch) -> torch.Tensor:
    if batch.pde_name.lower() == "burger" and state.ndim == 4:
        return state
    raise ValueError("Var4D/VIVID background view supports only Burgers [B,C,T,X]")


def _assimilation_mode(batch: PDEBatch) -> str:
    if batch.pde_name.lower() == "burger" and batch.target_fields.ndim == 4 and batch.target_fields.shape[-2] > 2:
        return "full_trajectory"
    raise ValueError("Var4D/VIVID assimilation mode supports only a complete Burgers trajectory")
