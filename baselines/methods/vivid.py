from __future__ import annotations

import time
import warnings

import torch
import torch.nn.functional as F

from baselines.common.data_adapter import PDEBatch
from baselines.common.metrics import physics_loss_metric

from .base import BaselineModel
from .official import official_source_info
from .pinn_sparse import _physics_weight_metadata, _select_physics_loss, observation_loss_from_batch
from .var4d import _assimilation_mode, _background_view, _initial_trajectory, _optimized_state_numel
from .voronoicnn import VoronoiCNNBaseline


class VIVIDBaseline(BaselineModel):
    name = "vivid"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        self.inverse_operator = VoronoiCNNBaseline().build(config.get("inverse_operator", config), data_spec)
        self.optimized_numel = _optimized_state_numel(data_spec)
        self.inverse_operator_trained = False
        if bool(self.config.get("train_inverse_operator", False)):
            self.set_backend(
                "vivid_style",
                "vivid_style",
                fallback_used=False,
                implementation_mode_effective="official_architecture",
                implementation_source="vivid_invobs_structure",
                official_import_success=False,
                adapter_status="vivid_style_trained_inverse_operator",
                **official_source_info("vivid"),
            )
        else:
            self.set_backend(
                "vivid_style",
                "vivid_style",
                fallback_used=False,
                implementation_mode_effective="adapted",
                implementation_source="vivid_style_no_inverse_operator",
                official_import_success=False,
                adapter_status="vivid_style_no_trained_inverse_operator",
                **official_source_info("vivid"),
            )
        return self

    def parameter_count(self) -> int:
        return int(self.optimized_numel + self.inverse_operator.parameter_count())

    def fit(self, train_loader, val_loader=None):
        if bool(self.config.get("train_inverse_operator", False)):
            self.inverse_operator_trained = True
            return self.inverse_operator.fit(train_loader, val_loader)
        return {"status": "per_instance_vivid_no_amortized_inverse_fit"}

    def predict(self, batch: PDEBatch):
        start = time.perf_counter()
        if not batch.metadata.get("supports_trajectory", False):
            mode = _assimilation_mode(batch)
            if mode == "two_level_surrogate":
                warnings.warn("VIVID requested without full trajectory; using a two-level dynamics surrogate.", RuntimeWarning, stacklevel=2)
            else:
                warnings.warn("VIVID requested without full trajectory; using state reconstruction variant.", RuntimeWarning, stacklevel=2)
        with torch.no_grad():
            if self.inverse_operator_trained:
                learned_state = self.inverse_operator.predict(batch)
            else:
                learned_state = batch.metadata.get("voronoi_grid", batch.input_fields).detach()
        state0, output_view, background, dyn_meta = _initial_trajectory(batch)
        batch.metadata["assimilation_mode"] = str(dyn_meta.get("assimilation_mode", _assimilation_mode(batch)))
        # Inject the learned inverse-operator estimate as the terminal state.
        if state0.ndim == 5:
            if learned_state.ndim == 5 and tuple(learned_state.shape) == tuple(state0.shape):
                state0 = learned_state.clone()
            else:
                state0 = state0.clone()
                terminal = learned_state[:, : state0.shape[1]]
                if terminal.ndim == 5:
                    terminal = terminal[:, :, -1]
                state0[:, :, -1] = terminal
        else:
            state0 = learned_state
        state = torch.nn.Parameter(state0.detach().clone())
        steps = int(self.config.get("refine_steps", 2))
        lr = float(self.config.get("lr", 1e-2))
        lam_inv = float(self.config.get("lambda_inverse_operator", 0.25))
        lam_obs = float(self.config.get("lambda_obs", 1.0))
        lam_b = float(self.config.get("lambda_background", 0.05))
        opt = torch.optim.Adam([state], lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            output = output_view(state)
            obs = observation_loss_from_batch(output, batch)
            inv = _mse_aligned(output, learned_state)
            bg = F.mse_loss(_background_view(state, batch), background)
            dyn_meta = {**dyn_meta, **_physics_weight_metadata(self.config)}
            dyn = _select_physics_loss(physics_loss_metric(state, batch.pde_name, dyn_meta), self.config, state)
            if not torch.isfinite(dyn):
                dyn = torch.tensor(0.0, device=state.device)
            loss = lam_obs * obs + lam_inv * inv + lam_b * bg + dyn
            loss.backward()
            opt.step()
        batch.metadata["inference_optimization_time"] = time.perf_counter() - start
        return output_view(state).detach()


def _mse_aligned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape == b.shape:
        return F.mse_loss(a, b)
    if a.ndim == 5 and b.ndim == 4:
        return F.mse_loss(a[:, :, -1], b[:, : a.shape[1]])
    if a.ndim == 4 and b.ndim == 5:
        return F.mse_loss(a[:, : b.shape[1]], b[:, :, -1])
    return F.mse_loss(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)[:, : a.reshape(a.shape[0], -1).shape[1]])
