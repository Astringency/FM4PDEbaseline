from __future__ import annotations

import time
import warnings

import torch
import torch.nn.functional as F

from baselines.common.data_adapter import PDEBatch
from baselines.common.metrics import pde_residual_metric

from .base import BaselineModel
from .var4d import _background_view, _initial_trajectory, _obs_loss, _optimized_state_numel
from .voronoicnn import VoronoiCNNBaseline


class VIVIDBaseline(BaselineModel):
    name = "vivid"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        self.inverse_operator = VoronoiCNNBaseline().build(config.get("inverse_operator", config), data_spec)
        self.optimized_numel = _optimized_state_numel(data_spec)
        return self

    def parameter_count(self) -> int:
        return int(self.optimized_numel + self.inverse_operator.parameter_count())

    def fit(self, train_loader, val_loader=None):
        return self.inverse_operator.fit(train_loader, val_loader)

    def predict(self, batch: PDEBatch):
        start = time.perf_counter()
        if not batch.metadata.get("supports_trajectory", False):
            warnings.warn("VIVID requested without full trajectory; using full-space state reconstruction variant.", RuntimeWarning, stacklevel=2)
        with torch.no_grad():
            learned_state = self.inverse_operator.predict(batch)
        state0, output_view, background, dyn_meta = _initial_trajectory(batch)
        # Inject the learned inverse-operator estimate as the terminal state.
        if state0.ndim == 5:
            state0 = state0.clone()
            state0[:, :, -1] = learned_state[:, : state0.shape[1]]
        else:
            state0 = learned_state
        state = torch.nn.Parameter(state0.detach().clone())
        steps = int(self.config.get("refine_steps", 2))
        lr = float(self.config.get("lr", 1e-2))
        lam_inv = float(self.config.get("lambda_inverse_operator", 0.25))
        lam_obs = float(self.config.get("lambda_obs", 1.0))
        lam_b = float(self.config.get("lambda_background", 0.05))
        lam_dyn = float(self.config.get("lambda_dynamics", self.config.get("lambda_pde", 0.01)))
        opt = torch.optim.Adam([state], lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            output = output_view(state)
            obs = _obs_loss(output, batch.target_fields, batch.mask)
            inv = F.mse_loss(output, learned_state)
            bg = F.mse_loss(_background_view(state, batch), background)
            dyn = pde_residual_metric(state, batch.pde_name, dyn_meta)
            if not torch.isfinite(dyn):
                dyn = torch.tensor(0.0, device=state.device)
            loss = lam_obs * obs + lam_inv * inv + lam_b * bg + lam_dyn * dyn
            loss.backward()
            opt.step()
        batch.metadata["inference_optimization_time"] = time.perf_counter() - start
        return output_view(state).detach()
