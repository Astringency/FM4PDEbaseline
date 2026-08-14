from __future__ import annotations

import time

import torch

from baselines.common.data_adapter import PDEBatch
from baselines.common.metrics import physics_loss_metric

from .base import BaselineModel
from .pinn_sparse import (
    STATIC_SPARSE_INVERSE_PDES,
    _physics_weight_metadata,
    _select_physics_loss,
    _smoothness_reg,
    observation_loss_from_batch,
    sparse_forward_observation_loss,
    sparse_inverse_observation_loss,
    sparse_inverse_physics_loss,
)


class PDEOptBaseline(BaselineModel):
    name = "pde_opt"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        self.optimized_numel = int(data_spec["target_numel"]) if "target_numel" in data_spec else int(data_spec["target_channels"])
        self.mark_canonical_math("pde_opt", adapter_status="canonical_pde_constrained_optimization")
        return self

    def parameter_count(self) -> int:
        return int(self.optimized_numel)

    def fit(self, train_loader, val_loader=None):
        return {"status": "per_instance_pde_constrained_optimization"}

    def predict(self, batch: PDEBatch):
        if batch.task == "sparse_inverse":
            return self._predict_sparse_inverse(batch)
        if batch.task == "sparse_forward":
            return self._predict_sparse_forward(batch)
        start = time.perf_counter()
        pred = torch.nn.Parameter(batch.metadata.get("voronoi_grid", batch.input_fields).detach().clone())
        steps = int(self.config.get("steps", 3))
        lam_obs = float(self.config.get("lambda_obs", 1.0))
        lam_reg = float(self.config.get("lambda_reg", 1e-5))
        opt_name = str(self.config.get("optimizer", "adam")).lower()
        if opt_name == "lbfgs":
            opt = torch.optim.LBFGS([pred], lr=float(self.config.get("lr", 1.0)), max_iter=steps)

            def closure():
                opt.zero_grad(set_to_none=True)
                loss = self._objective(pred, batch, lam_obs, lam_reg)
                loss.backward()
                return loss

            opt.step(closure)
        else:
            opt = torch.optim.Adam([pred], lr=float(self.config.get("lr", 5e-2)))
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                loss = self._objective(pred, batch, lam_obs, lam_reg)
                loss.backward()
                opt.step()
        batch.metadata["inference_optimization_time"] = time.perf_counter() - start
        return pred.detach()

    def _predict_sparse_inverse(self, batch: PDEBatch):
        pde = batch.pde_name.lower()
        if pde not in STATIC_SPARSE_INVERSE_PDES:
            raise NotImplementedError(f"PDE-Opt sparse_inverse is only enabled for static PDEs, got {batch.pde_name}")
        start = time.perf_counter()
        solution0 = batch.metadata.get("voronoi_grid", batch.input_fields).detach().clone()
        unknown0 = torch.zeros_like(batch.target_fields)
        solution = torch.nn.Parameter(solution0)
        unknown = torch.nn.Parameter(unknown0)
        steps = int(self.config.get("steps", 3))
        lr = float(self.config.get("lr", 5e-2))
        lam_obs = float(self.config.get("lambda_obs", 1.0))
        lam_reg = float(self.config.get("lambda_reg", 1e-5))
        opt_name = str(self.config.get("optimizer", "adam")).lower()
        params = [unknown, solution]
        if opt_name == "lbfgs":
            opt = torch.optim.LBFGS(params, lr=lr, max_iter=steps)

            def closure():
                opt.zero_grad(set_to_none=True)
                loss = self._sparse_inverse_objective(unknown, solution, batch, lam_obs, lam_reg)
                loss.backward()
                return loss

            opt.step(closure)
        else:
            opt = torch.optim.Adam(params, lr=lr)
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                loss = self._sparse_inverse_objective(unknown, solution, batch, lam_obs, lam_reg)
                loss.backward()
                opt.step()
        batch.metadata["inference_optimization_time"] = time.perf_counter() - start
        return unknown.detach()

    def _predict_sparse_forward(self, batch: PDEBatch):
        pde = batch.pde_name.lower()
        if pde not in STATIC_SPARSE_INVERSE_PDES:
            raise NotImplementedError(f"PDE-Opt sparse_forward is only enabled for static PDEs, got {batch.pde_name}")
        start = time.perf_counter()
        unknown0 = batch.metadata.get("voronoi_grid", batch.input_fields).detach().clone()
        solution0 = torch.zeros_like(batch.target_fields)
        unknown = torch.nn.Parameter(unknown0)
        solution = torch.nn.Parameter(solution0)
        steps = int(self.config.get("steps", 3))
        lr = float(self.config.get("lr", 5e-2))
        lam_obs = float(self.config.get("lambda_obs", 1.0))
        lam_reg = float(self.config.get("lambda_reg", 1e-5))
        opt_name = str(self.config.get("optimizer", "adam")).lower()
        params = [unknown, solution]
        if opt_name == "lbfgs":
            opt = torch.optim.LBFGS(params, lr=lr, max_iter=steps)

            def closure():
                opt.zero_grad(set_to_none=True)
                loss = self._sparse_forward_objective(unknown, solution, batch, lam_obs, lam_reg)
                loss.backward()
                return loss

            opt.step(closure)
        else:
            opt = torch.optim.Adam(params, lr=lr)
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                loss = self._sparse_forward_objective(unknown, solution, batch, lam_obs, lam_reg)
                loss.backward()
                opt.step()
        batch.metadata["inference_optimization_time"] = time.perf_counter() - start
        return solution.detach()

    def _objective(self, pred, batch, lam_obs, lam_reg):
        obs = observation_loss_from_batch(pred, batch)
        meta = {"input_fields": batch.input_fields, "full_tensor": batch.full_tensor, "task": batch.task, **batch.metadata}
        meta.update(_physics_weight_metadata(self.config))
        physics_value = _select_physics_loss(physics_loss_metric(pred, batch.pde_name, meta), self.config, pred)
        pde = physics_value if torch.isfinite(physics_value) else torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        reg = (pred[..., 1:, :] - pred[..., :-1, :]).pow(2).mean() + (pred[..., :, 1:] - pred[..., :, :-1]).pow(2).mean()
        return lam_obs * obs + pde + lam_reg * reg

    def _sparse_inverse_objective(self, unknown, solution, batch, lam_obs, lam_reg):
        obs = sparse_inverse_observation_loss(solution, batch)
        physics_value = sparse_inverse_physics_loss(unknown, solution, batch, item=None, config=self.config)
        pde = physics_value if torch.isfinite(physics_value) else torch.tensor(0.0, device=unknown.device, dtype=unknown.dtype)
        reg = _smoothness_reg(unknown) + _smoothness_reg(solution)
        return lam_obs * obs + pde + lam_reg * reg

    def _sparse_forward_objective(self, unknown, solution, batch, lam_obs, lam_reg):
        obs = sparse_forward_observation_loss(unknown, batch)
        physics_value = sparse_inverse_physics_loss(unknown, solution, batch, item=None, config=self.config)
        pde = physics_value if torch.isfinite(physics_value) else torch.tensor(0.0, device=unknown.device, dtype=unknown.dtype)
        reg = _smoothness_reg(unknown) + _smoothness_reg(solution)
        return lam_obs * obs + pde + lam_reg * reg
