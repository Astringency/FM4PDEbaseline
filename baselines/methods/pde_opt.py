from __future__ import annotations

import torch

from baselines.common.data_adapter import PDEBatch
from baselines.common.metrics import physics_loss_metric

from .base import BaselineModel, record_optimization_status, run_per_instance_optimizer
from .pinn_sparse import (
    STATIC_SPARSE_INVERSE_PDES,
    _synchronized_perf_counter,
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
        target_numel = int(data_spec.get("target_numel", data_spec["target_channels"]))
        input_numel = int(data_spec.get("input_numel", data_spec["input_channels"]))
        self.optimized_numel = (
            target_numel + input_numel
            if str(data_spec.get("task", "")) in {"sparse_inverse", "sparse_forward"}
            else target_numel
        )
        self.mark_canonical_math("pde_opt", adapter_status="canonical_pde_constrained_optimization")
        return self

    def parameter_count(self) -> int:
        return int(self.optimized_numel)

    def parameter_storage_count(self) -> int:
        return self.parameter_count()

    def fit(self, train_loader, val_loader=None):
        return {"status": "per_instance_pde_constrained_optimization"}

    def predict(self, batch: PDEBatch):
        if batch.task == "sparse_inverse":
            return self._predict_sparse_inverse(batch)
        if batch.task == "sparse_forward":
            return self._predict_sparse_forward(batch)
        if batch.task in {"sparse_solution", "sparse_reconstruction"}:
            raise RuntimeError(
                "PDE-Opt is disabled for the sensor-only sparse_solution protocol because its PDE objective "
                "requires hidden source/coefficient/initial fields. Use a separately named equal-context protocol."
            )
        start = _synchronized_perf_counter(batch)
        pred = torch.nn.Parameter(batch.metadata.get("voronoi_grid", batch.input_fields).detach().clone())
        steps = int(self.config.get("steps", 3))
        lam_obs = float(self.config.get("lambda_obs", 1.0))
        lam_reg = float(self.config.get("lambda_reg", 1e-5))
        opt_name = str(self.config.get("optimizer", "adam")).lower()
        opt = (
            torch.optim.LBFGS([pred], lr=float(self.config.get("lr", 1.0)), max_iter=steps)
            if opt_name == "lbfgs"
            else torch.optim.Adam([pred], lr=float(self.config.get("lr", 5e-2)))
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = self._objective(pred, batch, lam_obs, lam_reg)
            loss.backward()
            return loss

        status = run_per_instance_optimizer(opt, closure, steps, self.config)
        batch.metadata["inference_optimization_time"] = _synchronized_perf_counter(batch) - start
        record_optimization_status(batch, [status])
        return pred.detach()

    def _predict_sparse_inverse(self, batch: PDEBatch):
        pde = batch.pde_name.lower()
        if pde not in STATIC_SPARSE_INVERSE_PDES:
            raise NotImplementedError(f"PDE-Opt sparse_inverse is only enabled for static PDEs, got {batch.pde_name}")
        start = _synchronized_perf_counter(batch)
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
        opt = torch.optim.LBFGS(params, lr=lr, max_iter=steps) if opt_name == "lbfgs" else torch.optim.Adam(params, lr=lr)

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = self._sparse_inverse_objective(unknown, solution, batch, lam_obs, lam_reg)
            loss.backward()
            return loss

        status = run_per_instance_optimizer(opt, closure, steps, self.config)
        batch.metadata["inference_optimization_time"] = _synchronized_perf_counter(batch) - start
        record_optimization_status(batch, [status])
        return unknown.detach()

    def _predict_sparse_forward(self, batch: PDEBatch):
        pde = batch.pde_name.lower()
        if pde not in STATIC_SPARSE_INVERSE_PDES:
            raise NotImplementedError(f"PDE-Opt sparse_forward is only enabled for static PDEs, got {batch.pde_name}")
        start = _synchronized_perf_counter(batch)
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
        opt = torch.optim.LBFGS(params, lr=lr, max_iter=steps) if opt_name == "lbfgs" else torch.optim.Adam(params, lr=lr)

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = self._sparse_forward_objective(unknown, solution, batch, lam_obs, lam_reg)
            loss.backward()
            return loss

        status = run_per_instance_optimizer(opt, closure, steps, self.config)
        batch.metadata["inference_optimization_time"] = _synchronized_perf_counter(batch) - start
        record_optimization_status(batch, [status])
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
