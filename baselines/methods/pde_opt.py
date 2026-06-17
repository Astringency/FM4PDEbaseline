from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from baselines.common.data_adapter import PDEBatch
from baselines.common.metrics import pde_residual_metric

from .base import BaselineModel


class PDEOptBaseline(BaselineModel):
    name = "pde_opt"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        self.optimized_numel = int(data_spec["target_numel"]) if "target_numel" in data_spec else int(data_spec["target_channels"])
        return self

    def parameter_count(self) -> int:
        return int(self.optimized_numel)

    def fit(self, train_loader, val_loader=None):
        return {"status": "per_instance_pde_constrained_optimization"}

    def predict(self, batch: PDEBatch):
        start = time.perf_counter()
        pred = torch.nn.Parameter(batch.metadata.get("voronoi_grid", batch.input_fields).detach().clone())
        steps = int(self.config.get("steps", 3))
        lam_obs = float(self.config.get("lambda_obs", 1.0))
        lam_pde = float(self.config.get("lambda_pde", 0.01))
        lam_reg = float(self.config.get("lambda_reg", 1e-5))
        opt_name = str(self.config.get("optimizer", "adam")).lower()
        if opt_name == "lbfgs":
            opt = torch.optim.LBFGS([pred], lr=float(self.config.get("lr", 1.0)), max_iter=steps)

            def closure():
                opt.zero_grad(set_to_none=True)
                loss = self._objective(pred, batch, lam_obs, lam_pde, lam_reg)
                loss.backward()
                return loss

            opt.step(closure)
        else:
            opt = torch.optim.Adam([pred], lr=float(self.config.get("lr", 5e-2)))
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                loss = self._objective(pred, batch, lam_obs, lam_pde, lam_reg)
                loss.backward()
                opt.step()
        batch.metadata["inference_optimization_time"] = time.perf_counter() - start
        return pred.detach()

    def _objective(self, pred, batch, lam_obs, lam_pde, lam_reg):
        if batch.mask is not None:
            obs = (((pred - batch.target_fields) ** 2) * batch.mask.unsqueeze(0)).mean()
        else:
            obs = F.mse_loss(pred, batch.target_fields)
        residual = pde_residual_metric(
            pred,
            batch.pde_name,
            {"input_fields": batch.input_fields, "full_tensor": batch.full_tensor, "task": batch.task, **batch.metadata},
        )
        pde = residual if torch.isfinite(residual) else torch.tensor(0.0, device=pred.device)
        reg = (pred[..., 1:, :] - pred[..., :-1, :]).pow(2).mean() + (pred[..., :, 1:] - pred[..., :, :-1]).pow(2).mean()
        return lam_obs * obs + lam_pde * pde + lam_reg * reg
