from __future__ import annotations

import time
import warnings

import torch
import torch.nn as nn

from baselines.common.data_adapter import PDEBatch
from baselines.common.metrics import physics_loss_metric

from .base import BaselineModel
from .official import OfficialImportError, get_pc_bnn_net_class, official_source_info, requested_implementation_mode
from .pinn_sparse import _physics_weight_metadata, _select_physics_loss, _single_meta, observation_loss_from_batch
from .shared import NeuralField


class PCBNNBaseline(BaselineModel):
    name = "pc_bnn"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        self.particles = int(self.config.get("particles", 3))
        self.coord_dim = len(tuple(data_spec["target_shape"])[2:])
        self.target_channels = int(data_spec["target_channels"])
        self.hidden = int(self.config.get("hidden", 64))
        self.depth = int(self.config.get("depth", 4))
        self.official_net_cls = None
        backend = str(self.config.get("official_backend", "auto")).lower()
        implementation_mode = requested_implementation_mode(self.config)
        fallback_reason = ""
        if self.coord_dim == 2 and self.target_channels == 3 and implementation_mode != "adapted" and backend in {"auto", "pc_bnn", "official"}:
            try:
                self.official_net_cls = get_pc_bnn_net_class()
                self.set_backend(
                    "pc_bnn",
                    "pc_bnn",
                    fallback_used=False,
                    implementation_mode_effective="official",
                    implementation_source="pc_bnn",
                    official_import_success=True,
                    adapter_status="official_code_adapter",
                    **official_source_info("pc_bnn"),
                )
            except OfficialImportError as exc:
                fallback_reason = f"pc_bnn unavailable: {exc}"
                if backend in {"pc_bnn", "official"}:
                    warnings.warn(f"PC-BNN official Net unavailable, using local particle fallback: {exc}", RuntimeWarning, stacklevel=2)
        elif implementation_mode != "adapted" and backend in {"auto", "pc_bnn", "official"}:
            fallback_reason = "official PC-BNN adapter only supports 2D three-channel targets"
        if self.official_net_cls is None:
            requested_local = backend in {"local", "none"} or implementation_mode == "adapted"
            self.set_backend(
                "local",
                "local" if requested_local else ("official" if backend == "official" else backend),
                fallback_used=not requested_local,
                warning=fallback_reason,
                implementation_mode_effective="adapted",
                implementation_source="local_svgd_particle_field",
                official_import_success=False,
                adapter_status="local_adapted" if requested_local else "fallback_adapted",
            )
        return self

    def parameter_count(self) -> int:
        proto = self._new_particle(self.coord_dim, self.target_channels)
        return int(self.particles * sum(p.numel() for p in proto.parameters() if p.requires_grad))

    def fit(self, train_loader, val_loader=None):
        return {"status": "per_instance_svgd_particles_no_amortized_fit"}

    def predict(self, batch: PDEBatch):
        start = time.perf_counter()
        all_means = []
        all_stds = []
        steps = int(self.config.get("steps", 2))
        lr = float(self.config.get("lr", 1e-2))
        hidden = self.hidden
        depth = self.depth
        lam_obs = float(self.config.get("lambda_obs", 1.0))
        for item in range(batch.target_fields.shape[0]):
            coords = batch.coords[item].to(batch.target_fields.device, batch.target_fields.dtype)
            target = batch.target_fields[item : item + 1]
            particles = [
                self._new_particle(coords.shape[-1], target.shape[1]).to(target.device)
                for _ in range(self.particles)
            ]
            for particle_id, particle in enumerate(particles):
                torch.manual_seed(int(self.config.get("seed", 0)) + particle_id)
                for param in particle.parameters():
                    param.data.add_(torch.randn_like(param) * float(self.config.get("init_std", 1e-2)))
            for _ in range(max(steps, 0)):
                losses = []
                grads = []
                thetas = []
                for particle in particles:
                    pred = particle(coords).T.reshape_as(target)
                    loss = lam_obs * observation_loss_from_batch(pred, batch, item=item)
                    meta = _single_meta(batch, item)
                    meta.update(_physics_weight_metadata(self.config))
                    physics_value = _select_physics_loss(physics_loss_metric(pred, batch.pde_name, meta), self.config, pred)
                    if torch.isfinite(physics_value):
                        loss = loss + physics_value
                    grad = torch.autograd.grad(loss, tuple(particle.parameters()), retain_graph=False, create_graph=False)
                    losses.append(loss.detach())
                    grads.append(torch.cat([g.detach().reshape(-1) for g in grad]))
                    thetas.append(_flatten_params(particle))
                theta = torch.stack(thetas)
                grad_loss = torch.stack(grads)
                updates = _svgd_descent_direction(theta, grad_loss)
                with torch.no_grad():
                    for particle, update in zip(particles, updates):
                        _assign_flat_params(particle, _flatten_params(particle) - lr * update)
            preds = []
            with torch.no_grad():
                for particle in particles:
                    preds.append(particle(coords).T.reshape_as(target))
            stack = torch.stack(preds, dim=0)
            all_means.append(stack.mean(dim=0))
            all_stds.append(stack.std(dim=0))
        mean = torch.cat(all_means, dim=0)
        std = torch.cat(all_stds, dim=0)
        batch.metadata["predictive_std"] = std
        batch.metadata["inference_optimization_time"] = time.perf_counter() - start
        return mean

    def _new_particle(self, coord_dim: int, out_channels: int) -> nn.Module:
        if self.official_net_cls is not None and coord_dim == 2 and out_channels == 3:
            return self.official_net_cls(coord_dim, self.hidden)
        return NeuralField(coord_dim, out_channels, hidden=self.hidden, depth=self.depth)


def _flatten_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _assign_flat_params(model: torch.nn.Module, vector: torch.Tensor) -> None:
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.copy_(vector[offset : offset + numel].reshape_as(param))
        offset += numel


def _svgd_descent_direction(theta: torch.Tensor, grad_loss: torch.Tensor) -> torch.Tensor:
    particles = theta.shape[0]
    if particles == 1:
        return grad_loss
    sqdist = torch.cdist(theta, theta, p=2).pow(2)
    median = torch.median(sqdist.detach())
    bandwidth = median / torch.log(torch.tensor(float(particles + 1), device=theta.device, dtype=theta.dtype))
    bandwidth = bandwidth.clamp_min(1e-6)
    kernel = torch.exp(-sqdist / bandwidth)
    attractive = kernel.T @ grad_loss / particles
    repulsive = torch.zeros_like(theta)
    for i in range(particles):
        diff = theta - theta[i]
        repulsive[i] = (2.0 / bandwidth) * (kernel[:, i].unsqueeze(1) * diff).mean(dim=0)
    return attractive - repulsive
