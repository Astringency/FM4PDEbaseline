from __future__ import annotations

import time
import warnings

import torch
import torch.nn as nn

from baselines.common.data_adapter import PDEBatch
from baselines.common.metrics import physics_loss_metric

from .base import BaselineModel, LossPlateauStopper, record_optimization_status
from .official import (
    OfficialImportError,
    get_pc_bnn_net_class,
    get_pc_bnn_official_aligned_status,
    official_source_info,
    requested_implementation_mode,
    wrap_official_adapter_error,
)
from .pinn_sparse import (
    STATIC_SPARSE_INVERSE_PDES,
    _physics_weight_metadata,
    _select_physics_loss,
    _single_meta,
    observation_loss_from_batch,
    sparse_forward_observation_loss,
    sparse_inverse_observation_loss,
    sparse_inverse_physics_loss,
)
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
        self.official_aligned = False
        backend = str(self.config.get("official_backend", "auto")).lower()
        implementation_mode = requested_implementation_mode(self.config)
        fallback_reason = ""
        matched_official_setting = _pc_bnn_official_setting(data_spec)
        if implementation_mode == "official" and backend not in {"local", "none"}:
            if not matched_official_setting:
                raise OfficialImportError("strict official PC-BNN requires 2D three-channel shallow-water sparse reconstruction")
            try:
                self.official_net_cls = get_pc_bnn_net_class()
            except Exception as exc:
                raise wrap_official_adapter_error("PC-BNN Net", exc) from exc
            self.set_backend(
                "pc_bnn",
                "pc_bnn",
                fallback_used=False,
                implementation_mode_effective="official",
                implementation_source="pc_bnn_official_net_svgd_adapter",
                official_import_success=True,
                official_reimplementation_success=False,
                official_alignment_level="exact_code",
                official_alignment_notes="Uses the vendored official PC-BNN Net class with the local SVGD/physics adapter.",
                adapter_status="official_code_adapter",
                **official_source_info("pc_bnn"),
            )
        elif matched_official_setting and implementation_mode in {"official_aligned", "official_or_skip", "auto"} and backend in {"auto", "pc_bnn", "official"}:
            try:
                if implementation_mode in {"official_or_skip", "auto"}:
                    try:
                        self.official_net_cls = get_pc_bnn_net_class()
                        self.set_backend(
                            "pc_bnn",
                            "pc_bnn",
                            fallback_used=False,
                            implementation_mode_effective="official",
                            implementation_source="pc_bnn_official_net_svgd_adapter",
                            official_import_success=True,
                            official_reimplementation_success=False,
                            official_alignment_level="exact_code",
                            official_alignment_notes="Uses the vendored official PC-BNN Net class with the local SVGD/physics adapter.",
                            adapter_status="official_code_adapter",
                            **official_source_info("pc_bnn"),
                        )
                        return self
                    except Exception as exc:
                        fallback_reason = f"direct official PC-BNN Net unavailable; using official-aligned reimplementation: {wrap_official_adapter_error('PC-BNN Net', exc)}"
                get_pc_bnn_official_aligned_status()
            except Exception as exc:
                raise wrap_official_adapter_error("PC-BNN official-aligned", exc) from exc
            self.official_net_cls = OfficialAlignedPCBNNNet
            self.official_aligned = True
            self.set_backend(
                "pc_bnn_official_aligned",
                "pc_bnn",
                fallback_used=False,
                warning=fallback_reason,
                implementation_mode_effective="official_aligned",
                implementation_source="pc_bnn_official_aligned_reimplementation",
                official_import_success=False,
                official_reimplementation_success=True,
                official_alignment_level="objective",
                official_alignment_notes=(
                    "Uses the official PC-BNN sparse/noisy flow setting: coordinate-to-(u,v,p)-style "
                    "Swish MLP particles, SVGD posterior updates, observation likelihood, and physics-constrained residual loss."
                ),
                adapter_status="official_aligned_pcbnn_reimplementation",
                **official_source_info("pc_bnn"),
            )
        elif implementation_mode != "adapted" and backend in {"auto", "pc_bnn", "official"}:
            fallback_reason = "official-aligned PC-BNN supports only matched 2D three-channel shallow-water sparse reconstruction"
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
                official_reimplementation_success=False,
                official_alignment_level="local",
                official_alignment_notes="Generic local SVGD neural field; supplement/debug only.",
                adapter_status="local_generic_svgd_pcbnn" if requested_local else "fallback_generic_svgd_pcbnn",
            )
        return self

    def parameter_count(self) -> int:
        proto = self._new_particle(self.coord_dim, self.target_channels)
        return int(self.particles * sum(p.numel() for p in proto.parameters() if p.requires_grad))

    def fit(self, train_loader, val_loader=None):
        return {"status": "per_instance_svgd_particles_no_amortized_fit"}

    def predict(self, batch: PDEBatch):
        if batch.task in {"sparse_forward", "sparse_inverse"}:
            return self._predict_joint_sparse(batch)
        start = time.perf_counter()
        all_means = []
        all_stds = []
        statuses = []
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
            stopper = LossPlateauStopper(self.config)
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
                if stopper.update(torch.stack(losses).mean()):
                    break
            statuses.append(stopper.status())
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
        batch.metadata["posterior_particles"] = int(self.particles)
        batch.metadata["inference_optimization_time"] = time.perf_counter() - start
        record_optimization_status(batch, statuses)
        return mean

    def _predict_joint_sparse(self, batch: PDEBatch) -> torch.Tensor:
        if batch.pde_name.lower() not in STATIC_SPARSE_INVERSE_PDES:
            raise NotImplementedError(
                f"PC-BNN {batch.task} is only enabled for static PDEs, got {batch.pde_name}"
            )
        start = time.perf_counter()
        means: list[torch.Tensor] = []
        stds: list[torch.Tensor] = []
        statuses: list[dict] = []
        steps = int(self.config.get("steps", 2))
        lr = float(self.config.get("lr", 1e-2))
        lam_obs = float(self.config.get("lambda_obs", 1.0))
        for item in range(batch.target_fields.shape[0]):
            coords = batch.coords[item].to(batch.target_fields.device, batch.target_fields.dtype)
            if batch.task == "sparse_inverse":
                unknown_shape = batch.target_fields[item : item + 1].shape
                solution_shape = batch.input_fields[item : item + 1].shape
            else:
                unknown_shape = batch.input_fields[item : item + 1].shape
                solution_shape = batch.target_fields[item : item + 1].shape
            unknown_channels = int(unknown_shape[1])
            solution_channels = int(solution_shape[1])
            particles = [
                self._new_particle(coords.shape[-1], unknown_channels + solution_channels).to(batch.target_fields.device)
                for _ in range(self.particles)
            ]
            for particle_id, particle in enumerate(particles):
                torch.manual_seed(int(self.config.get("seed", 0)) + particle_id)
                for param in particle.parameters():
                    param.data.add_(torch.randn_like(param) * float(self.config.get("init_std", 1e-2)))
            stopper = LossPlateauStopper(self.config)
            for _ in range(max(steps, 0)):
                grads: list[torch.Tensor] = []
                thetas: list[torch.Tensor] = []
                losses: list[torch.Tensor] = []
                for particle in particles:
                    joint = particle(coords).T.reshape(1, unknown_channels + solution_channels, *unknown_shape[2:])
                    unknown = joint[:, :unknown_channels]
                    solution = joint[:, unknown_channels:]
                    obs = (
                        sparse_inverse_observation_loss(solution, batch, item=item)
                        if batch.task == "sparse_inverse"
                        else sparse_forward_observation_loss(unknown, batch, item=item)
                    )
                    loss = lam_obs * obs
                    physics = sparse_inverse_physics_loss(unknown, solution, batch, item, self.config)
                    if torch.isfinite(physics):
                        loss = loss + physics
                    losses.append(loss.detach())
                    grad = torch.autograd.grad(loss, tuple(particle.parameters()), retain_graph=False, create_graph=False)
                    grads.append(torch.cat([value.detach().reshape(-1) for value in grad]))
                    thetas.append(_flatten_params(particle))
                theta = torch.stack(thetas)
                updates = _svgd_descent_direction(theta, torch.stack(grads))
                with torch.no_grad():
                    for particle, update in zip(particles, updates):
                        _assign_flat_params(particle, _flatten_params(particle) - lr * update)
                if stopper.update(torch.stack(losses).mean()):
                    break
            statuses.append(stopper.status())
            predictions: list[torch.Tensor] = []
            with torch.no_grad():
                for particle in particles:
                    joint = particle(coords).T.reshape(1, unknown_channels + solution_channels, *unknown_shape[2:])
                    predictions.append(
                        joint[:, :unknown_channels]
                        if batch.task == "sparse_inverse"
                        else joint[:, unknown_channels:]
                    )
            stack = torch.stack(predictions, dim=0)
            means.append(stack.mean(dim=0))
            stds.append(stack.std(dim=0))
        mean = torch.cat(means, dim=0)
        batch.metadata["predictive_std"] = torch.cat(stds, dim=0)
        batch.metadata["posterior_particles"] = int(self.particles)
        batch.metadata["pc_bnn_joint_field_posterior"] = True
        batch.metadata["inference_optimization_time"] = time.perf_counter() - start
        record_optimization_status(batch, statuses)
        return mean

    def _new_particle(self, coord_dim: int, out_channels: int) -> nn.Module:
        if self.official_net_cls is not None and coord_dim == 2 and out_channels == 3:
            return self.official_net_cls(coord_dim, self.hidden)
        return NeuralField(coord_dim, out_channels, hidden=self.hidden, depth=self.depth)


class OfficialAlignedPCBNNNet(nn.Module):
    def __init__(self, n_feature: int, n_hidden: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(n_feature, n_hidden),
            _Swish(),
            nn.Linear(n_hidden, n_hidden),
            _Swish(),
            nn.Linear(n_hidden, n_hidden),
            _Swish(),
            nn.Linear(n_hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class _Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def _pc_bnn_official_setting(data_spec: dict) -> bool:
    pde = str(data_spec.get("pde", "")).lower()
    target_shape = tuple(data_spec.get("target_shape", ()))
    target_channels = int(data_spec.get("target_channels", 0) or 0)
    coord_dim = max(len(target_shape) - 2, 0)
    return pde == "shallow_water" and coord_dim == 2 and target_channels == 3


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
