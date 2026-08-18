from __future__ import annotations

import time
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.data_adapter import PDEBatch
from baselines.common.metrics import physics_loss_metric

from .base import BaselineModel, LossPlateauStopper, record_optimization_status
from .official import official_source_info
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
@dataclass(frozen=True)
class PCBNNPosteriorResult:
    mean: torch.Tensor
    std: torch.Tensor
    samples: torch.Tensor
    noise_precision: float
    status: dict


class _SparseTaskAdapter:
    """Task seam between the PC-BNN posterior engine and FM4PDE semantics."""

    def observation_loss(self, unknown: torch.Tensor, solution: torch.Tensor, batch: PDEBatch, item: int) -> torch.Tensor:
        raise NotImplementedError

    def result(self, unknown: torch.Tensor, solution: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def physics_loss(
        self,
        unknown: torch.Tensor,
        solution: torch.Tensor,
        batch: PDEBatch,
        item: int,
        config: dict,
    ) -> torch.Tensor:
        return sparse_inverse_physics_loss(unknown, solution, batch, item, config)


class _SparseForwardAdapter(_SparseTaskAdapter):
    def observation_loss(self, unknown, solution, batch, item):
        return sparse_forward_observation_loss(unknown, batch, item=item)

    def result(self, unknown, solution):
        return solution


class _SparseInverseAdapter(_SparseTaskAdapter):
    def observation_loss(self, unknown, solution, batch, item):
        return sparse_inverse_observation_loss(solution, batch, item=item)

    def result(self, unknown, solution):
        return unknown


class PCBNNBaseline(BaselineModel):
    name = "pc_bnn"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        self.particles = int(self.config.get("particles", 3))
        if self.particles <= 0:
            raise ValueError("PC-BNN requires at least one posterior particle")
        self.coord_dim = len(tuple(data_spec["target_shape"])[2:])
        self.target_channels = int(data_spec["target_channels"])
        self.hidden = int(self.config.get("hidden", 64))
        # The official network has three Swish hidden layers.  Task adaptation
        # changes only the output width needed for (a,u).
        self.depth = 3
        task = str(data_spec.get("task", ""))
        adapter_status = (
            "pc_bnn_adapted_static_pde"
            if task in {"sparse_forward", "sparse_inverse"}
            else "pc_bnn_adapted_reconstruction"
        )
        self.set_backend(
            "pc_bnn_adapted",
            "pc_bnn",
            fallback_used=False,
            implementation_mode_effective="adapted",
            implementation_source="official_pcbnn_posterior_with_fm4pde_task_adapter",
            official_import_success=False,
            official_reimplementation_success=False,
            official_alignment_level="algorithm",
            official_alignment_notes=(
                "Preserves the official three-layer Swish particle architecture, Student-t weight prior, "
                "Gamma noise-precision prior, SVGD update, observation likelihood, and physics likelihood; "
                "the output fields and PDE residual are adapted to FM4PDE static tasks."
            ),
            adapter_status=adapter_status,
            **official_source_info("pc_bnn"),
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
        all_samples = []
        noise_precisions = []
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
                    obs = observation_loss_from_batch(pred, batch, item=item)
                    meta = _single_meta(batch, item)
                    meta.update(_physics_weight_metadata(self.config))
                    physics_value = _select_physics_loss(physics_loss_metric(pred, batch.pde_name, meta), self.config, pred)
                    loss = _negative_log_posterior(
                        particle,
                        lam_obs * obs,
                        physics_value,
                        _observation_count(batch, item),
                        self.config,
                    )
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
                        particle.clamp_noise_precision()
                if stopper.update(torch.stack(losses).mean()):
                    break
            preds = []
            with torch.no_grad():
                for particle in particles:
                    preds.append(particle(coords).T.reshape_as(target))
            result = _posterior_result(preds, particles, stopper.status())
            statuses.append(result.status)
            all_means.append(result.mean)
            all_stds.append(result.std)
            all_samples.append(result.samples)
            noise_precisions.append(result.noise_precision)
        mean = torch.cat(all_means, dim=0)
        std = torch.cat(all_stds, dim=0)
        batch.metadata["predictive_std"] = std
        batch.metadata["posterior_samples"] = torch.stack(all_samples, dim=0)
        batch.metadata["posterior_noise_precision"] = noise_precisions
        batch.metadata["pc_bnn_posterior_objective"] = _POSTERIOR_OBJECTIVE
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
        adapter: _SparseTaskAdapter = (
            _SparseInverseAdapter() if batch.task == "sparse_inverse" else _SparseForwardAdapter()
        )
        means: list[torch.Tensor] = []
        stds: list[torch.Tensor] = []
        posterior_samples: list[torch.Tensor] = []
        noise_precisions: list[float] = []
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
                    unknown = _transform_unknown(joint[:, :unknown_channels], batch.pde_name, self.config)
                    solution = joint[:, unknown_channels:]
                    obs = adapter.observation_loss(unknown, solution, batch, item)
                    physics = adapter.physics_loss(unknown, solution, batch, item, self.config)
                    loss = _negative_log_posterior(
                        particle,
                        lam_obs * obs,
                        physics,
                        _observation_count(batch, item),
                        self.config,
                    )
                    losses.append(loss.detach())
                    grad = torch.autograd.grad(loss, tuple(particle.parameters()), retain_graph=False, create_graph=False)
                    grads.append(torch.cat([value.detach().reshape(-1) for value in grad]))
                    thetas.append(_flatten_params(particle))
                theta = torch.stack(thetas)
                updates = _svgd_descent_direction(theta, torch.stack(grads))
                with torch.no_grad():
                    for particle, update in zip(particles, updates):
                        _assign_flat_params(particle, _flatten_params(particle) - lr * update)
                        particle.clamp_noise_precision()
                if stopper.update(torch.stack(losses).mean()):
                    break
            predictions: list[torch.Tensor] = []
            with torch.no_grad():
                for particle in particles:
                    joint = particle(coords).T.reshape(1, unknown_channels + solution_channels, *unknown_shape[2:])
                    unknown = _transform_unknown(joint[:, :unknown_channels], batch.pde_name, self.config)
                    solution = joint[:, unknown_channels:]
                    predictions.append(adapter.result(unknown, solution))
            result = _posterior_result(predictions, particles, stopper.status())
            statuses.append(result.status)
            means.append(result.mean)
            stds.append(result.std)
            posterior_samples.append(result.samples)
            noise_precisions.append(result.noise_precision)
        mean = torch.cat(means, dim=0)
        batch.metadata["predictive_std"] = torch.cat(stds, dim=0)
        batch.metadata["posterior_samples"] = torch.stack(posterior_samples, dim=0)
        batch.metadata["posterior_noise_precision"] = noise_precisions
        batch.metadata["posterior_particles"] = int(self.particles)
        batch.metadata["pc_bnn_joint_field_posterior"] = True
        batch.metadata["pc_bnn_posterior_objective"] = _POSTERIOR_OBJECTIVE
        batch.metadata["pc_bnn_task_adapter"] = type(adapter).__name__
        batch.metadata["inference_optimization_time"] = time.perf_counter() - start
        record_optimization_status(batch, statuses)
        return mean

    def _new_particle(self, coord_dim: int, out_channels: int) -> nn.Module:
        return PCBNNParticle(
            coord_dim,
            self.hidden,
            out_channels,
            initial_noise_precision=float(self.config.get("initial_noise_precision", 100.0)),
        )


class PCBNNParticle(nn.Module):
    """Official PC-BNN Swish MLP with an adapted output width and noise posterior."""

    def __init__(self, n_feature: int, n_hidden: int, out_channels: int, *, initial_noise_precision: float) -> None:
        super().__init__()
        if initial_noise_precision <= 0:
            raise ValueError("initial_noise_precision must be positive")
        self.features = nn.Sequential(
            nn.Linear(n_feature, n_hidden),
            _Swish(),
            nn.Linear(n_hidden, n_hidden),
            _Swish(),
            nn.Linear(n_hidden, n_hidden),
            _Swish(),
            nn.Linear(n_hidden, out_channels),
        )
        self.log_beta = nn.Parameter(torch.tensor(math.log(initial_noise_precision), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def clamp_noise_precision(self) -> None:
        self.log_beta.clamp_(min=-12.0, max=20.0)


class _Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


_POSTERIOR_OBJECTIVE = "gaussian_observation+student_t_weight_prior+gamma_noise_precision+pde_likelihood"


def _negative_log_posterior(
    particle: PCBNNParticle,
    observation_mse: torch.Tensor,
    physics_loss: torch.Tensor,
    observation_count: int,
    config: dict,
) -> torch.Tensor:
    """Negative log posterior following the official PC-BNN hierarchy.

    The observation term uses the learned Gaussian precision.  The weight
    prior is the Student-t marginal induced by the official Gamma hierarchy,
    and ``log_beta`` retains the official Gamma prior.  PDE-specific residuals
    are supplied by the task Adapter rather than embedded in the engine.
    """
    count = max(int(observation_count), 1)
    log_beta = particle.log_beta
    beta = log_beta.exp()
    observation_nll = 0.5 * beta * float(count) * observation_mse - 0.5 * float(count) * log_beta

    weight_shape = float(config.get("weight_prior_shape", 1.0))
    weight_rate = max(float(config.get("weight_prior_rate", 0.05)), 1e-12)
    weight_prior = torch.zeros((), device=log_beta.device, dtype=log_beta.dtype)
    for parameter in particle.features.parameters():
        weight_prior = weight_prior + torch.log1p(0.5 / weight_rate * parameter.pow(2)).sum()
    weight_prior = (weight_shape + 0.5) * weight_prior

    beta_shape = float(config.get("beta_prior_shape", 2.0))
    beta_rate = float(config.get("beta_prior_rate", 1e-6))
    beta_prior = beta_rate * beta - (beta_shape - 1.0) * log_beta
    physics = physics_loss if torch.isfinite(physics_loss) else torch.zeros_like(observation_nll)
    return observation_nll + physics + weight_prior + beta_prior


def _observation_count(batch: PDEBatch, item: int) -> int:
    if batch.obs_values is not None:
        return int(batch.obs_values[item].numel())
    if batch.mask is not None:
        mask = batch.mask[item] if batch.mask.ndim == batch.input_fields.ndim else batch.mask
        return int(mask.sum().item())
    return int(batch.input_fields[item].numel())


def _transform_unknown(unknown: torch.Tensor, pde_name: str, config: dict) -> torch.Tensor:
    if str(pde_name).lower() == "darcy":
        floor = float(config.get("coefficient_floor", 1e-6))
        return F.softplus(unknown) + floor
    return unknown


def _posterior_result(
    predictions: list[torch.Tensor],
    particles: list[PCBNNParticle],
    status: dict,
) -> PCBNNPosteriorResult:
    stack = torch.stack(predictions, dim=0)
    precision = float(
        torch.stack([particle.log_beta.exp() for particle in particles]).mean().detach().cpu()
    )
    return PCBNNPosteriorResult(
        mean=stack.mean(dim=0),
        std=stack.std(dim=0, unbiased=False),
        samples=stack.squeeze(1),
        noise_precision=precision,
        status=status,
    )


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
    pairwise = sqdist.detach()[torch.triu_indices(particles, particles, offset=1, device=theta.device).unbind()]
    median = torch.median(pairwise)
    bandwidth = median / torch.log(torch.tensor(float(particles + 1), device=theta.device, dtype=theta.dtype))
    bandwidth = bandwidth.clamp_min(1e-6)
    kernel = torch.exp(-sqdist / bandwidth)
    attractive = kernel.T @ grad_loss / particles
    repulsive = torch.zeros_like(theta)
    for i in range(particles):
        diff = theta - theta[i]
        repulsive[i] = (2.0 / bandwidth) * (kernel[:, i].unsqueeze(1) * diff).mean(dim=0)
    # ``repulsive`` points from particle i toward the other particles.  It is
    # added to the descent direction because the caller subtracts the returned
    # update; this moves particle i away from its neighbours.
    return attractive + repulsive
