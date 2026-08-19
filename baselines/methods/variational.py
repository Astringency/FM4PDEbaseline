from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize

from baselines.common.data_adapter import PDEBatch


def observation_residuals(prediction: torch.Tensor, batch: PDEBatch) -> torch.Tensor:
    """Return ``H(x) - y`` for a one-sample sparse reconstruction batch."""

    if prediction.shape[0] != 1:
        raise ValueError("Variational observation extraction expects a one-sample prediction")
    if batch.mask is None:
        raise ValueError("Variational data assimilation requires an explicit observation mask")

    mask = batch.mask
    if tuple(mask.shape) == tuple(prediction.shape[1:]):
        sample_mask = mask
    elif tuple(mask.shape) == tuple(prediction.shape):
        sample_mask = mask[0]
    else:
        raise ValueError(
            f"Observation mask shape {tuple(mask.shape)} must match {tuple(prediction.shape)} "
            "with or without its batch dimension"
        )
    if not torch.equal(sample_mask, sample_mask[:1].expand_as(sample_mask)):
        raise ValueError("All state channels must use the same sensor locations")

    flat_indices = sample_mask[0].bool().reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    predicted = prediction.reshape(1, prediction.shape[1], -1)[0, :, flat_indices].transpose(0, 1)
    if batch.obs_values is not None:
        observed = batch.obs_values[0].to(device=prediction.device, dtype=prediction.dtype)
    else:
        observed = batch.input_fields.reshape(1, batch.input_fields.shape[1], -1)[0, :, flat_indices].transpose(0, 1)
        observed = observed.to(device=prediction.device, dtype=prediction.dtype)
    channels = min(int(predicted.shape[-1]), int(observed.shape[-1]))
    return predicted[..., :channels] - observed[..., :channels]


def balgovind_covariance_1d(
    size: int,
    *,
    variance: float,
    length_scale: float,
    device: torch.device,
    dtype: torch.dtype,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """Balgovind/Matérn-3/2 covariance used by the vendored VIVID code."""

    coordinate = torch.arange(size, device=device, dtype=dtype)
    distance = (coordinate[:, None] - coordinate[None, :]).abs()
    scale = max(float(length_scale), 1e-12)
    correlation = (1.0 + distance / scale) * torch.exp(-distance / scale)
    diagonal_scale = max(float(variance), 1e-12)
    return diagonal_scale * correlation + float(jitter) * torch.eye(size, device=device, dtype=dtype)


def periodic_balgovind_quadratic_2d(
    difference: torch.Tensor,
    *,
    variance: float,
    length_scale: float,
    eigenvalue_floor: float = 1e-6,
) -> torch.Tensor:
    """Matrix-free Balgovind background norm for a time-space state.

    The VIVID reference materializes a dense radial Balgovind matrix for a
    50x50 field.  A 128x128 Burgers state would make that matrix exceed 2 GiB,
    so the task adapter uses the same radial kernel with a circulant embedding.
    This keeps the covariance term and its inverse action while avoiding a
    dense 16,384-square matrix.
    """

    if difference.ndim != 4 or difference.shape[0] != 1 or difference.shape[1] != 1:
        raise ValueError(f"Expected a one-channel [1,1,T,X] difference, got {tuple(difference.shape)}")
    height, width = difference.shape[-2:]
    dtype = difference.dtype
    device = difference.device
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    dy = torch.minimum(y, height - y).reshape(height, 1)
    dx = torch.minimum(x, width - x).reshape(1, width)
    distance = torch.sqrt(dy.square() + dx.square())
    scale = max(float(length_scale), 1e-12)
    kernel = (1.0 + distance / scale) * torch.exp(-distance / scale)
    spectrum = torch.fft.fft2(kernel).real
    maximum = spectrum.abs().amax().clamp_min(torch.finfo(dtype).eps)
    spectrum = spectrum.clamp_min(maximum * float(eigenvalue_floor)) * max(float(variance), 1e-12)
    transformed = torch.fft.fft2(difference[0, 0], norm="ortho")
    return 0.5 * (transformed.abs().square() / spectrum).sum()


def scipy_lbfgsb(
    initial: torch.Tensor,
    objective: Callable[[torch.Tensor], torch.Tensor],
    *,
    max_steps: int,
    cost_decrement_tolerance: float = 1e-6,
    gradient_tolerance: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Minimize a torch objective with SciPy's L-BFGS-B implementation."""

    budget = max(int(max_steps), 0)
    if budget == 0:
        return initial.detach().clone(), {
            "requested_steps": 0,
            "completed_steps": 0,
            "function_evaluations": 0,
            "gradient_evaluations": 0,
            "nonfinite_trial_evaluations": 0,
            "early_stopped": False,
            "converged": False,
            "best_loss": None,
            "final_loss": None,
            "termination_reason": "zero optimization budget",
            "optimizer": "scipy_L-BFGS-B",
            "cost_decrement_tolerance": float(cost_decrement_tolerance),
        }

    shape = tuple(initial.shape)
    source_device = initial.device
    source_dtype = initial.dtype
    x0 = initial.detach().cpu().to(torch.float64).reshape(-1).numpy().copy()
    best_x = x0.copy()
    best_loss = float("inf")
    function_evaluations = 0
    nonfinite_evaluations = 0

    def value_and_gradient(flat: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal best_x, best_loss, function_evaluations, nonfinite_evaluations
        function_evaluations += 1
        state = torch.as_tensor(flat, device=source_device, dtype=source_dtype).reshape(shape).detach()
        state.requires_grad_(True)
        loss = objective(state)
        if loss.ndim != 0:
            raise ValueError(f"L-BFGS-B objective must be scalar, got shape {tuple(loss.shape)}")
        if not bool(torch.isfinite(loss).item()):
            # An L-BFGS-B line search may probe a dynamically unstable trial
            # state. Return a finite rejection penalty pointing back toward the
            # best finite trial so the line search can shorten its step.
            nonfinite_evaluations += 1
            reference = best_x if np.isfinite(best_loss) else x0
            displacement = np.asarray(flat, dtype=np.float64) - reference
            penalty = 1e20 + min(float(np.dot(displacement, displacement)), 1e10)
            return penalty, np.clip(2.0 * displacement, -1e10, 1e10)
        gradient = torch.autograd.grad(loss, state, create_graph=False, retain_graph=False)[0]
        if not bool(torch.isfinite(gradient).all().item()):
            nonfinite_evaluations += 1
            reference = best_x if np.isfinite(best_loss) else x0
            displacement = np.asarray(flat, dtype=np.float64) - reference
            penalty = 1e20 + min(float(np.dot(displacement, displacement)), 1e10)
            return penalty, np.clip(2.0 * displacement, -1e10, 1e10)
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_x = np.asarray(flat, dtype=np.float64).copy()
        return value, gradient.detach().cpu().to(torch.float64).reshape(-1).numpy()

    result = minimize(
        value_and_gradient,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": budget,
            "ftol": float(cost_decrement_tolerance),
            "gtol": float(gradient_tolerance),
            "maxls": 20,
        },
    )
    final_flat = best_x if np.isfinite(best_loss) else np.asarray(result.x, dtype=np.float64)
    optimized = torch.as_tensor(final_flat, device=source_device, dtype=source_dtype).reshape(shape)
    completed = int(getattr(result, "nit", 0))
    converged = bool(getattr(result, "success", False))
    status = {
        "requested_steps": budget,
        "completed_steps": completed,
        "function_evaluations": int(getattr(result, "nfev", function_evaluations)),
        "gradient_evaluations": int(getattr(result, "njev", function_evaluations)),
        "nonfinite_trial_evaluations": int(nonfinite_evaluations),
        "early_stopped": bool(converged and completed < budget),
        "converged": converged,
        "best_loss": None if not np.isfinite(best_loss) else float(best_loss),
        "final_loss": float(getattr(result, "fun", best_loss)),
        "termination_reason": str(getattr(result, "message", "")),
        "optimizer_status": int(getattr(result, "status", -1)),
        "optimizer": "scipy_L-BFGS-B",
        "cost_decrement_tolerance": float(cost_decrement_tolerance),
    }
    return optimized.detach(), status
