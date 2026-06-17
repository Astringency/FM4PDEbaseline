from __future__ import annotations

import math
import warnings
from typing import Any

import torch
import torch.nn.functional as F


def physics_losses(pred: torch.Tensor, pde_name: str, metadata: dict | None = None, reduction: str = "mean") -> dict[str, Any]:
    """Return structured physics losses for a prediction.

    ``residual_loss`` keeps the historical meaning of "interior PDE residual".
    This function adds explicit boundary and initial-condition terms and a
    weighted total.
    """

    metadata = metadata or {}
    pde = pde_name.lower()
    task = str(metadata.get("task", "")).lower()
    if pde == "poisson":
        losses = _poisson_losses(pred, metadata, inverse=task in {"inverse", "sparse_inverse"}, reduction=reduction)
    elif pde == "helmholtz":
        losses = _helmholtz_losses(pred, metadata, inverse=task in {"inverse", "sparse_inverse"}, reduction=reduction)
    elif pde == "darcy":
        losses = _darcy_losses(pred, metadata, inverse=task in {"inverse", "sparse_inverse"}, reduction=reduction)
    elif pde == "burger":
        losses = _burgers_losses(pred, metadata, reduction=reduction)
    elif pde in {"nsnonbounded", "navier_stokes", "ns"}:
        losses = _navier_stokes_losses(pred, metadata, reduction=reduction)
    elif pde in {"reaction_diffusion", "rd"}:
        losses = _reaction_diffusion_losses(pred, metadata, reduction=reduction)
    elif pde in {"shallow_water", "swe"}:
        losses = _shallow_water_losses(pred, metadata, reduction=reduction)
    elif pde == "heat":
        losses = _heat_losses(pred, metadata, reduction=reduction)
    elif pde in {"wave", "wave_equation"}:
        losses = _wave_losses(pred, metadata, reduction=reduction)
    elif pde in {"advection_diffusion", "advection-diffusion", "advdiff"}:
        losses = _advection_diffusion_losses(pred, metadata, reduction=reduction)
    elif pde in {"steady_heat_conduction", "nonlinear_heat_conduction", "steady_heat"}:
        losses = _steady_heat_conduction_losses(
            pred,
            metadata,
            inverse=task in {"inverse", "sparse_inverse"},
            reduction=reduction,
        )
    else:
        raise NotImplementedError(f"No PDE residual registered for {pde_name}")

    lam_int = _metadata_float(metadata, ("lambda_int", "lambda_pde"), 1.0)
    lam_bc = _metadata_float(metadata, ("lambda_bc",), 1.0)
    lam_ic = _metadata_float(metadata, ("lambda_ic",), 1.0)
    total = lam_int * losses["interior"] + lam_bc * losses["bc"] + lam_ic * losses["ic"]
    losses.update({"total": total, "lambda_int": lam_int, "lambda_bc": lam_bc, "lambda_ic": lam_ic})
    metadata["residual_mode"] = losses["mode"]
    return losses


def physics_loss(pred: torch.Tensor, pde_name: str, metadata: dict | None = None, reduction: str = "mean") -> torch.Tensor:
    return physics_losses(pred, pde_name, metadata, reduction)["total"]


def residual_loss(pred: torch.Tensor, pde_name: str, metadata: dict | None = None) -> torch.Tensor:
    return physics_losses(pred, pde_name, metadata)["interior"]


def residual_tensor(pred: torch.Tensor, pde_name: str, metadata: dict | None = None) -> torch.Tensor:
    residual = physics_losses(pred, pde_name, metadata)["residual"]
    if residual is None:
        raise NotImplementedError(f"No interior residual tensor available for {pde_name}")
    return residual


def poisson_inverse_residual(source: torch.Tensor, solution: torch.Tensor) -> torch.Tensor:
    h = _unit_spacing(solution)
    return interior_slice_2d(-laplacian(solution, spacing=h, boundary="dirichlet") - source)


def helmholtz_inverse_residual(source: torch.Tensor, solution: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    h = _unit_spacing(solution)
    return interior_slice_2d(-laplacian(solution, spacing=h, boundary="dirichlet") - (k**2) * solution - source)


def darcy_residual(coeff: torch.Tensor, solution: torch.Tensor) -> torch.Tensor:
    h = _unit_spacing(solution)
    grad_x = central_diff(solution, dim=-1, spacing=h, boundary="dirichlet")
    grad_y = central_diff(solution, dim=-2, spacing=h, boundary="dirichlet")
    flux_x = coeff * grad_x
    flux_y = coeff * grad_y
    div = central_diff(flux_x, dim=-1, spacing=h, boundary="dirichlet") + central_diff(flux_y, dim=-2, spacing=h, boundary="dirichlet")
    return interior_slice_2d(-div - 1.0)


def burgers_residual(u: torch.Tensor, nu: float = 0.01, metadata: dict | None = None) -> torch.Tensor:
    # u: [B,1,T,X]
    metadata = metadata or {}
    dt = _time_step(u.shape[-2], float(metadata.get("final_time", 1.0)), metadata)
    dx = 1.0 / max(u.shape[-1] - 1, 1)
    u_t = central_diff(u, dim=-2, spacing=dt, boundary="replicate")
    u_x = central_diff(u, dim=-1, spacing=dx, boundary="periodic")
    u_xx = laplacian_1d(u, dim=-1, spacing=dx, boundary="periodic")
    return u_t + u * u_x - nu * u_xx


def navier_stokes_vorticity_residual(w: torch.Tensor, metadata: dict | None = None) -> torch.Tensor:
    # w: [B,1,T,H,W], periodic vorticity equation.
    metadata = metadata or {}
    nu = float(metadata.get("nu", metadata.get("viscosity", 1e-3)))
    final_time = float(metadata.get("final_time", 1.0))
    dt = _time_step(w.shape[2], final_time, metadata)
    h = 1.0 / max(w.shape[-1], 1)
    omega = w[:, 0]
    psi = _streamfunction_from_vorticity(omega)
    vel_x = central_diff(psi, dim=-2, spacing=h, boundary="periodic")
    vel_y = -central_diff(psi, dim=-1, spacing=h, boundary="periodic")
    omega_t = central_diff(omega, dim=1, spacing=dt, boundary="replicate")
    omega_x = central_diff(omega, dim=-1, spacing=h, boundary="periodic")
    omega_y = central_diff(omega, dim=-2, spacing=h, boundary="periodic")
    lap = laplacian(omega.unsqueeze(1).reshape(-1, 1, *omega.shape[-2:]), spacing=h, boundary="periodic")
    lap = lap.reshape_as(omega)
    forcing = _ns_forcing(w.shape[-2], w.shape[-1], w.device, w.dtype).view(1, 1, w.shape[-2], w.shape[-1])
    return (omega_t + vel_x * omega_x + vel_y * omega_y - nu * lap - forcing).unsqueeze(1)


def reaction_diffusion_residual(uv: torch.Tensor, metadata: dict | None = None) -> torch.Tensor:
    # uv: [B,2,T,H,W], Neumann on [-1,1]^2.
    metadata = metadata or {}
    du = float(metadata.get("D_u", metadata.get("du", 1e-3)))
    dv = float(metadata.get("D_v", metadata.get("dv", 5e-3)))
    k = float(metadata.get("k", 5e-3))
    final_time = float(metadata.get("final_time", 5.0))
    dt = _time_step(uv.shape[2], final_time, metadata)
    dx = 2.0 / max(uv.shape[-1] - 1, 1)
    u = uv[:, 0]
    v = uv[:, 1]
    u_t = central_diff(u, dim=1, spacing=dt, boundary="replicate")
    v_t = central_diff(v, dim=1, spacing=dt, boundary="replicate")
    lap_u = laplacian(u.reshape(-1, 1, *u.shape[-2:]), spacing=dx, boundary="neumann").reshape_as(u)
    lap_v = laplacian(v.reshape(-1, 1, *v.shape[-2:]), spacing=dx, boundary="neumann").reshape_as(v)
    res_u = u_t - (du * lap_u + (u - u**3 - k - v))
    res_v = v_t - (dv * lap_v + (u - v))
    return torch.stack([res_u, res_v], dim=1)


def shallow_water_residual(q: torch.Tensor, metadata: dict | None = None) -> torch.Tensor:
    # q: [B,3,T,H,W] with conservative variables [h,hu,hv].
    metadata = metadata or {}
    g = float(metadata.get("g", 1.0))
    final_time = float(metadata.get("final_time", 1.0))
    domain = float(metadata.get("domain_length", 5.0))
    dt = _time_step(q.shape[2], final_time, metadata)
    dx = domain / max(q.shape[-1] - 1, 1)
    h = q[:, 0].clamp_min(float(metadata.get("min_depth", 1e-4)))
    hu = q[:, 1]
    hv = q[:, 2]
    u = hu / h
    v = hv / h
    flux_x_h = hu
    flux_y_h = hv
    flux_x_hu = hu * u + 0.5 * g * h**2
    flux_y_hu = hu * v
    flux_x_hv = hv * u
    flux_y_hv = hv * v + 0.5 * g * h**2
    h_t = central_diff(h, dim=1, spacing=dt, boundary="replicate")
    hu_t = central_diff(hu, dim=1, spacing=dt, boundary="replicate")
    hv_t = central_diff(hv, dim=1, spacing=dt, boundary="replicate")
    res_h = h_t + central_diff(flux_x_h, dim=-1, spacing=dx, boundary="neumann") + central_diff(flux_y_h, dim=-2, spacing=dx, boundary="neumann")
    res_hu = hu_t + central_diff(flux_x_hu, dim=-1, spacing=dx, boundary="neumann") + central_diff(flux_y_hu, dim=-2, spacing=dx, boundary="neumann")
    res_hv = hv_t + central_diff(flux_x_hv, dim=-1, spacing=dx, boundary="neumann") + central_diff(flux_y_hv, dim=-2, spacing=dx, boundary="neumann")
    return torch.stack([res_h, res_hu, res_hv], dim=1)


def heat_residual(
    u: torch.Tensor,
    alpha: torch.Tensor | float | None = None,
    metadata: dict | None = None,
    pred: torch.Tensor | None = None,
) -> torch.Tensor:
    # u: [B,1,T,H,W], default periodic heat equation u_t = alpha Delta u.
    metadata = metadata or {}
    alpha_field = _parameter_field(
        metadata,
        u[:, :1, 0],
        ("alpha", "fixed_alpha", "diffusivity"),
        default=1e-3,
        input_channel=1,
        input_min_channels=2,
        full_channel=1,
        full_min_channels=3,
        pred_channel=1,
        pred_min_channels=2,
        pred=pred,
        value=alpha,
    )
    final_time = float(metadata.get("final_time", metadata.get("T", 1.0)))
    dt = _time_step(u.shape[2], final_time, metadata)
    boundary = _boundary_mode(metadata, default="periodic")
    dx = _spatial_step(u, metadata, boundary=boundary, default_domain=1.0)
    u_t = central_diff(u[:, :1], dim=2, spacing=dt, boundary="replicate")
    lap = _trajectory_laplacian(u[:, :1], spacing=dx, boundary=boundary)
    return u_t - alpha_field.unsqueeze(2) * lap


def wave_residual(
    q: torch.Tensor,
    wave_speed: torch.Tensor | float | None = None,
    metadata: dict | None = None,
    pred: torch.Tensor | None = None,
) -> torch.Tensor:
    # q: [B,2,T,H,W] with channels [u, v=u_t].
    metadata = metadata or {}
    c_field = _parameter_field(
        metadata,
        q[:, :1, 0],
        ("c", "fixed_c", "wave_speed"),
        default=1.0,
        input_channel=2,
        input_min_channels=3,
        full_channel=2,
        full_min_channels=5,
        pred_channel=2,
        pred_min_channels=3,
        pred=pred,
        value=wave_speed,
    )
    final_time = float(metadata.get("final_time", metadata.get("T", 1.0)))
    dt = _time_step(q.shape[2], final_time, metadata)
    boundary = _boundary_mode(metadata, default="periodic")
    dx = _spatial_step(q, metadata, boundary=boundary, default_domain=1.0)
    u = q[:, :1]
    v = q[:, 1:2]
    u_t = central_diff(u, dim=2, spacing=dt, boundary="replicate")
    v_t = central_diff(v, dim=2, spacing=dt, boundary="replicate")
    lap_u = _trajectory_laplacian(u, spacing=dx, boundary=boundary)
    return torch.cat([u_t - v, v_t - c_field.square().unsqueeze(2) * lap_u], dim=1)


def advection_diffusion_residual(u: torch.Tensor, metadata: dict | None = None, pred: torch.Tensor | None = None) -> torch.Tensor:
    # u: [B,1,T,H,W], periodic advection-diffusion equation.
    metadata = metadata or {}
    ref = u[:, :1, 0]
    bx = _parameter_field(
        metadata,
        ref,
        ("b_x", "bx", "velocity_x"),
        default=0.0,
        input_channel=1,
        input_min_channels=4,
        full_channel=1,
        full_min_channels=5,
        pred_channel=1,
        pred_min_channels=4,
        pred=pred,
    )
    by = _parameter_field(
        metadata,
        ref,
        ("b_y", "by", "velocity_y"),
        default=0.0,
        input_channel=2,
        input_min_channels=4,
        full_channel=2,
        full_min_channels=5,
        pred_channel=2,
        pred_min_channels=4,
        pred=pred,
    )
    kappa = _parameter_field(
        metadata,
        ref,
        ("kappa", "diffusivity"),
        default=1e-3,
        input_channel=3,
        input_min_channels=4,
        full_channel=3,
        full_min_channels=5,
        pred_channel=3,
        pred_min_channels=4,
        pred=pred,
    )
    final_time = float(metadata.get("final_time", metadata.get("T", 1.0)))
    dt = _time_step(u.shape[2], final_time, metadata)
    boundary = _boundary_mode(metadata, default="periodic")
    dx = _spatial_step(u, metadata, boundary=boundary, default_domain=1.0)
    field = u[:, :1]
    u_t = central_diff(field, dim=2, spacing=dt, boundary="replicate")
    u_x = central_diff(field, dim=-1, spacing=dx, boundary=boundary)
    u_y = central_diff(field, dim=-2, spacing=dx, boundary=boundary)
    lap_u = _trajectory_laplacian(field, spacing=dx, boundary=boundary)
    return u_t + bx.unsqueeze(2) * u_x + by.unsqueeze(2) * u_y - kappa.unsqueeze(2) * lap_u


def steady_heat_conduction_residual(source: torch.Tensor, solution: torch.Tensor, metadata: dict | None = None) -> torch.Tensor:
    # -div(lambda(u) grad u) = f, lambda(u)=1+0.05*(u-298).
    metadata = metadata or {}
    h = _spatial_step(solution, metadata, boundary="mixed", default_domain=1.0)
    conductivity = (1.0 + 0.05 * (solution[:, :1] - 298.0)).clamp_min(float(metadata.get("lambda_min", 0.1)))
    grad_x = central_diff(solution[:, :1], dim=-1, spacing=h, boundary="neumann")
    grad_y = central_diff(solution[:, :1], dim=-2, spacing=h, boundary="neumann")
    flux_x = conductivity * grad_x
    flux_y = conductivity * grad_y
    div = central_diff(flux_x, dim=-1, spacing=h, boundary="neumann") + central_diff(flux_y, dim=-2, spacing=h, boundary="neumann")
    return interior_slice_2d(-div - source[:, :1])


def interior_slice_2d(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2] <= 2 or x.shape[-1] <= 2:
        return x
    return x[..., 1:-1, 1:-1]


def boundary_values_2d(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return field[..., 0, :], field[..., -1, :], field[..., :, 0], field[..., :, -1]


def dirichlet_zero_bc_loss(field: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    top, bottom, left, right = boundary_values_2d(field)
    return _mean_square(torch.cat([v.reshape(-1) for v in (top, bottom, left, right)]), reduction)


def neumann_zero_bc_loss(field: torch.Tensor, spacing_x: float, spacing_y: float | None = None, reduction: str = "mean") -> torch.Tensor:
    spacing_y = spacing_x if spacing_y is None else spacing_y
    if field.shape[-2] < 2 or field.shape[-1] < 2:
        return _zero_scalar(field)
    d_top = (field[..., 1, :] - field[..., 0, :]) / spacing_y
    d_bottom = (field[..., -1, :] - field[..., -2, :]) / spacing_y
    d_left = (field[..., :, 1] - field[..., :, 0]) / spacing_x
    d_right = (field[..., :, -1] - field[..., :, -2]) / spacing_x
    return _mean_square(torch.cat([v.reshape(-1) for v in (d_top, d_bottom, d_left, d_right)]), reduction)


def periodic_bc_loss(field: torch.Tensor, dims: tuple[int, ...] | None = None, reduction: str = "mean") -> torch.Tensor:
    dims = dims or (-2, -1)
    terms = []
    for dim in dims:
        dim = dim if dim >= 0 else field.ndim + dim
        if field.shape[dim] < 2:
            continue
        first = _take_dim(field, dim, 0)
        last = _take_dim(field, dim, -1)
        terms.append((first - last).reshape(-1))
        if field.shape[dim] > 2:
            d_first = _take_dim(field, dim, 1) - _take_dim(field, dim, 0)
            d_last = _take_dim(field, dim, -1) - _take_dim(field, dim, -2)
            terms.append((d_first - d_last).reshape(-1))
    if not terms:
        return _zero_scalar(field)
    return _mean_square(torch.cat(terms), reduction)


def central_diff(x: torch.Tensor, dim: int, spacing: float, boundary: str) -> torch.Tensor:
    dim = dim if dim >= 0 else x.ndim + dim
    if x.shape[dim] < 2:
        return torch.zeros_like(x)
    if boundary == "periodic":
        return (torch.roll(x, shifts=-1, dims=dim) - torch.roll(x, shifts=1, dims=dim)) / (2.0 * spacing)
    out = torch.zeros_like(x)
    idx_mid = [slice(None)] * x.ndim
    idx_l = [slice(None)] * x.ndim
    idx_r = [slice(None)] * x.ndim
    idx_mid[dim] = slice(1, -1)
    idx_l[dim] = slice(0, -2)
    idx_r[dim] = slice(2, None)
    out[tuple(idx_mid)] = (x[tuple(idx_r)] - x[tuple(idx_l)]) / (2.0 * spacing)
    if boundary in {"neumann", "replicate"}:
        idx0 = [slice(None)] * x.ndim
        idx1 = [slice(None)] * x.ndim
        idx0[dim] = 0
        idx1[dim] = 1
        out[tuple(idx0)] = (x[tuple(idx1)] - x[tuple(idx0)]) / spacing
        idxe = [slice(None)] * x.ndim
        idxm = [slice(None)] * x.ndim
        idxe[dim] = -1
        idxm[dim] = -2
        out[tuple(idxe)] = (x[tuple(idxe)] - x[tuple(idxm)]) / spacing
    elif boundary == "dirichlet":
        # Boundary values are constrained by a separate BC term. The derivative
        # there is not used for interior residual means, so leave zeros.
        pass
    else:
        raise ValueError(f"Unknown boundary mode '{boundary}'")
    return out


def laplacian(u: torch.Tensor, spacing: float, boundary: str = "neumann") -> torch.Tensor:
    if u.ndim != 4:
        raise ValueError(f"laplacian expects [B,C,H,W], got {tuple(u.shape)}")
    if boundary == "periodic":
        return (
            torch.roll(u, 1, -1)
            + torch.roll(u, -1, -1)
            + torch.roll(u, 1, -2)
            + torch.roll(u, -1, -2)
            - 4.0 * u
        ) / (spacing**2)
    if boundary == "dirichlet":
        padded = F.pad(u, (1, 1, 1, 1), mode="constant", value=0.0)
    elif boundary in {"neumann", "replicate"}:
        padded = F.pad(u, (1, 1, 1, 1), mode="replicate")
    else:
        raise ValueError(f"Unknown boundary mode '{boundary}'")
    return (
        padded[:, :, :-2, 1:-1]
        + padded[:, :, 2:, 1:-1]
        + padded[:, :, 1:-1, :-2]
        + padded[:, :, 1:-1, 2:]
        - 4.0 * u
    ) / (spacing**2)


def laplacian_1d(u: torch.Tensor, dim: int, spacing: float, boundary: str = "periodic") -> torch.Tensor:
    dim = dim if dim >= 0 else u.ndim + dim
    if u.shape[dim] < 2:
        return torch.zeros_like(u)
    if boundary == "periodic":
        return (torch.roll(u, 1, dim) + torch.roll(u, -1, dim) - 2.0 * u) / (spacing**2)
    out = torch.zeros_like(u)
    idx = [slice(None)] * u.ndim
    idx_l = [slice(None)] * u.ndim
    idx_r = [slice(None)] * u.ndim
    idx[dim] = slice(1, -1)
    idx_l[dim] = slice(0, -2)
    idx_r[dim] = slice(2, None)
    out[tuple(idx)] = (u[tuple(idx_l)] - 2.0 * u[tuple(idx)] + u[tuple(idx_r)]) / (spacing**2)
    if boundary in {"neumann", "replicate"}:
        idx0 = [slice(None)] * u.ndim
        idx1 = [slice(None)] * u.ndim
        idx0[dim] = 0
        idx1[dim] = 1
        out[tuple(idx0)] = (u[tuple(idx0)] - 2.0 * u[tuple(idx0)] + u[tuple(idx1)]) / (spacing**2)
        idxe = [slice(None)] * u.ndim
        idxm = [slice(None)] * u.ndim
        idxe[dim] = -1
        idxm[dim] = -2
        out[tuple(idxe)] = (u[tuple(idxm)] - 2.0 * u[tuple(idxe)] + u[tuple(idxe)]) / (spacing**2)
    return out


def _poisson_losses(pred: torch.Tensor, metadata: dict, inverse: bool, reduction: str) -> dict[str, Any]:
    if inverse:
        solution = _solution_from_metadata(metadata, pred)
        residual = poisson_inverse_residual(pred[:, :1], solution[:, :1])
    else:
        solution = pred[:, :1]
        source = _coefficient_from_metadata(metadata, pred)
        residual = poisson_inverse_residual(source[:, :1], solution)
    return _loss_dict(pred, residual, dirichlet_zero_bc_loss(solution, reduction), _zero_scalar(pred), "steady_dirichlet", reduction)


def _helmholtz_losses(pred: torch.Tensor, metadata: dict, inverse: bool, reduction: str) -> dict[str, Any]:
    k = float(metadata.get("k", 1.0))
    if inverse:
        solution = _solution_from_metadata(metadata, pred)
        residual = helmholtz_inverse_residual(pred[:, :1], solution[:, :1], k=k)
    else:
        solution = pred[:, :1]
        source = _coefficient_from_metadata(metadata, pred)
        residual = helmholtz_inverse_residual(source[:, :1], solution, k=k)
    return _loss_dict(pred, residual, dirichlet_zero_bc_loss(solution, reduction), _zero_scalar(pred), "steady_dirichlet", reduction)


def _darcy_losses(pred: torch.Tensor, metadata: dict, inverse: bool, reduction: str) -> dict[str, Any]:
    if inverse:
        coeff = pred[:, :1]
        solution = _solution_from_metadata(metadata, pred)
    else:
        coeff = _coefficient_from_metadata(metadata, pred)
        solution = pred[:, :1]
    residual = darcy_residual(coeff[:, :1], solution[:, :1])
    return _loss_dict(pred, residual, dirichlet_zero_bc_loss(solution[:, :1], reduction), _zero_scalar(pred), "steady_dirichlet", reduction)


def _burgers_losses(pred: torch.Tensor, metadata: dict, reduction: str) -> dict[str, Any]:
    trajectory, mode = _as_burgers_trajectory(pred, metadata)
    nu = float(metadata.get("nu", metadata.get("viscosity", 0.01)))
    residual = burgers_residual(trajectory, nu=nu, metadata=metadata)
    bc = periodic_bc_loss(trajectory, dims=(-1,), reduction=reduction)
    initial = _burgers_initial_from_metadata(metadata, trajectory)
    ic = _mean_square(trajectory[:, :, 0, :] - initial, reduction)
    return _loss_dict(pred, residual, bc, ic, mode, reduction)


def _navier_stokes_losses(pred: torch.Tensor, metadata: dict, reduction: str) -> dict[str, Any]:
    trajectory, mode = _as_ns_trajectory(pred, metadata)
    residual = navier_stokes_vorticity_residual(trajectory, metadata)
    bc = periodic_bc_loss(trajectory, dims=(-2, -1), reduction=reduction)
    initial = _initial_from_metadata(metadata, trajectory, channels=1)
    ic = _mean_square(trajectory[:, :, 0] - initial[:, :1], reduction)
    return _loss_dict(pred, residual, bc, ic, mode, reduction)


def _reaction_diffusion_losses(pred: torch.Tensor, metadata: dict, reduction: str) -> dict[str, Any]:
    trajectory, mode = _as_rd_trajectory(pred, metadata)
    residual = reaction_diffusion_residual(trajectory, metadata)
    dx = 2.0 / max(trajectory.shape[-1] - 1, 1)
    bc = neumann_zero_bc_loss(trajectory, spacing_x=dx, spacing_y=dx, reduction=reduction)
    initial = _initial_from_metadata(metadata, trajectory, channels=2)
    ic = _mean_square(trajectory[:, :2, 0] - initial[:, :2], reduction)
    return _loss_dict(pred, residual, bc, ic, mode, reduction)


def _shallow_water_losses(pred: torch.Tensor, metadata: dict, reduction: str) -> dict[str, Any]:
    trajectory, mode = _as_swe_trajectory(pred, metadata)
    residual = shallow_water_residual(trajectory, metadata)
    domain = float(metadata.get("domain_length", 5.0))
    dx = domain / max(trajectory.shape[-1] - 1, 1)
    bc = neumann_zero_bc_loss(trajectory, spacing_x=dx, spacing_y=dx, reduction=reduction)
    initial = _initial_from_metadata(metadata, trajectory, channels=3)
    ic = _mean_square(trajectory[:, :3, 0] - initial[:, :3], reduction)
    return _loss_dict(pred, residual, bc, ic, mode, reduction)


def _heat_losses(pred: torch.Tensor, metadata: dict, reduction: str) -> dict[str, Any]:
    trajectory, mode = _as_heat_trajectory(pred, metadata)
    residual = heat_residual(trajectory, metadata=metadata, pred=pred)
    boundary = _boundary_mode(metadata, default="periodic")
    dx = _spatial_step(trajectory, metadata, boundary=boundary, default_domain=1.0)
    bc = _trajectory_bc_loss(trajectory, boundary, dx, reduction)
    initial = _initial_channels_from_metadata(metadata, pred, channels=1, full_channel=0, input_channel=0)
    ic = _mean_square(trajectory[:, :1, 0] - initial[:, :1], reduction)
    return _loss_dict(pred, residual, bc, ic, mode, reduction)


def _wave_losses(pred: torch.Tensor, metadata: dict, reduction: str) -> dict[str, Any]:
    trajectory, mode = _as_wave_trajectory(pred, metadata)
    residual = wave_residual(trajectory, metadata=metadata, pred=pred)
    boundary = _boundary_mode(metadata, default="periodic")
    dx = _spatial_step(trajectory, metadata, boundary=boundary, default_domain=1.0)
    bc = _trajectory_bc_loss(trajectory, boundary, dx, reduction)
    initial = _initial_channels_from_metadata(metadata, pred, channels=2, full_channel=0, input_channel=0)
    ic = _mean_square(trajectory[:, :2, 0] - initial[:, :2], reduction)
    return _loss_dict(pred, residual, bc, ic, mode, reduction)


def _advection_diffusion_losses(pred: torch.Tensor, metadata: dict, reduction: str) -> dict[str, Any]:
    trajectory, mode = _as_advection_diffusion_trajectory(pred, metadata)
    residual = advection_diffusion_residual(trajectory, metadata, pred=pred)
    boundary = _boundary_mode(metadata, default="periodic")
    dx = _spatial_step(trajectory, metadata, boundary=boundary, default_domain=1.0)
    bc = _trajectory_bc_loss(trajectory, boundary, dx, reduction)
    initial = _initial_channels_from_metadata(metadata, pred, channels=1, full_channel=0, input_channel=0)
    ic = _mean_square(trajectory[:, :1, 0] - initial[:, :1], reduction)
    return _loss_dict(pred, residual, bc, ic, mode, reduction)


def _steady_heat_conduction_losses(pred: torch.Tensor, metadata: dict, inverse: bool, reduction: str) -> dict[str, Any]:
    if inverse:
        source = pred[:, :1]
        solution = _solution_channels_from_metadata(metadata, pred, channels=1, full_channel=2, input_channel=0)
    else:
        source = _source_channels_from_metadata(metadata, pred, channels=1, full_channel=0, input_channel=0)
        solution = pred[:, :1]
    u_d = _parameter_field(
        metadata,
        solution[:, :1],
        ("u_D", "u_d", "dirichlet_temperature", "sink_temperature"),
        default=298.0,
        input_channel=1,
        full_channel=1 if not inverse else 3,
        pred_channel=1,
        pred=pred,
    )
    residual = steady_heat_conduction_residual(source, solution, metadata)
    dx = _spatial_step(solution, metadata, boundary="mixed", default_domain=1.0)
    bc = _steady_heat_mixed_bc_loss(solution[:, :1], u_d, dx, reduction)
    return _loss_dict(pred, residual, bc, _zero_scalar(pred), "steady_mixed_bc", reduction)


def _loss_dict(ref: torch.Tensor, residual: torch.Tensor, bc: torch.Tensor, ic: torch.Tensor, mode: str, reduction: str) -> dict[str, Any]:
    return {
        "interior": _mean_square(residual, reduction) if residual is not None else _zero_scalar(ref),
        "bc": bc,
        "ic": ic,
        "total": _zero_scalar(ref),
        "residual": residual,
        "mode": mode,
    }


def _coefficient_from_metadata(metadata: dict, pred: torch.Tensor) -> torch.Tensor:
    if isinstance(metadata.get("coeff_fields"), torch.Tensor):
        return metadata["coeff_fields"].to(pred.device, pred.dtype)
    if isinstance(metadata.get("source_fields"), torch.Tensor):
        return metadata["source_fields"].to(pred.device, pred.dtype)
    if isinstance(metadata.get("full_tensor"), torch.Tensor):
        full = metadata["full_tensor"].to(pred.device, pred.dtype)
        if full.ndim == 4:
            return full[:, :1]
    input_fields = metadata.get("input_fields")
    if isinstance(input_fields, torch.Tensor):
        if "sparse" in str(metadata.get("task", "")).lower():
            warnings.warn("Falling back to sparse input_fields as PDE coefficient/source; full_tensor metadata is preferred.", RuntimeWarning, stacklevel=2)
        return input_fields.to(pred.device, pred.dtype)
    raise ValueError("Missing coefficient/source field for PDE residual")


def _solution_from_metadata(metadata: dict, pred: torch.Tensor) -> torch.Tensor:
    if isinstance(metadata.get("solution_fields"), torch.Tensor):
        return metadata["solution_fields"].to(pred.device, pred.dtype)
    if isinstance(metadata.get("full_tensor"), torch.Tensor):
        full = metadata["full_tensor"].to(pred.device, pred.dtype)
        if full.ndim == 4 and full.shape[1] > 1:
            return full[:, 1:2]
    input_fields = metadata.get("input_fields")
    if isinstance(input_fields, torch.Tensor):
        return input_fields.to(pred.device, pred.dtype)
    raise ValueError("Missing solution field for inverse PDE residual")


def _as_burgers_trajectory(pred: torch.Tensor, metadata: dict) -> tuple[torch.Tensor, str]:
    if pred.ndim == 4:
        return pred[:, :1], "full_trajectory" if pred.shape[-2] > 2 else "two_level"
    if pred.ndim == 3:
        return pred[:, None], "full_trajectory" if pred.shape[-2] > 2 else "two_level"
    raise ValueError(f"Burgers field expects [B,1,T,X] or [B,T,X], got {tuple(pred.shape)}")


def _as_ns_trajectory(pred: torch.Tensor, metadata: dict) -> tuple[torch.Tensor, str]:
    if pred.ndim == 5:
        return pred[:, :1], "full_trajectory" if pred.shape[2] > 2 else "two_level"
    if pred.ndim != 4:
        raise ValueError(f"NS residual expects [B,T,H,W], [B,1,H,W], or [B,1,T,H,W], got {tuple(pred.shape)}")
    full = metadata.get("full_tensor")
    input_fields = metadata.get("input_fields")
    if pred.shape[1] > 1:
        traj = pred.unsqueeze(1)
        if isinstance(full, torch.Tensor) and full.ndim == 5:
            initial = full[:, :, :1].to(pred.device, pred.dtype)
        elif isinstance(input_fields, torch.Tensor) and input_fields.shape[1] == 1:
            initial = input_fields[:, :, None].to(pred.device, pred.dtype)
        else:
            warnings.warn("Missing NS initial state metadata; using first predicted frame as initial state.", RuntimeWarning, stacklevel=2)
            initial = traj[:, :, :1]
        return torch.cat([initial, traj], dim=2), "full_trajectory"
    initial = _initial_from_metadata(metadata, pred, channels=1)[:, :1]
    return torch.stack([initial, pred[:, :1]], dim=2), "two_level"


def _as_rd_trajectory(pred: torch.Tensor, metadata: dict) -> tuple[torch.Tensor, str]:
    if pred.ndim == 5:
        return pred[:, :2], "full_trajectory" if pred.shape[2] > 2 else "two_level"
    if pred.ndim != 4 or pred.shape[1] < 2:
        raise ValueError(f"Reaction-diffusion residual expects [B,2,H,W] or [B,2,T,H,W], got {tuple(pred.shape)}")
    initial = _initial_from_metadata(metadata, pred, channels=2)
    return torch.stack([initial[:, :2], pred[:, :2]], dim=2), "two_level"


def _as_swe_trajectory(pred: torch.Tensor, metadata: dict) -> tuple[torch.Tensor, str]:
    if pred.ndim == 5:
        return pred[:, :3], "full_trajectory" if pred.shape[2] > 2 else "two_level"
    if pred.ndim != 4 or pred.shape[1] < 3:
        raise ValueError(f"Shallow-water residual expects [B,3,H,W] or [B,3,T,H,W], got {tuple(pred.shape)}")
    initial = _initial_from_metadata(metadata, pred, channels=3)
    return torch.stack([initial[:, :3], pred[:, :3]], dim=2), "two_level"


def _as_heat_trajectory(pred: torch.Tensor, metadata: dict) -> tuple[torch.Tensor, str]:
    if pred.ndim == 5:
        return pred[:, :1], "full_trajectory" if pred.shape[2] > 2 else "two_level"
    if pred.ndim != 4 or pred.shape[1] < 1:
        raise ValueError(f"Heat residual expects [B,1,H,W] or [B,1,T,H,W], got {tuple(pred.shape)}")
    initial = _initial_channels_from_metadata(metadata, pred, channels=1, full_channel=0, input_channel=0)
    return torch.stack([initial[:, :1], pred[:, :1]], dim=2), "two_level"


def _as_wave_trajectory(pred: torch.Tensor, metadata: dict) -> tuple[torch.Tensor, str]:
    if pred.ndim == 5:
        if pred.shape[1] < 2:
            raise ValueError(f"Wave trajectory needs [u,v] channels, got {tuple(pred.shape)}")
        return pred[:, :2], "full_trajectory" if pred.shape[2] > 2 else "two_level"
    if pred.ndim != 4 or pred.shape[1] < 2:
        raise ValueError(f"Wave residual expects [B,2,H,W] or [B,2,T,H,W], got {tuple(pred.shape)}")
    initial = _initial_channels_from_metadata(metadata, pred, channels=2, full_channel=0, input_channel=0)
    return torch.stack([initial[:, :2], pred[:, :2]], dim=2), "two_level"


def _as_advection_diffusion_trajectory(pred: torch.Tensor, metadata: dict) -> tuple[torch.Tensor, str]:
    if pred.ndim == 5:
        return pred[:, :1], "full_trajectory" if pred.shape[2] > 2 else "two_level"
    if pred.ndim != 4 or pred.shape[1] < 1:
        raise ValueError(f"Advection-diffusion residual expects [B,1,H,W] or [B,1,T,H,W], got {tuple(pred.shape)}")
    initial = _initial_channels_from_metadata(metadata, pred, channels=1, full_channel=0, input_channel=0)
    return torch.stack([initial[:, :1], pred[:, :1]], dim=2), "two_level"


def _initial_channels_from_metadata(
    metadata: dict,
    pred: torch.Tensor,
    channels: int,
    full_channel: int = 0,
    input_channel: int = 0,
) -> torch.Tensor:
    if isinstance(metadata.get("background_fields"), torch.Tensor):
        return metadata["background_fields"].to(pred.device, pred.dtype)[:, :channels]
    if isinstance(metadata.get("full_tensor"), torch.Tensor):
        full = metadata["full_tensor"].to(pred.device, pred.dtype)
        if full.ndim == 5:
            idx = int(metadata.get("input_time_index", 0))
            return full[:, :channels, idx]
        if full.ndim == 4 and full.shape[1] >= full_channel + channels:
            return full[:, full_channel : full_channel + channels]
    if isinstance(metadata.get("input_fields"), torch.Tensor):
        input_fields = metadata["input_fields"].to(pred.device, pred.dtype)
        if input_fields.ndim == 4 and input_fields.shape[1] >= input_channel + channels:
            return input_fields[:, input_channel : input_channel + channels]
    warnings.warn("Missing initial/background state for time-dependent residual; using predicted first frame.", RuntimeWarning, stacklevel=2)
    if pred.ndim == 5:
        return pred[:, :channels, 0]
    return pred[:, :channels]


def _source_channels_from_metadata(
    metadata: dict,
    pred: torch.Tensor,
    channels: int,
    full_channel: int = 0,
    input_channel: int = 0,
) -> torch.Tensor:
    for key in ("source_fields", "coeff_fields"):
        if isinstance(metadata.get(key), torch.Tensor):
            return metadata[key].to(pred.device, pred.dtype)[:, :channels]
    if isinstance(metadata.get("full_tensor"), torch.Tensor):
        full = metadata["full_tensor"].to(pred.device, pred.dtype)
        if full.ndim == 4 and full.shape[1] >= full_channel + channels:
            return full[:, full_channel : full_channel + channels]
    if isinstance(metadata.get("input_fields"), torch.Tensor):
        input_fields = metadata["input_fields"].to(pred.device, pred.dtype)
        if input_fields.ndim == 4 and input_fields.shape[1] >= input_channel + channels:
            if "sparse" in str(metadata.get("task", "")).lower():
                warnings.warn("Falling back to sparse input_fields as PDE source; full_tensor metadata is preferred.", RuntimeWarning, stacklevel=2)
            return input_fields[:, input_channel : input_channel + channels]
    raise ValueError("Missing source field for PDE residual")


def _solution_channels_from_metadata(
    metadata: dict,
    pred: torch.Tensor,
    channels: int,
    full_channel: int,
    input_channel: int = 0,
) -> torch.Tensor:
    if isinstance(metadata.get("solution_fields"), torch.Tensor):
        return metadata["solution_fields"].to(pred.device, pred.dtype)[:, :channels]
    if isinstance(metadata.get("full_tensor"), torch.Tensor):
        full = metadata["full_tensor"].to(pred.device, pred.dtype)
        if full.ndim == 4 and full.shape[1] >= full_channel + channels:
            return full[:, full_channel : full_channel + channels]
    if isinstance(metadata.get("input_fields"), torch.Tensor):
        input_fields = metadata["input_fields"].to(pred.device, pred.dtype)
        if input_fields.ndim == 4 and input_fields.shape[1] >= input_channel + channels:
            return input_fields[:, input_channel : input_channel + channels]
    raise ValueError("Missing solution field for inverse PDE residual")


def _initial_from_metadata(metadata: dict, pred: torch.Tensor, channels: int) -> torch.Tensor:
    if isinstance(metadata.get("background_fields"), torch.Tensor):
        return metadata["background_fields"].to(pred.device, pred.dtype)
    if isinstance(metadata.get("input_fields"), torch.Tensor) and metadata["input_fields"].shape[1] == channels:
        input_fields = metadata["input_fields"].to(pred.device, pred.dtype)
        return input_fields[:, :channels]
    if isinstance(metadata.get("full_tensor"), torch.Tensor):
        full = metadata["full_tensor"].to(pred.device, pred.dtype)
        if full.ndim == 5:
            idx = int(metadata.get("input_time_index", 0))
            return full[:, :channels, idx]
    warnings.warn("Missing initial/background state for time-dependent residual; using predicted first frame.", RuntimeWarning, stacklevel=2)
    if pred.ndim == 5:
        return pred[:, :channels, 0]
    return pred[:, :channels]


def _burgers_initial_from_metadata(metadata: dict, pred: torch.Tensor) -> torch.Tensor:
    if isinstance(metadata.get("initial_1d"), torch.Tensor):
        initial = metadata["initial_1d"].to(pred.device, pred.dtype)
        return initial.reshape(pred.shape[0], 1, pred.shape[-1])
    if isinstance(metadata.get("full_tensor"), torch.Tensor):
        full = metadata["full_tensor"].to(pred.device, pred.dtype)
        if full.ndim == 4:
            return full[:, :1, 0, :]
    if isinstance(metadata.get("input_fields"), torch.Tensor):
        input_fields = metadata["input_fields"].to(pred.device, pred.dtype)
        if input_fields.ndim == 4:
            return input_fields[:, :1, 0, :]
    warnings.warn("Missing Burgers initial condition metadata; using predicted first time slice.", RuntimeWarning, stacklevel=2)
    return pred[:, :1, 0, :]


def _parameter_field(
    metadata: dict,
    ref: torch.Tensor,
    keys: tuple[str, ...],
    default: float,
    input_channel: int | None = None,
    full_channel: int | None = None,
    pred_channel: int | None = None,
    input_min_channels: int | None = None,
    full_min_channels: int | None = None,
    pred_min_channels: int | None = None,
    pred: torch.Tensor | None = None,
    value: torch.Tensor | float | None = None,
) -> torch.Tensor:
    if value is not None:
        return _expand_parameter_field(value, ref)
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return _expand_parameter_field(metadata[key], ref)

    if isinstance(metadata.get("full_tensor"), torch.Tensor) and full_channel is not None:
        full = metadata["full_tensor"].to(ref.device, ref.dtype)
        min_channels = full_min_channels or (full_channel + 1)
        if full.ndim == 4 and full.shape[1] >= max(full_channel + 1, min_channels):
            return _expand_parameter_field(full[:, full_channel : full_channel + 1], ref)

    if isinstance(metadata.get("input_fields"), torch.Tensor) and input_channel is not None:
        input_fields = metadata["input_fields"].to(ref.device, ref.dtype)
        min_channels = input_min_channels or (input_channel + 1)
        if input_fields.ndim == 4 and input_fields.shape[1] >= max(input_channel + 1, min_channels):
            return _expand_parameter_field(input_fields[:, input_channel : input_channel + 1], ref)

    if pred is not None and pred_channel is not None:
        min_channels = pred_min_channels or (pred_channel + 1)
        if pred.ndim == 4 and pred.shape[1] >= max(pred_channel + 1, min_channels):
            return _expand_parameter_field(pred[:, pred_channel : pred_channel + 1], ref)

    return torch.ones_like(ref[:, :1]) * float(default)


def _expand_parameter_field(value: torch.Tensor | float, ref: torch.Tensor) -> torch.Tensor:
    target = ref[:, :1]
    tensor = torch.as_tensor(value, device=ref.device, dtype=ref.dtype)
    if tensor.ndim == 0:
        return torch.ones_like(target) * tensor
    if tensor.ndim == 1:
        if tensor.numel() == 1:
            return torch.ones_like(target) * tensor.reshape(())
        if tensor.shape[0] == target.shape[0]:
            return tensor.reshape(target.shape[0], 1, 1, 1).expand_as(target)
    if tensor.ndim == 2:
        if tuple(tensor.shape) == tuple(target.shape[-2:]):
            return tensor.reshape(1, 1, *target.shape[-2:]).expand_as(target)
        if tensor.shape[0] == target.shape[0] and tensor.shape[1] == 1:
            return tensor.reshape(target.shape[0], 1, 1, 1).expand_as(target)
    if tensor.ndim == 3 and tuple(tensor.shape[-2:]) == tuple(target.shape[-2:]):
        if tensor.shape[0] in {1, target.shape[0]}:
            tensor = tensor.unsqueeze(1)
    if tensor.ndim >= 4:
        if tensor.shape[0] not in {1, target.shape[0]}:
            raise ValueError(f"Parameter batch dimension {tensor.shape[0]} cannot broadcast to {target.shape[0]}")
        tensor = tensor[:, :1]
        if tuple(tensor.shape[-2:]) != tuple(target.shape[-2:]):
            tensor = F.interpolate(tensor, size=target.shape[-2:], mode="nearest")
        return tensor.expand_as(target)
    raise ValueError(f"Cannot broadcast parameter shape {tuple(tensor.shape)} to {tuple(target.shape)}")


def _trajectory_laplacian(field: torch.Tensor, spacing: float, boundary: str) -> torch.Tensor:
    if field.ndim != 5:
        raise ValueError(f"Trajectory laplacian expects [B,C,T,H,W], got {tuple(field.shape)}")
    b, c, t, h, w = field.shape
    flat = field.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    lap = laplacian(flat, spacing=spacing, boundary=boundary)
    return lap.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def _boundary_mode(metadata: dict, default: str = "periodic") -> str:
    raw = metadata.get("bc", metadata.get("boundary", metadata.get("boundary_condition", default)))
    mode = str(raw).lower().replace("-", "_")
    if "periodic" in mode:
        return "periodic"
    if "neumann" in mode or "extrapolation" in mode:
        return "neumann"
    if "dirichlet" in mode:
        return "dirichlet"
    return default


def _spatial_step(field: torch.Tensor, metadata: dict, boundary: str, default_domain: float) -> float:
    if "dx" in metadata and metadata["dx"] is not None:
        return float(metadata["dx"])
    domain = float(metadata.get("domain_length", metadata.get("spatial_domain_length", default_domain)))
    points = max(field.shape[-1], 1)
    if boundary == "periodic":
        return domain / points
    return domain / max(points - 1, 1)


def _trajectory_bc_loss(trajectory: torch.Tensor, boundary: str, spacing: float, reduction: str) -> torch.Tensor:
    if boundary == "periodic":
        return periodic_bc_loss(trajectory, dims=(-2, -1), reduction=reduction)
    if boundary == "neumann":
        return neumann_zero_bc_loss(trajectory, spacing_x=spacing, spacing_y=spacing, reduction=reduction)
    if boundary == "dirichlet":
        return dirichlet_zero_bc_loss(trajectory, reduction=reduction)
    raise ValueError(f"Unknown boundary mode '{boundary}'")


def _steady_heat_mixed_bc_loss(solution: torch.Tensor, u_d: torch.Tensor, spacing: float, reduction: str) -> torch.Tensor:
    if solution.shape[-2] < 2 or solution.shape[-1] < 2:
        return _zero_scalar(solution)
    bottom_target = u_d[..., -1, :]
    terms = [
        (solution[..., -1, :] - bottom_target).reshape(-1),
        ((solution[..., 1, :] - solution[..., 0, :]) / spacing).reshape(-1),
        ((solution[..., :, 1] - solution[..., :, 0]) / spacing).reshape(-1),
        ((solution[..., :, -1] - solution[..., :, -2]) / spacing).reshape(-1),
    ]
    return _mean_square(torch.cat(terms), reduction)


def _time_step(num_steps: int, final_time: float, metadata: dict) -> float:
    if num_steps <= 1:
        return final_time
    if isinstance(metadata.get("time_values"), torch.Tensor):
        t = metadata["time_values"].detach().cpu().float()
        if len(t) > 1:
            return float((t[-1] - t[0]) / (len(t) - 1))
    if isinstance(metadata.get("time_values"), (list, tuple)) and len(metadata["time_values"]) > 1:
        values = metadata["time_values"]
        return float((values[-1] - values[0]) / (len(values) - 1))
    if isinstance(metadata.get("full_tensor"), torch.Tensor):
        full = metadata["full_tensor"]
        if full.ndim == 5 and full.shape[2] > 1:
            input_idx = int(metadata.get("input_time_index", 0))
            if num_steps == full.shape[2]:
                return final_time / max(full.shape[2] - 1, 1)
            segment = final_time * (full.shape[2] - 1 - input_idx) / max(full.shape[2] - 1, 1)
            return segment / max(num_steps - 1, 1)
    return final_time / max(num_steps - 1, 1)


def _streamfunction_from_vorticity(omega: torch.Tensor) -> torch.Tensor:
    # omega: [B,T,H,W], solve omega = -Delta psi on periodic [0,1]^2.
    b, t, h, w = omega.shape
    omega_ft = torch.fft.rfft2(omega.reshape(b * t, h, w))
    kx = torch.fft.fftfreq(h, d=1.0 / h, device=omega.device, dtype=omega.dtype).view(h, 1)
    ky = torch.fft.rfftfreq(w, d=1.0 / w, device=omega.device, dtype=omega.dtype).view(1, w // 2 + 1)
    denom = (2.0 * math.pi) ** 2 * (kx**2 + ky**2)
    denom[0, 0] = 1.0
    psi_ft = omega_ft / denom
    psi_ft[:, 0, 0] = 0.0
    return torch.fft.irfft2(psi_ft, s=(h, w)).reshape(b, t, h, w)


def _ns_forcing(h: int, w: int, device, dtype) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    phase = 2.0 * math.pi * (xx + yy)
    return 0.1 * (torch.sin(phase) + torch.cos(phase))


def _unit_spacing(field: torch.Tensor) -> float:
    return 1.0 / max(field.shape[-1] - 1, 1)


def _metadata_float(metadata: dict, keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return float(metadata[key])
    return default


def _mean_square(x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    if x.numel() == 0:
        return _zero_scalar(x)
    sq = x.pow(2)
    if reduction == "sum":
        return sq.sum()
    if reduction != "mean":
        raise ValueError(f"Unsupported reduction '{reduction}'")
    return sq.mean()


def _zero_scalar(ref: torch.Tensor) -> torch.Tensor:
    return ref.sum() * 0.0


def _take_dim(x: torch.Tensor, dim: int, index: int) -> torch.Tensor:
    idx = [slice(None)] * x.ndim
    idx[dim] = index
    return x[tuple(idx)]
