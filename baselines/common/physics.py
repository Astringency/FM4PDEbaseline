from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def residual_loss(pred: torch.Tensor, pde_name: str, metadata: dict | None = None) -> torch.Tensor:
    residual = residual_tensor(pred, pde_name, metadata)
    return (residual**2).mean()


def residual_tensor(pred: torch.Tensor, pde_name: str, metadata: dict | None = None) -> torch.Tensor:
    metadata = metadata or {}
    pde = pde_name.lower()
    task = str(metadata.get("task", "")).lower()
    if pde == "poisson":
        return _poisson_residual(pred, metadata, inverse=task in {"inverse", "sparse_inverse"})
    if pde == "helmholtz":
        return _helmholtz_residual(pred, metadata, inverse=task in {"inverse", "sparse_inverse"})
    if pde == "darcy":
        return _darcy_task_residual(pred, metadata, inverse=task in {"inverse", "sparse_inverse"})
    if pde == "burger":
        return burgers_residual(_as_burgers_field(pred))
    if pde in {"nsnonbounded", "navier_stokes", "ns"}:
        return navier_stokes_vorticity_residual(_as_ns_trajectory(pred, metadata), metadata)
    if pde in {"reaction_diffusion", "rd"}:
        return reaction_diffusion_residual(_as_rd_trajectory(pred, metadata), metadata)
    if pde in {"shallow_water", "swe"}:
        return shallow_water_residual(_as_swe_trajectory(pred, metadata), metadata)
    raise NotImplementedError(f"No PDE residual registered for {pde_name}")


def poisson_inverse_residual(source: torch.Tensor, solution: torch.Tensor) -> torch.Tensor:
    return -laplacian(solution, spacing=1.0 / max(solution.shape[-1] - 1, 1), boundary="dirichlet") - source


def helmholtz_inverse_residual(source: torch.Tensor, solution: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    return -laplacian(solution, spacing=1.0 / max(solution.shape[-1] - 1, 1), boundary="dirichlet") - (k**2) * solution - source


def darcy_residual(coeff: torch.Tensor, solution: torch.Tensor) -> torch.Tensor:
    h = 1.0 / max(solution.shape[-1] - 1, 1)
    grad_x = central_diff(solution, dim=-1, spacing=h, boundary="dirichlet")
    grad_y = central_diff(solution, dim=-2, spacing=h, boundary="dirichlet")
    flux_x = coeff * grad_x
    flux_y = coeff * grad_y
    div = central_diff(flux_x, dim=-1, spacing=h, boundary="dirichlet") + central_diff(flux_y, dim=-2, spacing=h, boundary="dirichlet")
    return -div - 1.0


def burgers_residual(u: torch.Tensor, nu: float = 0.01) -> torch.Tensor:
    # u: [B,1,T,X]
    dt = 1.0 / max(u.shape[-2] - 1, 1)
    dx = 1.0 / max(u.shape[-1] - 1, 1)
    u_t = central_diff(u, dim=-2, spacing=dt, boundary="periodic")
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
    u_x = central_diff(psi, dim=-2, spacing=h, boundary="periodic")
    u_y = -central_diff(psi, dim=-1, spacing=h, boundary="periodic")
    omega_t = central_diff(omega, dim=1, spacing=dt, boundary="replicate")
    omega_x = central_diff(omega, dim=-1, spacing=h, boundary="periodic")
    omega_y = central_diff(omega, dim=-2, spacing=h, boundary="periodic")
    lap = laplacian(omega.unsqueeze(1).reshape(-1, 1, *omega.shape[-2:]), spacing=h, boundary="periodic")
    lap = lap.reshape_as(omega)
    forcing = _ns_forcing(w.shape[-2], w.shape[-1], w.device, w.dtype).view(1, 1, w.shape[-2], w.shape[-1])
    return (omega_t + u_x * omega_x + u_y * omega_y - nu * lap - forcing).unsqueeze(1)


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
    mode = "replicate"
    padded = F.pad(u, (1, 1, 1, 1), mode=mode)
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
    return out


def _poisson_residual(pred: torch.Tensor, metadata: dict, inverse: bool) -> torch.Tensor:
    if inverse:
        solution = _solution_from_metadata(metadata, pred)
        return poisson_inverse_residual(pred[:, :1], solution[:, :1])
    source = _coeff_from_metadata(metadata, pred)
    return poisson_inverse_residual(source[:, :1], pred[:, :1])


def _helmholtz_residual(pred: torch.Tensor, metadata: dict, inverse: bool) -> torch.Tensor:
    k = float(metadata.get("k", 1.0))
    if inverse:
        solution = _solution_from_metadata(metadata, pred)
        return helmholtz_inverse_residual(pred[:, :1], solution[:, :1], k=k)
    source = _coeff_from_metadata(metadata, pred)
    return helmholtz_inverse_residual(source[:, :1], pred[:, :1], k=k)


def _darcy_task_residual(pred: torch.Tensor, metadata: dict, inverse: bool) -> torch.Tensor:
    if inverse:
        solution = _solution_from_metadata(metadata, pred)
        return darcy_residual(pred[:, :1], solution[:, :1])
    coeff = _coeff_from_metadata(metadata, pred)
    return darcy_residual(coeff[:, :1], pred[:, :1])


def _coeff_from_metadata(metadata: dict, pred: torch.Tensor) -> torch.Tensor:
    if isinstance(metadata.get("coeff_fields"), torch.Tensor):
        return metadata["coeff_fields"].to(pred.device, pred.dtype)
    if isinstance(metadata.get("input_fields"), torch.Tensor) and metadata["input_fields"].shape[1] <= pred.shape[1]:
        return metadata["input_fields"].to(pred.device, pred.dtype)
    if isinstance(metadata.get("full_tensor"), torch.Tensor):
        full = metadata["full_tensor"].to(pred.device, pred.dtype)
        if full.ndim == 4:
            return full[:, :1]
    raise ValueError("Missing coefficient/source field for PDE residual")


def _solution_from_metadata(metadata: dict, pred: torch.Tensor) -> torch.Tensor:
    if isinstance(metadata.get("solution_fields"), torch.Tensor):
        return metadata["solution_fields"].to(pred.device, pred.dtype)
    if isinstance(metadata.get("input_fields"), torch.Tensor):
        return metadata["input_fields"].to(pred.device, pred.dtype)
    if isinstance(metadata.get("full_tensor"), torch.Tensor):
        full = metadata["full_tensor"].to(pred.device, pred.dtype)
        if full.ndim == 4:
            return full[:, 1:2]
    raise ValueError("Missing solution field for inverse PDE residual")


def _as_burgers_field(pred: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 4:
        return pred[:, :1]
    raise ValueError(f"Burgers field expects [B,1,T,X], got {tuple(pred.shape)}")


def _as_ns_trajectory(pred: torch.Tensor, metadata: dict) -> torch.Tensor:
    if pred.ndim == 5:
        return pred[:, :1]
    if pred.ndim != 4:
        raise ValueError(f"NS residual expects [B,T,H,W] or [B,1,T,H,W], got {tuple(pred.shape)}")
    full = metadata.get("full_tensor")
    input_fields = metadata.get("input_fields")
    if pred.shape[1] > 1:
        traj = pred.unsqueeze(1)
        if isinstance(full, torch.Tensor):
            initial = full[:, :, :1].to(pred.device, pred.dtype)
        elif isinstance(input_fields, torch.Tensor) and input_fields.shape[1] == 1:
            initial = input_fields[:, :, None].to(pred.device, pred.dtype)
        else:
            initial = traj[:, :, :1]
        return torch.cat([initial, traj], dim=2)
    if isinstance(full, torch.Tensor):
        initial = full[:, :, :1].to(pred.device, pred.dtype)
        return torch.cat([initial, pred[:, :, None]], dim=2)
    if isinstance(input_fields, torch.Tensor):
        return torch.stack([input_fields[:, :1].to(pred.device, pred.dtype), pred[:, :1]], dim=2).squeeze(3)
    return pred[:, :1, None]


def _as_rd_trajectory(pred: torch.Tensor, metadata: dict) -> torch.Tensor:
    if pred.ndim == 5:
        return pred[:, :2]
    if pred.ndim != 4 or pred.shape[1] < 2:
        raise ValueError(f"Reaction-diffusion residual expects [B,2,H,W] or [B,2,T,H,W], got {tuple(pred.shape)}")
    initial = _time_initial_from_metadata(metadata, pred, channels=2)
    return torch.stack([initial[:, :2], pred[:, :2]], dim=2)


def _as_swe_trajectory(pred: torch.Tensor, metadata: dict) -> torch.Tensor:
    if pred.ndim == 5:
        return pred[:, :3]
    if pred.ndim != 4 or pred.shape[1] < 3:
        raise ValueError(f"Shallow-water residual expects [B,3,H,W] or [B,3,T,H,W], got {tuple(pred.shape)}")
    initial = _time_initial_from_metadata(metadata, pred, channels=3)
    return torch.stack([initial[:, :3], pred[:, :3]], dim=2)


def _time_initial_from_metadata(metadata: dict, pred: torch.Tensor, channels: int) -> torch.Tensor:
    if isinstance(metadata.get("background_fields"), torch.Tensor):
        return metadata["background_fields"].to(pred.device, pred.dtype)
    if isinstance(metadata.get("input_fields"), torch.Tensor) and metadata["input_fields"].shape[1] == channels:
        return metadata["input_fields"].to(pred.device, pred.dtype)
    if isinstance(metadata.get("full_tensor"), torch.Tensor):
        full = metadata["full_tensor"].to(pred.device, pred.dtype)
        if full.ndim == 5:
            idx = int(metadata.get("input_time_index", 0))
            return full[:, :channels, idx]
    raise ValueError("Missing initial/background state for time-dependent residual")


def _time_step(num_steps: int, final_time: float, metadata: dict) -> float:
    if num_steps == 2 and isinstance(metadata.get("full_tensor"), torch.Tensor):
        full = metadata["full_tensor"]
        if full.ndim == 5 and full.shape[2] > 1:
            input_idx = int(metadata.get("input_time_index", 0))
            segment = final_time * (full.shape[2] - 1 - input_idx) / max(full.shape[2] - 1, 1)
            return segment
    if isinstance(metadata.get("time_values"), torch.Tensor):
        t = metadata["time_values"].detach().cpu().float()
        if len(t) > 1:
            return float((t[-1] - t[0]) / (len(t) - 1))
    if isinstance(metadata.get("time_values"), (list, tuple)) and len(metadata["time_values"]) > 1:
        values = metadata["time_values"]
        return float((values[-1] - values[0]) / (len(values) - 1))
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
