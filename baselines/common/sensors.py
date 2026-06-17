from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Literal

import torch


SensorMode = Literal["random", "fixed", "grid", "time_varying"]


def make_coordinate_grid(shape: tuple[int, ...], batch_size: int | None = None, device=None) -> torch.Tensor:
    axes = [torch.linspace(0.0, 1.0, steps=s, device=device) for s in shape]
    mesh = torch.meshgrid(*axes, indexing="ij")
    coords = torch.stack(mesh, dim=-1).reshape(-1, len(shape))
    if batch_size is not None:
        coords = coords.unsqueeze(0).repeat(batch_size, 1, 1)
    return coords


def make_sensor_mask(
    shape,
    num_sensors: int,
    mode: SensorMode,
    seed: int,
    time_dim: int | None = None,
) -> torch.Tensor:
    """Create a reusable sensor mask.

    ``shape`` may include a leading channel dimension. The returned mask has the
    same shape and shares sensor locations across channels. For 5D tensors, pass
    the shape after batch, e.g. ``(C,T,H,W)``.
    """
    shape = tuple(int(s) for s in shape)
    if len(shape) < 2:
        raise ValueError(f"Sensor mask needs at least two dimensions, got {shape}")
    channel_dim = 0
    obs_shape = shape[1:]
    device = None
    generator = torch.Generator().manual_seed(int(seed))
    base = torch.zeros(obs_shape, dtype=torch.float32, device=device)
    total = int(math.prod(obs_shape))
    num = min(int(num_sensors), total)

    if mode in {"random", "fixed"}:
        perm = torch.randperm(total, generator=generator)[:num]
        base.reshape(-1)[perm] = 1.0
    elif mode == "grid":
        dims = len(obs_shape)
        per_dim = max(1, int(round(num ** (1.0 / dims))))
        indices = []
        for size in obs_shape:
            idx = torch.linspace(0, size - 1, steps=min(size, per_dim)).round().long().unique()
            indices.append(idx)
        mesh = torch.meshgrid(*indices, indexing="ij")
        flat = torch.stack([m.reshape(-1) for m in mesh], dim=-1)[:num]
        base[tuple(flat[:, d] for d in range(flat.shape[1]))] = 1.0
    elif mode == "time_varying":
        if time_dim is None:
            time_dim = 0
        if time_dim < 0 or time_dim >= len(obs_shape):
            raise ValueError(f"time_dim={time_dim} incompatible with observation shape {obs_shape}")
        per_t_shape = obs_shape[:time_dim] + obs_shape[time_dim + 1 :]
        per_t_total = int(math.prod(per_t_shape))
        t_count = obs_shape[time_dim]
        per_t_num = min(num, per_t_total)
        for t in range(t_count):
            local = torch.zeros(per_t_shape, dtype=torch.float32)
            perm = torch.randperm(per_t_total, generator=generator)[:per_t_num]
            local.reshape(-1)[perm] = 1.0
            sl = [slice(None)] * len(obs_shape)
            sl[time_dim] = t
            base[tuple(sl)] = local
    else:
        raise ValueError(f"Unknown sensor mode '{mode}'")

    return base.unsqueeze(channel_dim).repeat(shape[0], *([1] * len(obs_shape)))


def add_noise(obs_values: torch.Tensor, noise_level, relative: bool = True, seed: int = 0) -> torch.Tensor:
    noise_level = float(noise_level)
    if noise_level == 0:
        return obs_values
    generator = torch.Generator(device=obs_values.device).manual_seed(int(seed))
    if relative:
        scale = obs_values.std().clamp_min(1e-12) * noise_level
    else:
        scale = torch.as_tensor(noise_level, dtype=obs_values.dtype, device=obs_values.device)
    return obs_values + torch.randn(obs_values.shape, dtype=obs_values.dtype, device=obs_values.device, generator=generator) * scale


def extract_observations(fields: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``obs_values`` and ``obs_coords`` from ``[B,C,*grid]`` fields."""
    if fields.ndim < 4:
        raise ValueError(f"Expected fields [B,C,*grid], got {tuple(fields.shape)}")
    if tuple(mask.shape) != tuple(fields.shape[1:]):
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match fields without batch {tuple(fields.shape[1:])}")
    c = fields.shape[1]
    grid_shape = tuple(fields.shape[2:])
    spatial_mask = mask[0].bool()
    flat_idx = spatial_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    flat = fields.reshape(fields.shape[0], c, -1)
    values = flat[:, :, flat_idx].permute(0, 2, 1).contiguous()
    coords_all = make_coordinate_grid(grid_shape, batch_size=fields.shape[0], device=fields.device)
    coords = coords_all[:, flat_idx, :]
    return values, coords


def build_observation_tensors(
    target_fields: torch.Tensor,
    num_sensors: int,
    mode: SensorMode,
    seed: int,
    noise_level: float = 0.0,
) -> dict[str, torch.Tensor | str]:
    from .voronoi import voronoi_fill

    time_dim = 0 if target_fields.ndim == 5 else None
    mask = make_sensor_mask(tuple(target_fields.shape[1:]), num_sensors, mode, seed, time_dim=time_dim).to(target_fields.device)
    masked_grid = target_fields * mask.unsqueeze(0)
    obs_values, obs_coords = extract_observations(target_fields, mask)
    obs_values = add_noise(obs_values, noise_level, relative=True, seed=seed)
    if noise_level:
        # Keep the masked-grid values consistent with the noisy observation list.
        masked_grid = torch.zeros_like(target_fields)
        c = target_fields.shape[1]
        spatial_mask = mask[0].bool()
        flat_idx = spatial_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        masked_flat = masked_grid.reshape(target_fields.shape[0], c, -1)
        masked_flat[:, :, flat_idx] = obs_values.permute(0, 2, 1)
        masked_grid = masked_flat.reshape_as(target_fields)
    voronoi_grid = voronoi_fill(masked_grid, mask)
    mask_id = hashlib.sha1(mask.detach().cpu().numpy().tobytes()).hexdigest()[:16]
    return {
        "mask": mask,
        "obs_values": obs_values,
        "obs_coords": obs_coords,
        "masked_grid": masked_grid,
        "voronoi_grid": voronoi_grid,
        "mask_id": mask_id,
    }


def save_mask(mask: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mask.detach().cpu(), path)


def load_mask(path: str | Path, map_location=None) -> torch.Tensor:
    return torch.load(path, map_location=map_location)

