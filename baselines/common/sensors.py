from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Literal, Sequence

import torch


SensorMode = Literal[
    "random",
    "random_per_sample",
    "fixed",
    "grid",
    "time_varying",
    "time_slices_per_sample",
]
SensorBudgetMode = Literal["per_time", "total"]
MULTICONDITION_MODES = ("a_only", "u_only", "both")
DEFAULT_CONDITION_PROBABILITIES = {
    "a_only": 0.3333333333,
    "u_only": 0.3333333333,
    "both": 0.3333333334,
}


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
    sensor_budget_mode: SensorBudgetMode = "per_time",
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

    if mode in {"random", "random_per_sample", "fixed"}:
        if time_dim is not None and sensor_budget_mode == "per_time":
            if time_dim < 0 or time_dim >= len(obs_shape):
                raise ValueError(
                    f"time_dim={time_dim} incompatible with observation shape {obs_shape}"
                )
            per_t_shape = obs_shape[:time_dim] + obs_shape[time_dim + 1 :]
            per_t_total = int(math.prod(per_t_shape))
            per_t_num = min(int(num_sensors), per_t_total)
            for t in range(obs_shape[time_dim]):
                local = torch.zeros(per_t_shape, dtype=torch.float32)
                perm = torch.randperm(per_t_total, generator=generator)[:per_t_num]
                local.reshape(-1)[perm] = 1.0
                sl = [slice(None)] * len(obs_shape)
                sl[time_dim] = t
                base[tuple(sl)] = local
        elif sensor_budget_mode == "total" or time_dim is None:
            perm = torch.randperm(total, generator=generator)[:num]
            base.reshape(-1)[perm] = 1.0
        else:
            raise ValueError(
                f"sensor_budget_mode must be per_time or total, got {sensor_budget_mode!r}"
            )
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
    elif mode == "time_slices_per_sample":
        if time_dim is None:
            raise ValueError("time_slices_per_sample requires an explicit time_dim")
        if time_dim < 0 or time_dim >= len(obs_shape):
            raise ValueError(f"time_dim={time_dim} incompatible with observation shape {obs_shape}")
        selected = torch.randperm(obs_shape[time_dim], generator=generator)[: min(num, obs_shape[time_dim])]
        for time_index in selected.tolist():
            sl = [slice(None)] * len(obs_shape)
            sl[time_dim] = int(time_index)
            base[tuple(sl)] = 1.0
    elif mode == "time_varying":
        if time_dim is None:
            time_dim = 0
        if time_dim < 0 or time_dim >= len(obs_shape):
            raise ValueError(f"time_dim={time_dim} incompatible with observation shape {obs_shape}")
        per_t_shape = obs_shape[:time_dim] + obs_shape[time_dim + 1 :]
        per_t_total = int(math.prod(per_t_shape))
        t_count = obs_shape[time_dim]
        if sensor_budget_mode == "per_time":
            per_t_num = min(num, per_t_total)
            for t in range(t_count):
                local = torch.zeros(per_t_shape, dtype=torch.float32)
                perm = torch.randperm(per_t_total, generator=generator)[:per_t_num]
                local.reshape(-1)[perm] = 1.0
                sl = [slice(None)] * len(obs_shape)
                sl[time_dim] = t
                base[tuple(sl)] = local
        elif sensor_budget_mode == "total":
            total_time_space = int(t_count * per_t_total)
            perm = torch.randperm(total_time_space, generator=generator)[: min(num, total_time_space)]
            base.reshape(-1)[perm] = 1.0
        else:
            raise ValueError(f"sensor_budget_mode must be per_time or total, got {sensor_budget_mode!r}")
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
    """Return observations from fields using a shared or per-sample mask.

    ``mask`` is either ``[C,*grid]`` (one fixed layout shared by the batch) or
    ``[B,C,*grid]`` (one layout per sample). All channel masks must share the
    same locations and all samples must contain the same sensor count so the
    observations can be represented by a dense ``[B,N,C]`` tensor.
    """
    if fields.ndim < 4:
        raise ValueError(f"Expected fields [B,C,*grid], got {tuple(fields.shape)}")
    shared = tuple(mask.shape) == tuple(fields.shape[1:])
    batched = tuple(mask.shape) == tuple(fields.shape)
    if not shared and not batched:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} must match fields with or without batch: "
            f"{tuple(fields.shape)} or {tuple(fields.shape[1:])}"
        )
    c = fields.shape[1]
    grid_shape = tuple(fields.shape[2:])
    flat = fields.reshape(fields.shape[0], c, -1)
    values: list[torch.Tensor] = []
    coords: list[torch.Tensor] = []
    counts: list[int] = []
    for sample in range(fields.shape[0]):
        sample_mask = mask if shared else mask[sample]
        if not torch.equal(sample_mask, sample_mask[:1].expand_as(sample_mask)):
            raise ValueError("Sensor locations must be identical across channels")
        flat_idx = sample_mask[0].bool().reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        counts.append(int(flat_idx.numel()))
        values.append(flat[sample, :, flat_idx].transpose(0, 1).contiguous())
        coords.append(_normalized_coords_from_flat_indices(flat_idx, grid_shape))
    if len(set(counts)) > 1:
        raise ValueError(f"Per-sample masks must contain the same number of sensors, got {counts}")
    return torch.stack(values, dim=0), torch.stack(coords, dim=0)


def _normalized_coords_from_flat_indices(
    flat_indices: torch.Tensor, grid_shape: tuple[int, ...]
) -> torch.Tensor:
    """Compute only sensor coordinates instead of materializing the full grid."""
    coordinates: list[torch.Tensor] = []
    for axis, size in enumerate(grid_shape):
        stride = int(math.prod(grid_shape[axis + 1 :]))
        integer_coordinate = torch.div(flat_indices, stride, rounding_mode="floor") % size
        denominator = max(int(size) - 1, 1)
        coordinates.append(integer_coordinate.to(torch.float32) / denominator)
    return torch.stack(coordinates, dim=-1)


def build_observation_tensors(
    target_fields: torch.Tensor,
    num_sensors: int,
    mode: SensorMode,
    seed: int,
    noise_level: float = 0.0,
    sensor_budget_mode: SensorBudgetMode = "per_time",
    time_dim: int | None = None,
    sample_ids: Sequence[str | int] | None = None,
    split: str = "",
    epoch: int = 0,
    build_voronoi_grid: bool = True,
) -> dict[str, Any]:
    if time_dim is None and target_fields.ndim == 5:
        time_dim = 0
    if mode in {"random_per_sample", "time_slices_per_sample"}:
        if sample_ids is None:
            sample_ids = list(range(int(target_fields.shape[0])))
        if len(sample_ids) != int(target_fields.shape[0]):
            raise ValueError(
                f"sample_ids length {len(sample_ids)} does not match batch size {target_fields.shape[0]}"
            )
        effective_epoch = int(epoch) if str(split).lower() == "train" else 0
        masks = [
            make_sensor_mask(
                tuple(target_fields.shape[1:]),
                num_sensors,
                mode,
                _sample_mask_seed(seed, str(split), sample_id, effective_epoch),
                time_dim=time_dim,
                sensor_budget_mode=sensor_budget_mode,
            )
            for sample_id in sample_ids
        ]
        mask = torch.stack(masks, dim=0).to(target_fields.device)
        masked_grid = target_fields * mask
    else:
        mask = make_sensor_mask(
            tuple(target_fields.shape[1:]),
            num_sensors,
            mode,
            seed,
            time_dim=time_dim,
            sensor_budget_mode=sensor_budget_mode,
        ).to(target_fields.device)
        masked_grid = target_fields * mask.unsqueeze(0)
    obs_values, obs_coords = extract_observations(target_fields, mask)
    if mode in {"random_per_sample", "time_slices_per_sample"} and noise_level:
        assert sample_ids is not None
        obs_values = torch.cat(
            [
                add_noise(
                    obs_values[index : index + 1],
                    noise_level,
                    relative=True,
                    seed=_sample_noise_seed(seed, str(split), sample_id, effective_epoch),
                )
                for index, sample_id in enumerate(sample_ids)
            ],
            dim=0,
        )
    else:
        obs_values = add_noise(obs_values, noise_level, relative=True, seed=seed)
    if noise_level:
        # Keep the masked-grid values consistent with the noisy observation list.
        masked_grid = torch.zeros_like(target_fields)
        c = target_fields.shape[1]
        masked_flat = masked_grid.reshape(target_fields.shape[0], c, -1)
        for sample in range(target_fields.shape[0]):
            sample_mask = mask if mask.ndim == target_fields.ndim - 1 else mask[sample]
            flat_idx = sample_mask[0].bool().reshape(-1).nonzero(as_tuple=False).squeeze(-1)
            masked_flat[sample, :, flat_idx] = obs_values[sample].transpose(0, 1)
        masked_grid = masked_flat.reshape_as(target_fields)
    voronoi_grid = None
    if build_voronoi_grid:
        from .voronoi import voronoi_fill

        voronoi_grid = voronoi_fill(masked_grid, mask)
    if mask.ndim == target_fields.ndim:
        mask_ids = [_mask_id(sample_mask) for sample_mask in mask]
        sample_mask = mask[0]
    else:
        mask_ids = [_mask_id(mask)] * int(target_fields.shape[0])
        sample_mask = mask
    mask_id = _mask_id(mask)
    num_observations_total = int(sample_mask[0].sum().detach().cpu())
    num_sensors_per_time: int | list[int]
    if time_dim is not None:
        spatial_mask = sample_mask[0]
        dims = tuple(i for i in range(spatial_mask.ndim) if i != time_dim)
        counts = spatial_mask.sum(dim=dims).detach().cpu().to(torch.long).tolist()
        num_sensors_per_time = [int(x) for x in counts]
    else:
        num_sensors_per_time = int(num_observations_total)
    return {
        "mask": mask,
        "obs_values": obs_values,
        "obs_coords": obs_coords,
        "masked_grid": masked_grid,
        "voronoi_grid": voronoi_grid,
        "mask_id": mask_id,
        "mask_ids": mask_ids,
        "num_observations_total": num_observations_total,
        "num_sensors_per_time": num_sensors_per_time,
        "sensor_budget_mode": sensor_budget_mode,
    }


def validate_condition_probabilities(
    probabilities: dict[str, float] | None,
    *,
    tolerance: float = 1e-8,
) -> dict[str, float]:
    """Validate and canonicalize the three multicondition probabilities."""
    values = dict(DEFAULT_CONDITION_PROBABILITIES if probabilities is None else probabilities)
    if set(values) != set(MULTICONDITION_MODES):
        raise ValueError(
            "condition_probabilities must define exactly a_only, u_only, and both"
        )
    normalized = {mode: float(values[mode]) for mode in MULTICONDITION_MODES}
    if any(value < 0.0 for value in normalized.values()):
        raise ValueError("condition_probabilities must all be non-negative")
    total = sum(normalized.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=float(tolerance)):
        raise ValueError(
            f"condition_probabilities must sum to 1 within {tolerance:g}, got {total:.12g}"
        )
    return normalized


def deterministic_condition_mode(
    base_seed: int,
    split: str,
    sample_id: str | int,
    epoch: int,
    probabilities: dict[str, float] | None = None,
) -> str:
    """Choose a reproducible per-sample training condition with a stable hash."""
    probabilities = validate_condition_probabilities(probabilities)
    effective_epoch = int(epoch) if str(split).lower() == "train" else 0
    payload = (
        f"condition|{int(base_seed)}|{str(split).lower()}|{sample_id}|{effective_epoch}"
    ).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    draw = integer / float(2**64)
    cumulative = 0.0
    for mode in MULTICONDITION_MODES:
        cumulative += probabilities[mode]
        if draw < cumulative:
            return mode
    return MULTICONDITION_MODES[-1]


def build_multicondition_observation_tensors(
    joint_fields: torch.Tensor,
    num_sensors: int,
    mode: SensorMode,
    seed: int,
    condition_modes: Sequence[str],
    noise_level: float = 0.0,
    sensor_budget_mode: SensorBudgetMode = "total",
    sample_ids: Sequence[str | int] | None = None,
    split: str = "",
    epoch: int = 0,
    build_voronoi_grid: bool = True,
) -> dict[str, Any]:
    """Build value, presence, and base-mask views for the two-field task.

    ``num_sensors`` always counts spatial locations.  Both modalities reuse the
    same base locations; ``condition_modes`` only changes which modality masks
    and values are active.
    """
    if joint_fields.ndim != 4 or int(joint_fields.shape[1]) != 2:
        raise ValueError(
            "sparse_solution_multicondition requires joint fields shaped [B,2,H,W], "
            f"got {tuple(joint_fields.shape)}"
        )
    if len(condition_modes) != int(joint_fields.shape[0]):
        raise ValueError(
            f"condition_modes length {len(condition_modes)} does not match batch size {joint_fields.shape[0]}"
        )
    invalid = sorted(set(str(value) for value in condition_modes) - set(MULTICONDITION_MODES))
    if invalid:
        raise ValueError(f"Unknown multicondition modes: {invalid}")

    base = build_observation_tensors(
        joint_fields,
        num_sensors=num_sensors,
        mode=mode,
        seed=seed,
        noise_level=noise_level,
        sensor_budget_mode=sensor_budget_mode,
        sample_ids=sample_ids,
        split=split,
        epoch=epoch,
        build_voronoi_grid=False,
    )
    base_mask = base["mask"]
    if base_mask.ndim == joint_fields.ndim - 1:
        base_mask_batched = base_mask.unsqueeze(0).expand(
            joint_fields.shape[0], *base_mask.shape
        )
    else:
        base_mask_batched = base_mask
    modality_presence = torch.tensor(
        [
            [1.0, 0.0] if condition == "a_only" else [0.0, 1.0] if condition == "u_only" else [1.0, 1.0]
            for condition in condition_modes
        ],
        dtype=joint_fields.dtype,
        device=joint_fields.device,
    )
    presence_grid = modality_presence.reshape(-1, 2, 1, 1)
    active_mask = base_mask_batched.to(joint_fields.dtype) * presence_grid
    # base['masked_grid'] contains only noisy or clean observations at the base
    # locations. Multiplication by the presence grid cannot reveal an inactive
    # modality or an unobserved spatial value.
    masked_grid = base["masked_grid"] * presence_grid
    sensor_presence = modality_presence.unsqueeze(1).expand(
        -1, int(base["obs_values"].shape[1]), -1
    )
    obs_values = base["obs_values"] * sensor_presence
    voronoi_grid = None
    if build_voronoi_grid:
        from .voronoi import voronoi_fill_per_channel

        voronoi_grid = voronoi_fill_per_channel(masked_grid, active_mask)

    location_count = int(base["num_observations_total"])
    count_a = [location_count if mode_name in {"a_only", "both"} else 0 for mode_name in condition_modes]
    count_u = [location_count if mode_name in {"u_only", "both"} else 0 for mode_name in condition_modes]
    active_mask_ids = [_mask_id(active_mask[index]) for index in range(active_mask.shape[0])]
    return {
        "mask": active_mask,
        "base_mask": base_mask_batched,
        "obs_values": obs_values,
        "obs_presence": sensor_presence,
        "obs_coords": base["obs_coords"],
        "masked_grid": masked_grid,
        "voronoi_grid": voronoi_grid,
        "mask_id": _mask_id(active_mask),
        "mask_ids": active_mask_ids,
        "base_mask_id": base["mask_id"],
        "base_mask_ids": list(base.get("mask_ids", [])),
        "num_sensor_locations": location_count,
        "num_scalar_observations_a": count_a,
        "num_scalar_observations_u": count_u,
        "num_scalar_observations_total": [a + u for a, u in zip(count_a, count_u)],
        "condition_modes": [str(value) for value in condition_modes],
        "sensor_budget_mode": sensor_budget_mode,
    }


def _sample_mask_seed(base_seed: int, split: str, sample_id: str | int, epoch: int) -> int:
    payload = f"mask|{int(base_seed)}|{split.lower()}|{sample_id}|{int(epoch)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _sample_noise_seed(base_seed: int, split: str, sample_id: str | int, epoch: int) -> int:
    payload = f"noise|{int(base_seed)}|{split.lower()}|{sample_id}|{int(epoch)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _mask_id(mask: torch.Tensor) -> str:
    return hashlib.sha1(mask.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def save_mask(mask: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mask.detach().cpu(), path)


def load_mask(path: str | Path, map_location=None) -> torch.Tensor:
    return torch.load(path, map_location=map_location)
