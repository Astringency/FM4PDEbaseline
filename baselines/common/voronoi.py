from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def _nearest_flat_indices(channel_mask: torch.Tensor) -> torch.Tensor | None:
    """Map every grid point to its nearest observed point once per sample."""
    shape = tuple(channel_mask.shape)
    flat_mask = channel_mask.reshape(-1).bool()
    if flat_mask.sum() == 0:
        return None
    if flat_mask.sum() == flat_mask.numel():
        return torch.arange(flat_mask.numel(), device=channel_mask.device)
    if channel_mask.device.type == "cpu":
        missing = ~channel_mask.detach().cpu().bool().numpy()
        nearest_coords = distance_transform_edt(
            missing,
            return_distances=False,
            return_indices=True,
        )
        nearest_flat = np.ravel_multi_index(nearest_coords, shape).reshape(-1)
        return torch.from_numpy(nearest_flat.copy()).to(dtype=torch.long)
    coords = torch.stack(
        torch.meshgrid(
            *[
                torch.arange(s, device=channel_mask.device, dtype=torch.float32)
                for s in shape
            ],
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, len(shape))
    sensor_coords = coords[flat_mask]
    sensor_indices = flat_mask.nonzero(as_tuple=False).squeeze(-1)
    return sensor_indices[torch.cdist(coords, sensor_coords).argmin(dim=1)]


def voronoi_fill(masked_grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Voronoi-like nearest-sensor filling for channel-first tensors.

    Supports ``[B,C,H,W]`` and ``[B,C,T,H,W]``. The method is intentionally
    simple and deterministic for reproducible baseline inputs. It uses nearest
    observed grid point in Euclidean grid-index space.
    """
    if masked_grid.ndim < 4:
        raise ValueError(f"Expected masked_grid [B,C,*grid], got {tuple(masked_grid.shape)}")
    shared = tuple(mask.shape) == tuple(masked_grid.shape[1:])
    batched = tuple(mask.shape) == tuple(masked_grid.shape)
    if not shared and not batched:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} must match grid with or without batch: "
            f"{tuple(masked_grid.shape)} or {tuple(masked_grid.shape[1:])}"
        )
    out = torch.empty_like(masked_grid)
    for b in range(masked_grid.shape[0]):
        sample_mask = mask if shared else mask[b]
        if not torch.equal(sample_mask, sample_mask[:1].expand_as(sample_mask)):
            raise ValueError("Voronoi fill requires identical sensor locations across channels")
        nearest = _nearest_flat_indices(sample_mask[0])
        if nearest is None:
            out[b].zero_()
            continue
        values = masked_grid[b].reshape(masked_grid.shape[1], -1)
        out[b] = values[:, nearest].reshape_as(masked_grid[b])
    return out


def voronoi_fill_per_channel(masked_grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill each value channel using only that channel's observed locations.

    Unlike :func:`voronoi_fill`, this helper intentionally permits different
    masks for different channels.  A completely absent channel stays exactly
    zero, which is the missing-modality contract used only by
    ``sparse_solution_multicondition``.
    """
    if masked_grid.ndim < 4:
        raise ValueError(f"Expected masked_grid [B,C,*grid], got {tuple(masked_grid.shape)}")
    shared = tuple(mask.shape) == tuple(masked_grid.shape[1:])
    batched = tuple(mask.shape) == tuple(masked_grid.shape)
    if not shared and not batched:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} must match grid with or without batch: "
            f"{tuple(masked_grid.shape)} or {tuple(masked_grid.shape[1:])}"
        )
    out = torch.zeros_like(masked_grid)
    for batch_index in range(masked_grid.shape[0]):
        sample_mask = mask if shared else mask[batch_index]
        for channel_index in range(masked_grid.shape[1]):
            nearest = _nearest_flat_indices(sample_mask[channel_index])
            if nearest is None:
                continue
            values = masked_grid[batch_index, channel_index].reshape(-1)
            out[batch_index, channel_index] = values[nearest].reshape_as(
                masked_grid[batch_index, channel_index]
            )
    return out
