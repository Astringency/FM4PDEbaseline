from __future__ import annotations

import torch


def _nearest_fill_single(channel_values: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    shape = tuple(channel_values.shape)
    flat_mask = channel_mask.reshape(-1).bool()
    if flat_mask.sum() == 0:
        return torch.zeros_like(channel_values)
    if flat_mask.sum() == flat_mask.numel():
        return channel_values
    coords = torch.stack(
        torch.meshgrid(
            *[torch.arange(s, device=channel_values.device, dtype=torch.float32) for s in shape],
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, len(shape))
    sensor_coords = coords[flat_mask]
    dist = torch.cdist(coords, sensor_coords)
    nearest = dist.argmin(dim=1)
    sensor_values = channel_values.reshape(-1)[flat_mask]
    return sensor_values[nearest].reshape(shape)


def voronoi_fill(masked_grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Voronoi-like nearest-sensor filling for channel-first tensors.

    Supports ``[B,C,H,W]`` and ``[B,C,T,H,W]``. The method is intentionally
    simple and deterministic for reproducible baseline inputs. It uses nearest
    observed grid point in Euclidean grid-index space.
    """
    if masked_grid.ndim < 4:
        raise ValueError(f"Expected masked_grid [B,C,*grid], got {tuple(masked_grid.shape)}")
    if tuple(mask.shape) != tuple(masked_grid.shape[1:]):
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match grid {tuple(masked_grid.shape[1:])}")
    out = torch.empty_like(masked_grid)
    for b in range(masked_grid.shape[0]):
        for c in range(masked_grid.shape[1]):
            out[b, c] = _nearest_fill_single(masked_grid[b, c], mask[c])
    return out

