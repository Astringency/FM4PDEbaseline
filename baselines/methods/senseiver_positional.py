from __future__ import annotations

import math

import torch


def senseiver_fourier_features(
    coords: torch.Tensor,
    spatial_shape: tuple[int, ...],
    num_frequency_bands: int,
) -> torch.Tensor:
    """Encode normalized grid coordinates like the vendored Senseiver code.

    ``coords`` are the FM4PDE coordinates in ``[0, 1]``.  Upstream Senseiver
    first maps each coordinate to ``[-1, 1]`` and, for every spatial axis,
    applies linearly spaced frequencies from 1 to half that axis resolution.
    The output order is all sine bands followed by all cosine bands, matching
    ``offical/Senseiver/positional.py::PositionalEncoder``.
    """

    spatial_shape = tuple(int(size) for size in spatial_shape)
    bands = int(num_frequency_bands)
    if bands < 1:
        raise ValueError(f"num_frequency_bands must be positive, got {bands}")
    if coords.ndim < 2:
        raise ValueError(f"Senseiver coordinates must have a point and coordinate dimension, got {tuple(coords.shape)}")
    if coords.shape[-1] != len(spatial_shape):
        raise ValueError(
            f"Coordinate dimension {coords.shape[-1]} does not match spatial shape {spatial_shape}"
        )
    if not torch.is_floating_point(coords):
        coords = coords.float()

    positions = coords.mul(2.0).sub(1.0)
    frequency_grids = []
    for axis, size in enumerate(spatial_shape):
        frequencies = torch.linspace(
            1.0,
            float(size) / 2.0,
            bands,
            device=coords.device,
            dtype=coords.dtype,
        )
        frequency_grids.append(positions[..., axis : axis + 1] * frequencies)
    return torch.cat(
        [torch.sin(math.pi * grid) for grid in frequency_grids]
        + [torch.cos(math.pi * grid) for grid in frequency_grids],
        dim=-1,
    )
