from __future__ import annotations

import torch
import torch.nn as nn

from baselines.common.data_adapter import PDEBatch


class OfficialAlignedVIVIDInverseObservation(nn.Module):
    """VIVID/invobs-style sparse-observation inverter.

    VIVID trains a CNN on Voronoi-interpolated sparse observations before
    variational refinement. The invobs implementation uses time-space
    convolutional inverse observation models. This module combines those
    architecture semantics for FM4PDE tensors.
    """

    def __init__(self, channels: int, tensor_ndim: int, width: int = 48, depth: int = 6) -> None:
        super().__init__()
        self.channels = int(channels)
        self.tensor_ndim = int(tensor_ndim)
        in_channels = self.channels * 2
        if self.tensor_ndim == 5:
            conv = nn.Conv3d
            norm = nn.BatchNorm3d
        else:
            conv = nn.Conv2d
            norm = nn.BatchNorm2d
        layers: list[nn.Module] = []
        current = in_channels
        for _ in range(max(int(depth), 1)):
            layers.extend([conv(current, width, kernel_size=3, padding=1), norm(width), nn.SiLU()])
            current = width
        layers.append(conv(current, self.channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, batch_or_tensor: PDEBatch | torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(batch_or_tensor, PDEBatch):
            x, mask = vivid_inverse_observation_input(batch_or_tensor)
        else:
            x = batch_or_tensor
        if mask is None:
            mask_tensor = torch.zeros_like(x)
        else:
            mask_tensor = mask.to(x.device, x.dtype)
            while mask_tensor.ndim < x.ndim:
                mask_tensor = mask_tensor.unsqueeze(0)
            if mask_tensor.shape[0] == 1 and x.shape[0] != 1:
                mask_tensor = mask_tensor.expand(x.shape[0], *mask_tensor.shape[1:])
            mask_tensor = mask_tensor[:, : x.shape[1]]
        return self.net(torch.cat([x, mask_tensor], dim=1))


def vivid_inverse_observation_input(batch: PDEBatch) -> tuple[torch.Tensor, torch.Tensor | None]:
    x = batch.metadata.get("voronoi_grid", batch.input_fields)
    if not isinstance(x, torch.Tensor):
        x = batch.input_fields
    mask = batch.mask
    return x, mask
