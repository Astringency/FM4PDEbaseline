from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared import SpectralConv2d, grid_channels


class IFNOMLP2d(nn.Module):
    """Pointwise MLP used by the official iFNO scripts after FNO blocks."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IFNOCouplingLayer2d(nn.Module):
    """Multiplicative invertible FNO coupling layer matching the iFNO scripts."""

    def __init__(self, half_width: int, modes1: int, modes2: int, beta: float = 2.0, padding: int = 0) -> None:
        super().__init__()
        self.beta = float(beta)
        self.padding = int(max(padding, 0))
        self.conv12 = SpectralConv2d(half_width, half_width, modes1, modes2)
        self.mlp12 = IFNOMLP2d(half_width, half_width, half_width)
        self.w12 = nn.Conv2d(half_width, half_width, 1)
        self.conv21 = SpectralConv2d(half_width, half_width, modes1, modes2)
        self.mlp21 = IFNOMLP2d(half_width, half_width, half_width)
        self.w21 = nn.Conv2d(half_width, half_width, 1)

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        a, b = torch.chunk(x, 2, dim=1)
        if not inverse:
            s2 = self._scale_from_b(b)
            va = a * s2
            s1 = self._scale_from_a(va)
            vb = b * s1
            return torch.cat([va, vb], dim=1)
        s1 = self._scale_from_a(a)
        ub = b / s1
        s2 = self._scale_from_b(ub)
        ua = a / s2
        return torch.cat([ua, ub], dim=1)

    def _scale_from_b(self, x: torch.Tensor) -> torch.Tensor:
        return self._positive_scale(self.mlp12(self._spectral_with_padding(self.conv12, x)) + self.w12(x))

    def _scale_from_a(self, x: torch.Tensor) -> torch.Tensor:
        return self._positive_scale(self.mlp21(self._spectral_with_padding(self.conv21, x)) + self.w21(x))

    def _positive_scale(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(F.gelu(x), beta=self.beta).clamp_min(1e-4)

    def _spectral_with_padding(self, conv: SpectralConv2d, x: torch.Tensor) -> torch.Tensor:
        if self.padding <= 0:
            return conv(x)
        padded = F.pad(x, [0, self.padding, 0, self.padding])
        out = conv(padded)
        return out[..., : x.shape[-2], : x.shape[-1]]


class OfficialAlignedIFNO2d(nn.Module):
    """Import-safe iFNO architecture reimplementation.

    The vendored official implementation stores physical fields with two grid
    coordinate channels and trains a single invertible coupling backbone for
    forward and backward maps. This module keeps that same data flow while
    adapting the physical channel counts to FM4PDE tasks.
    """

    def __init__(
        self,
        input_channels: int,
        target_channels: int,
        width: int = 64,
        modes1: int = 16,
        modes2: int = 16,
        layers: int = 4,
        beta: float = 2.0,
        padding: int = 0,
    ) -> None:
        super().__init__()
        if width % 2:
            width += 1
        self.input_channels = int(input_channels)
        self.target_channels = int(target_channels)
        self.width = int(width)
        self.lift_x = nn.Conv2d(self.input_channels + 2, self.width, 1)
        self.lift_y = nn.Conv2d(self.target_channels + 2, self.width, 1)
        self.proj_y = IFNOMLP2d(self.width, self.target_channels + 2, self.width * 4)
        self.proj_x = IFNOMLP2d(self.width, self.input_channels + 2, self.width * 4)
        half = self.width // 2
        self.layers = nn.ModuleList(
            [IFNOCouplingLayer2d(half, modes1, modes2, beta=beta, padding=padding) for _ in range(max(int(layers), 1))]
        )

    def forward_map(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_aug = self._augment(x)
        z0 = self.lift_x(x_aug)
        recon = self.proj_x(z0)
        z = z0
        for layer in self.layers:
            z = layer(z, inverse=False)
        y_aug = self.proj_y(z)
        return y_aug[:, : self.target_channels], F.mse_loss(recon, x_aug)

    def inverse_map(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_aug = self._augment(y)
        z0 = self.lift_y(y_aug)
        recon = self.proj_y(z0)
        z = z0
        for layer in reversed(self.layers):
            z = layer(z, inverse=True)
        x_aug = self.proj_x(z)
        return x_aug[:, : self.input_channels], F.mse_loss(recon, y_aug)

    def forward_then_inverse(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.forward_map(x)
        x_rec, _ = self.inverse_map(y)
        return x_rec

    def inverse_then_forward(self, y: torch.Tensor) -> torch.Tensor:
        x, _ = self.inverse_map(y)
        y_rec, _ = self.forward_map(x)
        return y_rec

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, grid_channels(x)], dim=1)
