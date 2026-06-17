from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .official import OfficialImportError, get_recfno_fno_classes


def grid_channels(x: torch.Tensor) -> torch.Tensor:
    b, _, h, w = x.shape
    yy = torch.linspace(0.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).repeat(b, 1, 1, w)
    xx = torch.linspace(0.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).repeat(b, 1, h, 1)
    return torch.cat([yy, xx], dim=1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(b, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(h, w))


class FNO2dNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int = 32, modes1: int = 12, modes2: int = 12, layers: int = 4, add_coords: bool = True) -> None:
        super().__init__()
        self.add_coords = add_coords
        lifted = in_channels + (2 if add_coords else 0)
        self.fc0 = nn.Conv2d(lifted, width, 1)
        self.convs = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.fc1 = nn.Conv2d(width, max(width, 64), 1)
        self.fc2 = nn.Conv2d(max(width, 64), out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"FNO2dNet expects [B,C,H,W], got {tuple(x.shape)}")
        if self.add_coords:
            x = torch.cat([x, grid_channels(x)], dim=1)
        x = self.fc0(x)
        for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
            x = conv(x) + w(x)
            if i != len(self.convs) - 1:
                x = F.gelu(x)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


class OfficialRecFNOVoronoiFNO2dNet(nn.Module):
    """Adapter for the vendored RecFNO VoronoiFNO2d NCHW implementation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        add_coords: bool = True,
    ) -> None:
        super().__init__()
        _fno2d, voronoi_fno2d, _spectral = get_recfno_fno_classes()
        self.add_coords = add_coords
        self.net = voronoi_fno2d(
            modes1=modes1,
            modes2=modes2,
            width=width,
            in_channels=in_channels + (2 if add_coords else 0),
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.add_coords:
            x = torch.cat([x, grid_channels(x)], dim=1)
        return self.net(x)


def make_official_recfno_fno_or_local(
    in_channels: int,
    out_channels: int,
    width: int,
    modes1: int,
    modes2: int,
    layers: int = 4,
    add_coords: bool = True,
) -> nn.Module:
    try:
        return OfficialRecFNOVoronoiFNO2dNet(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes1=modes1,
            modes2=modes2,
            add_coords=add_coords,
        )
    except OfficialImportError:
        return FNO2dNet(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes1=modes1,
            modes2=modes2,
            layers=layers,
            add_coords=add_coords,
        )


class ConvReconNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int = 48, depth: int = 7) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, width, 7, padding=3), nn.GELU()]
        for _ in range(max(depth - 2, 1)):
            layers += [nn.Conv2d(width, width, 5, padding=2), nn.GELU()]
        layers.append(nn.Conv2d(width, out_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(max(depth - 1, 1)):
            layers += [nn.Linear(last, hidden), nn.GELU()]
            last = hidden
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralField(nn.Module):
    def __init__(self, coord_dim: int, out_channels: int, hidden: int = 64, depth: int = 4) -> None:
        super().__init__()
        self.mlp = MLP(coord_dim, out_channels, hidden=hidden, depth=depth)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(coords)


def flatten_grid(fields: torch.Tensor) -> torch.Tensor:
    return fields.reshape(fields.shape[0], -1)


def unflatten_grid(values: torch.Tensor, out_shape: tuple[int, ...]) -> torch.Tensor:
    return values.reshape(values.shape[0], *out_shape)


def query_coords_for_shape(shape: tuple[int, int], batch_size: int, device, dtype) -> torch.Tensor:
    h, w = shape
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, h, device=device, dtype=dtype),
        torch.linspace(0, 1, w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([yy, xx], dim=-1).reshape(1, h * w, 2).repeat(batch_size, 1, 1)
