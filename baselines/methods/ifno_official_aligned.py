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


class OfficialVanillaVAE(nn.Module):
    """Official iFNO VAE made resolution-adaptive at the data seam.

    The official network is defined for 64x64 fields. FM4PDE fields are resized
    to that resolution before the unchanged five-stage encoder/decoder and
    resized back afterwards.
    """

    def __init__(self, latent_dim: int, hidden_dims: list[int] | None = None, resolution: int = 64) -> None:
        super().__init__()
        dims = list(hidden_dims or [32, 64, 128, 256, 512])
        if len(dims) != 5:
            raise ValueError("Official iFNO VAE requires five hidden dimensions")
        self.resolution = int(resolution)
        if self.resolution != 64:
            raise ValueError("Official iFNO VAE topology requires vae_resolution=64")
        modules: list[nn.Module] = []
        in_channels = 1
        for channels in dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, channels, 3, stride=2, padding=1), nn.GELU()))
            in_channels = channels
        self.encoder = nn.Sequential(*modules)
        bottleneck = dims[-1] * 4
        self.fc_mu = nn.Linear(bottleneck, int(latent_dim))
        self.fc_var = nn.Linear(bottleneck, int(latent_dim))
        self.decoder_input = nn.Linear(int(latent_dim), bottleneck)
        reversed_dims = list(reversed(dims))
        decoder: list[nn.Module] = []
        for index in range(len(reversed_dims) - 1):
            decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(reversed_dims[index], reversed_dims[index + 1], 3, stride=2, padding=1, output_padding=1),
                    nn.GELU(),
                )
            )
        self.decoder = nn.Sequential(*decoder)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(reversed_dims[-1], reversed_dims[-1], 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.Conv2d(reversed_dims[-1], 1, 3, padding=1),
        )
        self.bottleneck_channels = dims[-1]

    def encode(self, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(self._to_official_resolution(field))
        flat = torch.flatten(encoded, start_dim=1)
        return self.fc_mu(flat), self.fc_var(flat)

    def decode(self, latent: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        decoded = self.decoder_input(latent).reshape(-1, self.bottleneck_channels, 2, 2)
        decoded = self.final_layer(self.decoder(decoded))
        return F.interpolate(decoded, size=output_size, mode="bilinear", align_corners=False)

    def forward(self, field: torch.Tensor, *, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_size = tuple(int(value) for value in field.shape[-2:])
        mu, log_var = self.encode(field)
        if sample:
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        else:
            latent = mu
        return self.decode(latent, output_size), mu, log_var

    def _to_official_resolution(self, field: torch.Tensor) -> torch.Tensor:
        if tuple(field.shape[-2:]) == (self.resolution, self.resolution):
            return field
        return F.interpolate(field, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)


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
        rank: int = 24,
        vae_hidden_dims: list[int] | None = None,
        vae_resolution: int = 64,
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
        self.vae_net = OfficialVanillaVAE(rank, hidden_dims=vae_hidden_dims, resolution=vae_resolution)

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

    def vae_reconstruct(self, x: torch.Tensor, *, sample: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first, mu, log_var = self.vae_net(x[:, :1], sample=sample)
        if x.shape[1] == 1:
            return first, mu, log_var
        return torch.cat([first, x[:, 1:]], dim=1), mu, log_var

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, grid_channels(x)], dim=1)
