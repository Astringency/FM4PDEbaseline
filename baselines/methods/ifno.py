from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.data_adapter import PDEBatch

from .base import BaselineModel, _to_device_batch
from .shared import FNO2dNet, SpectralConv2d, grid_channels


class _CouplingBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int) -> None:
        super().__init__()
        half = width // 2
        self.s12 = SpectralConv2d(half, half, modes1, modes2)
        self.w12 = nn.Conv2d(half, half, 1)
        self.s21 = SpectralConv2d(half, half, modes1, modes2)
        self.w21 = nn.Conv2d(half, half, 1)

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=1)
        if not inverse:
            scale2 = F.softplus(self.s12(x2) + self.w12(x2)) + 1e-3
            y1 = x1 * scale2
            scale1 = F.softplus(self.s21(y1) + self.w21(y1)) + 1e-3
            y2 = x2 * scale1
            return torch.cat([y1, y2], dim=1)
        scale1 = F.softplus(self.s21(x1) + self.w21(x1)) + 1e-3
        y2 = x2 / scale1
        scale2 = F.softplus(self.s12(y2) + self.w12(y2)) + 1e-3
        y1 = x1 / scale2
        return torch.cat([y1, y2], dim=1)


class IFNOBaseline(BaselineModel):
    name = "ifno"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        width = int(self.config.get("width", 32))
        if width % 2:
            width += 1
        modes1 = int(self.config.get("modes1", 12))
        modes2 = int(self.config.get("modes2", 12))
        layers = int(self.config.get("layers", 3))
        self.input_channels = int(data_spec["input_channels"])
        self.target_channels = int(data_spec["target_channels"])
        self.lift_x = nn.Conv2d(self.input_channels + 2, width, 1)
        self.lift_y = nn.Conv2d(self.target_channels + 2, width, 1)
        self.blocks = nn.ModuleList([_CouplingBlock(width, modes1, modes2) for _ in range(layers)])
        self.proj_y = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, self.target_channels, 1))
        self.proj_x = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, self.input_channels, 1))
        # Sparse inverse fallback uses a small FNO because sparse observations
        # are not exactly invertible in the iFNO sense.
        self.sparse_inverse = FNO2dNet(self.target_channels, self.input_channels, width=width, modes1=modes1, modes2=modes2)
        self.set_backend("local", "local", fallback_used=False)
        return self

    def fit(self, train_loader, val_loader=None):
        device = torch.device(self.config.get("device", "cpu"))
        epochs = int(self.config.get("epochs", 1))
        lr = float(self.config.get("lr", 1e-3))
        max_steps = self.config.get("max_steps")
        cycle_weight = float(self.config.get("cycle_weight", 0.1))
        self.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        history = {"train_loss": []}
        for _ in range(epochs):
            total = 0.0
            count = 0
            self.train()
            for step, batch in enumerate(train_loader):
                if max_steps is not None and step >= int(max_steps):
                    break
                batch = _to_device_batch(batch, device)
                x, y = _physical_pair(batch)
                opt.zero_grad(set_to_none=True)
                y_pred = self._forward_map(x)
                x_pred = self._inverse_map(y)
                cycle_x = self._inverse_map(y_pred)
                cycle_y = self._forward_map(x_pred)
                loss = F.mse_loss(y_pred, y) + F.mse_loss(x_pred, x)
                loss = loss + cycle_weight * (F.mse_loss(cycle_x, x) + F.mse_loss(cycle_y, y))
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu())
                count += 1
            history["train_loss"].append(total / max(count, 1))
        return history

    def predict(self, batch: PDEBatch):
        if batch.task in {"inverse", "sparse_inverse"}:
            x = batch.input_fields
            if batch.task == "sparse_inverse":
                return self.sparse_inverse(x)
            return self._inverse_map(x)
        x = batch.input_fields
        return self._forward_map(x)

    def _forward_map(self, x: torch.Tensor) -> torch.Tensor:
        z = self.lift_x(torch.cat([x, grid_channels(x)], dim=1))
        for block in self.blocks:
            z = block(z, inverse=False)
        return self.proj_y(z)

    def _inverse_map(self, y: torch.Tensor) -> torch.Tensor:
        z = self.lift_y(torch.cat([y, grid_channels(y)], dim=1))
        for block in reversed(self.blocks):
            z = block(z, inverse=True)
        return self.proj_x(z)


def _physical_pair(batch: PDEBatch) -> tuple[torch.Tensor, torch.Tensor]:
    if batch.task in {"inverse", "sparse_inverse"}:
        return batch.target_fields, batch.input_fields
    return batch.input_fields, batch.target_fields
