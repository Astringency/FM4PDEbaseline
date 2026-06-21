from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.data_adapter import PDEBatch
from baselines.common.normalization import estimate_normalization_stats, normalize_batch_input_target

from .base import BaselineModel, _to_device_batch, restore_state_dict, snapshot_state_dict
from .ifno_official_aligned import OfficialAlignedIFNO2d
from .official import (
    OfficialImportError,
    get_ifno_official_aligned_status,
    get_ifno_official_status,
    official_source_info,
    requested_implementation_mode,
)
from .shared import SpectralConv2d, grid_channels


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
        beta = float(self.config.get("beta", 2.0))
        padding = int(self.config.get("padding", 0))
        backend = str(self.config.get("official_backend", "auto")).lower()
        implementation_mode = requested_implementation_mode(self.config)
        self.input_channels = int(data_spec["input_channels"])
        self.target_channels = int(data_spec["target_channels"])
        requested_local = backend in {"local", "none"} or implementation_mode == "adapted"
        if implementation_mode == "official" and not requested_local:
            get_ifno_official_status()
            raise OfficialImportError("direct official iFNO model adapter is not implemented after import status validation")
        self.official_aligned = (
            not requested_local
            and implementation_mode in {"official_or_skip", "official_aligned", "official_architecture", "auto"}
            and backend in {"auto", "ifno", "official"}
        )
        if self.official_aligned:
            direct_import_success = False
            direct_import_warning = ""
            if implementation_mode in {"official_or_skip", "auto"}:
                try:
                    get_ifno_official_status()
                    direct_import_success = True
                except OfficialImportError as exc:
                    direct_import_warning = f"direct official iFNO import unavailable; using official-aligned reimplementation: {exc}"
            get_ifno_official_aligned_status()
            self.operator = OfficialAlignedIFNO2d(
                self.input_channels,
                self.target_channels,
                width=width,
                modes1=modes1,
                modes2=modes2,
                layers=layers,
                beta=beta,
                padding=padding,
            )
            effective = "official_architecture" if implementation_mode == "official_architecture" else "official_aligned"
            self.set_backend(
                "ifno_official_aligned",
                "ifno",
                fallback_used=False,
                warning=direct_import_warning,
                implementation_mode_effective=effective,
                implementation_source="ifno_official_aligned_reimplementation",
                official_import_success=False,
                official_reimplementation_success=True,
                official_alignment_level="architecture" if effective == "official_architecture" else "objective",
                official_alignment_notes=(
                    "Reimplements vendored iFNO p1/p2 lift, q1/q2 pointwise projections, "
                    "multiplicative FNO coupling blocks, grid-coordinate augmentation, and bidirectional/cycle objectives."
                ),
                adapter_status="official_architecture_ifno_reimplementation"
                if effective == "official_architecture"
                else "official_aligned_ifno_reimplementation",
                **official_source_info("ifno"),
            )
            return self
        self.lift_x = nn.Conv2d(self.input_channels + 2, width, 1)
        self.lift_y = nn.Conv2d(self.target_channels + 2, width, 1)
        self.blocks = nn.ModuleList([_CouplingBlock(width, modes1, modes2) for _ in range(layers)])
        self.proj_y = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, self.target_channels, 1))
        self.proj_x = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, self.input_channels, 1))
        self.set_backend(
            "local_ifno_simplified",
            "local",
            fallback_used=False,
            warning="explicit local/debug iFNO path; not paper main-table eligible",
            implementation_mode_effective="adapted",
            implementation_source="local_simplified_ifno",
            official_import_success=False,
            official_reimplementation_success=False,
            official_alignment_level="local",
            official_alignment_notes="Simplified local/debug iFNO-style coupling, not an official architecture claim.",
            adapter_status="local_debug_ifno",
            **official_source_info("ifno"),
        )
        return self

    def fit(self, train_loader, val_loader=None):
        device = torch.device(self.config.get("device", "cpu"))
        epochs = int(self.config.get("epochs", 1))
        lr = float(self.config.get("lr", 1e-3))
        max_steps = self.config.get("max_steps")
        max_val_steps = self.config.get("max_val_steps")
        cycle_weight = float(self.config.get("cycle_weight", 0.1))
        normalize = bool(self.config.get("normalize", False))
        if normalize and self.normalization_stats is None:
            self.normalization_stats = estimate_normalization_stats(
                train_loader,
                max_batches=self.config.get("normalization_max_batches"),
                eps=float(self.config.get("normalization_eps", 1e-6)),
            )
        self.uses_normalization = bool(normalize and self.normalization_stats is not None)
        self.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        history = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": None,
            "best_val_loss": None,
            "normalize": self.uses_normalization,
            "normalization_stats": self.normalization_stats.json_summary() if self.normalization_stats is not None else None,
        }
        best_val = None
        best_state = None
        for epoch in range(epochs):
            total = 0.0
            count = 0
            self.train()
            for step, batch in enumerate(train_loader):
                if max_steps is not None and step >= int(max_steps):
                    break
                batch = _to_device_batch(batch, device)
                if self.uses_normalization and self.normalization_stats is not None:
                    batch = normalize_batch_input_target(batch, self.normalization_stats)
                opt.zero_grad(set_to_none=True)
                loss = _ifno_training_loss(self, batch, cycle_weight)
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu())
                count += 1
            history["train_loss"].append(total / max(count, 1))
            if val_loader is not None:
                self.eval()
                val_total = 0.0
                val_count = 0
                with torch.no_grad():
                    for step, batch in enumerate(val_loader):
                        if max_val_steps is not None and step >= int(max_val_steps):
                            break
                        batch = _to_device_batch(batch, device)
                        if self.uses_normalization and self.normalization_stats is not None:
                            batch = normalize_batch_input_target(batch, self.normalization_stats)
                        val_total += float(_ifno_training_loss(self, batch, cycle_weight).detach().cpu())
                        val_count += 1
                val_loss = val_total / max(val_count, 1)
                history["val_loss"].append(val_loss)
                if best_val is None or val_loss < best_val:
                    best_val = val_loss
                    history["best_epoch"] = epoch
                    history["best_val_loss"] = val_loss
                    best_state = snapshot_state_dict(self)
        if best_state is not None:
            restore_state_dict(self, best_state)
        return history

    def predict(self, batch: PDEBatch):
        if batch.task.startswith("sparse"):
            raise NotImplementedError("iFNO sparse reconstruction/inverse is unsupported without an official sparse inference adapter")
        if batch.task == "inverse":
            return self._inverse_map(batch.input_fields)
        x = batch.input_fields
        return self._forward_map(x)

    def _forward_map(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "official_aligned", False):
            pred, _ = self.operator.forward_map(x)
            return pred
        z = self.lift_x(torch.cat([x, grid_channels(x)], dim=1))
        for block in self.blocks:
            z = block(z, inverse=False)
        return self.proj_y(z)

    def _inverse_map(self, y: torch.Tensor) -> torch.Tensor:
        if getattr(self, "official_aligned", False):
            pred, _ = self.operator.inverse_map(y)
            return pred
        z = self.lift_y(torch.cat([y, grid_channels(y)], dim=1))
        for block in reversed(self.blocks):
            z = block(z, inverse=True)
        return self.proj_x(z)

    def _forward_map_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "official_aligned", False):
            return self.operator.forward_map(x)
        return self._forward_map(x), torch.tensor(0.0, device=x.device, dtype=x.dtype)

    def _inverse_map_with_aux(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "official_aligned", False):
            return self.operator.inverse_map(y)
        return self._inverse_map(y), torch.tensor(0.0, device=y.device, dtype=y.dtype)


def _physical_pair(batch: PDEBatch) -> tuple[torch.Tensor, torch.Tensor]:
    if batch.task in {"inverse", "sparse_inverse"}:
        return batch.target_fields, batch.input_fields
    return batch.input_fields, batch.target_fields


def _ifno_training_loss(model: IFNOBaseline, batch: PDEBatch, cycle_weight: float) -> torch.Tensor:
    x, y = _physical_pair(batch)
    y_pred, recon_x = model._forward_map_with_aux(x)
    x_pred, recon_y = model._inverse_map_with_aux(y)
    cycle_x = model._inverse_map(y_pred)
    cycle_y = model._forward_map(x_pred)
    loss = F.mse_loss(y_pred, y) + F.mse_loss(x_pred, x)
    loss = loss + cycle_weight * (F.mse_loss(cycle_x, x) + F.mse_loss(cycle_y, y))
    return loss + float(model.config.get("reconstruction_weight", 0.1)) * (recon_x + recon_y)
