from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from baselines.common.data_adapter import PDEBatch

from .base import BaselineModel, run_supervised_fit
from .official import OfficialImportError, get_recfno_unet_class, official_source_info, requested_implementation_mode
from .shared import ConvReconNet, grid_channels


class VoronoiCNNBaseline(BaselineModel):
    name = "voronoicnn"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        target_channels = int(data_spec["target_channels"])
        architecture_in_channels = target_channels + target_channels
        adapted_in_channels = architecture_in_channels + 2
        backend = str(self.config.get("official_backend", "auto")).lower()
        implementation_mode = requested_implementation_mode(self.config)
        min_res = min(tuple(data_spec.get("target_shape", (0, 0, 0, 0)))[-2:])
        fallback_reason = ""
        self.include_coords = True
        if implementation_mode == "official_architecture" or backend in {"voronoicnn", "official_architecture"}:
            self.include_coords = False
            self.net = _VoronoiCNNArchitectureNet(
                in_channels=architecture_in_channels,
                out_channels=target_channels,
                width=int(self.config.get("width", 48)),
            )
            self.set_backend(
                "voronoicnn_architecture",
                "voronoicnn",
                fallback_used=False,
                implementation_mode_effective="official_architecture",
                implementation_source="voronoicnn_official_architecture_reimplementation",
                official_import_success=False,
                adapter_status="official_architecture_reimplementation",
                **official_source_info("voronoi_cnn"),
            )
            return self
        if implementation_mode != "adapted" and backend == "recfno" and min_res >= 16:
            try:
                unet = get_recfno_unet_class()
                self.net = unet(in_channels=adapted_in_channels, out_channels=target_channels)
                self.set_backend(
                    "recfno_unet",
                    "recfno_unet",
                    fallback_used=True,
                    implementation_mode_effective="adapted",
                    implementation_source="recfno_unet_as_voronoi_cnn_adaptation",
                    official_import_success=True,
                    adapter_status="fallback_recfno_unet_adaptation",
                    **official_source_info("recfno"),
                )
                return self
            except OfficialImportError as exc:
                fallback_reason = f"recfno_unet unavailable: {exc}"
                warnings.warn(f"RecFNO UNet adaptation unavailable, using local CNN fallback: {exc}", RuntimeWarning, stacklevel=2)
        elif implementation_mode != "adapted" and backend == "recfno" and min_res < 16:
            fallback_reason = f"target resolution {min_res} is too small for official RecFNO UNet"
        elif implementation_mode != "adapted" and backend in {"auto", "official"}:
            fallback_reason = "official Voronoi-CNN Keras scripts are not importable; set implementation_mode=official_architecture for the disclosed PyTorch architecture reimplementation"
        self.net = ConvReconNet(adapted_in_channels, target_channels, width=int(self.config.get("width", 48)))
        requested_local = backend in {"local", "none"} or implementation_mode == "adapted"
        self.set_backend(
            "local",
            "local" if requested_local else ("official" if backend == "official" else backend),
            fallback_used=not requested_local,
            warning=fallback_reason,
            implementation_mode_effective="adapted",
            implementation_source="local_voronoi_cnn",
            official_import_success=False,
            adapter_status="local_adapted" if requested_local else "fallback_adapted",
        )
        return self

    def fit(self, train_loader, val_loader=None):
        return run_supervised_fit(self, train_loader, val_loader)

    def predict(self, batch: PDEBatch):
        vor = batch.metadata.get("voronoi_grid", batch.input_fields)
        mask = batch.mask
        if mask is None:
            mask_grid = torch.ones_like(vor)
        else:
            mask_grid = mask.unsqueeze(0).repeat(vor.shape[0], 1, 1, 1).to(vor.device, vor.dtype)
        parts = [vor, mask_grid]
        if self.include_coords:
            parts.append(grid_channels(vor))
        x = torch.cat(parts, dim=1)
        return self.net(x)


class _VoronoiCNNArchitectureNet(nn.Module):
    """PyTorch reimplementation of the published Voronoi-CNN Conv2D stack."""

    def __init__(self, in_channels: int, out_channels: int, width: int = 48) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c = in_channels
        for _ in range(7):
            layers.extend([nn.Conv2d(c, width, kernel_size=7, padding=3), nn.ReLU(inplace=True)])
            c = width
        layers.append(nn.Conv2d(width, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
