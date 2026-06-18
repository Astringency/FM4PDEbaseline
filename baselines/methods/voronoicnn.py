from __future__ import annotations

import warnings

import torch

from baselines.common.data_adapter import PDEBatch

from .base import BaselineModel, run_supervised_fit
from .official import OfficialImportError, get_recfno_unet_class, official_source_info, requested_implementation_mode
from .shared import ConvReconNet, grid_channels


class VoronoiCNNBaseline(BaselineModel):
    name = "voronoicnn"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        target_channels = int(data_spec["target_channels"])
        in_channels = target_channels + target_channels + 2
        backend = str(self.config.get("official_backend", "auto")).lower()
        implementation_mode = requested_implementation_mode(self.config)
        min_res = min(tuple(data_spec.get("target_shape", (0, 0, 0, 0)))[-2:])
        fallback_reason = ""
        if implementation_mode != "adapted" and backend in {"auto", "recfno", "official"} and min_res >= 16:
            try:
                unet = get_recfno_unet_class()
                self.net = unet(in_channels=in_channels, out_channels=target_channels)
                self.set_backend(
                    "recfno_unet",
                    "recfno_unet",
                    fallback_used=False,
                    implementation_mode_effective="official_architecture",
                    implementation_source="recfno_unet",
                    official_import_success=True,
                    adapter_status="official_architecture_reimplementation",
                    **official_source_info("recfno"),
                )
                return self
            except OfficialImportError as exc:
                fallback_reason = f"recfno_unet unavailable: {exc}"
                if backend in {"recfno", "official"}:
                    warnings.warn(f"RecFNO official UNet unavailable, using local CNN fallback: {exc}", RuntimeWarning, stacklevel=2)
        elif implementation_mode != "adapted" and backend in {"auto", "recfno", "official"} and min_res < 16:
            fallback_reason = f"target resolution {min_res} is too small for official RecFNO UNet"
        self.net = ConvReconNet(in_channels, target_channels, width=int(self.config.get("width", 48)))
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
        x = torch.cat([vor, mask_grid, grid_channels(vor)], dim=1)
        return self.net(x)
