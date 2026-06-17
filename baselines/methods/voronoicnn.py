from __future__ import annotations

import warnings

import torch

from baselines.common.data_adapter import PDEBatch

from .base import BaselineModel, run_supervised_fit
from .official import OfficialImportError, get_recfno_unet_class
from .shared import ConvReconNet, grid_channels


class VoronoiCNNBaseline(BaselineModel):
    name = "voronoicnn"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        target_channels = int(data_spec["target_channels"])
        in_channels = target_channels + target_channels + 2
        backend = str(self.config.get("official_backend", "auto")).lower()
        min_res = min(tuple(data_spec.get("target_shape", (0, 0, 0, 0)))[-2:])
        if backend in {"auto", "recfno", "official"} and min_res >= 16:
            try:
                unet = get_recfno_unet_class()
                self.net = unet(in_channels=in_channels, out_channels=target_channels)
                self.official_backend = "recfno_unet"
                return self
            except OfficialImportError as exc:
                if backend == "recfno":
                    warnings.warn(f"RecFNO official UNet unavailable, using local CNN fallback: {exc}", RuntimeWarning, stacklevel=2)
        self.net = ConvReconNet(in_channels, target_channels, width=int(self.config.get("width", 48)))
        self.official_backend = "local"
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
