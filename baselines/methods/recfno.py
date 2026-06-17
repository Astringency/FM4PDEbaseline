from __future__ import annotations

import torch

from baselines.common.data_adapter import PDEBatch

from .base import BaselineModel, run_supervised_fit
from .shared import FNO2dNet, grid_channels


class RecFNOBaseline(BaselineModel):
    name = "recfno"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        target_channels = int(data_spec["target_channels"])
        # RecFNO sparse embeddings: mask embedding and Voronoi embedding are
        # implemented here; MLP embedding is reserved for future extension.
        in_channels = target_channels + target_channels + 2
        self.embedding = str(self.config.get("embedding", "mask"))
        self.net = FNO2dNet(
            in_channels=in_channels,
            out_channels=target_channels,
            width=int(self.config.get("width", 24)),
            modes1=int(self.config.get("modes1", 12)),
            modes2=int(self.config.get("modes2", 12)),
            add_coords=False,
        )
        return self

    def fit(self, train_loader, val_loader=None):
        return run_supervised_fit(self, train_loader, val_loader)

    def predict(self, batch: PDEBatch):
        if self.embedding == "voronoi":
            base = batch.metadata.get("voronoi_grid", batch.input_fields)
        else:
            base = batch.metadata.get("masked_grid", batch.input_fields)
        mask = batch.mask
        if mask is None:
            mask_grid = torch.ones_like(base)
        else:
            mask_grid = mask.unsqueeze(0).repeat(base.shape[0], 1, 1, 1).to(base.device, base.dtype)
        x = torch.cat([base, mask_grid, grid_channels(base)], dim=1)
        return self.net(x)

