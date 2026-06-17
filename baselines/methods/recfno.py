from __future__ import annotations

import torch

from baselines.common.data_adapter import PDEBatch

from .base import BaselineModel, run_supervised_fit
from .official import OfficialImportError
from .shared import FNO2dNet, OfficialRecFNOVoronoiFNO2dNet, grid_channels


class RecFNOBaseline(BaselineModel):
    name = "recfno"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        target_channels = int(data_spec["target_channels"])
        # RecFNO sparse embeddings: mask embedding and Voronoi embedding are
        # implemented here; MLP embedding is reserved for future extension.
        in_channels = target_channels + target_channels + 2
        self.embedding = str(self.config.get("embedding", "mask"))
        modes1, modes2 = _clamped_modes(self.config, data_spec)
        backend = str(self.config.get("official_backend", "auto")).lower()
        width = int(self.config.get("width", 24))
        if backend not in {"local", "none"}:
            try:
                self.net = OfficialRecFNOVoronoiFNO2dNet(
                    in_channels=in_channels,
                    out_channels=target_channels,
                    width=width,
                    modes1=modes1,
                    modes2=modes2,
                    add_coords=False,
                )
                self.set_backend("recfno", "recfno", fallback_used=False)
                return self
            except OfficialImportError as exc:
                self.backend_warning = f"recfno unavailable: {exc}"
        self.net = FNO2dNet(
            in_channels=in_channels,
            out_channels=target_channels,
            width=width,
            modes1=modes1,
            modes2=modes2,
            add_coords=False,
        )
        requested_local = backend in {"local", "none"}
        self.set_backend(
            "local",
            "local" if requested_local else ("official" if backend == "official" else backend),
            fallback_used=not requested_local,
            warning="" if requested_local else self.backend_warning,
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


def _clamped_modes(config: dict, data_spec: dict) -> tuple[int, int]:
    shape = tuple(data_spec.get("target_shape", ()))
    h = int(shape[-2]) if len(shape) >= 2 else 32
    w = int(shape[-1]) if len(shape) >= 1 else h
    return max(1, min(int(config.get("modes1", 12)), h)), max(1, min(int(config.get("modes2", 12)), w // 2 + 1))
