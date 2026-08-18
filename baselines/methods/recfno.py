from __future__ import annotations

import torch

from baselines.common.data_adapter import PDEBatch

from .base import BaselineModel, run_supervised_fit
from .official import OfficialImportError, official_source_info, requested_implementation_mode, wrap_official_adapter_error
from .shared import FNO2dNet, OfficialRecFNOVoronoiFNO2dNet, grid_channels


class RecFNOBaseline(BaselineModel):
    name = "recfno"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        input_channels = int(data_spec["input_channels"])
        target_channels = int(data_spec["target_channels"])
        # The vendored RecFNO VoronoiFNO2d is trained on the representation
        # [Voronoi-filled observed field, observation mask, coordinates].
        # Keep that representation for the local fallback as well so changing
        # the backend does not silently change the task input.
        in_channels = input_channels + input_channels + 2
        required_representation = "voronoi_mask_coords"
        configured_representation = self.config.get("input_representation")
        legacy_embedding = self.config.get("embedding")
        if configured_representation is None and legacy_embedding is not None:
            configured_representation = "voronoi_mask_coords" if str(legacy_embedding).lower() == "voronoi" else str(legacy_embedding)
        configured_representation = str(configured_representation or required_representation).lower()
        if configured_representation != required_representation:
            raise ValueError(
                "RecFNO input_representation must be 'voronoi_mask_coords' so the vendored "
                f"VoronoiFNO2d receives its declared input contract, got {configured_representation!r}"
            )
        self.input_representation = required_representation
        self.config["input_representation"] = required_representation
        modes1, modes2 = _clamped_modes(self.config, data_spec)
        backend = str(self.config.get("official_backend", "auto")).lower()
        implementation_mode = requested_implementation_mode(self.config)
        width = int(self.config.get("width", 24))
        if implementation_mode != "adapted" and backend not in {"local", "none"}:
            try:
                self.net = OfficialRecFNOVoronoiFNO2dNet(
                    in_channels=in_channels,
                    out_channels=target_channels,
                    width=width,
                    modes1=modes1,
                    modes2=modes2,
                    add_coords=False,
                )
                self.set_backend(
                    "recfno",
                    "recfno",
                    fallback_used=False,
                    implementation_mode_effective="adapted",
                    implementation_source="vendored_recfno_voronoifno2d_component_with_unified_training",
                    official_import_success=True,
                    official_reimplementation_success=False,
                    official_alignment_level="component",
                    official_alignment_notes=(
                        "Directly imports the vendored RecFNO VoronoiFNO2d component and uses its "
                        "Voronoi+mask+coordinates representation (with FM4PDE normalized grid coordinates); "
                        "optimization and data loading use the FM4PDE unified training protocol, not the "
                        "upstream end-to-end script."
                    ),
                    adapter_status="official_component_unified_training_adapter",
                    **official_source_info("recfno"),
                )
                return self
            except Exception as exc:
                exc = wrap_official_adapter_error("RecFNO", exc)
                self.backend_warning = f"recfno unavailable: {exc}"
        if implementation_mode == "official" and self.backend_warning:
            raise OfficialImportError(self.backend_warning)
        self.net = FNO2dNet(
            in_channels=in_channels,
            out_channels=target_channels,
            width=width,
            modes1=modes1,
            modes2=modes2,
            add_coords=False,
        )
        requested_local = backend in {"local", "none"} or implementation_mode == "adapted"
        self.set_backend(
            "local",
            "local" if requested_local else ("official" if backend == "official" else backend),
            fallback_used=not requested_local,
            warning="" if requested_local else self.backend_warning,
            implementation_mode_effective="adapted",
            implementation_source="local_recfno_architecture_adaptation",
            official_import_success=False,
            official_reimplementation_success=False,
            official_alignment_level="adapted",
            official_alignment_notes=(
                "Local FNO-style fallback using the RecFNO Voronoi+mask+coordinates input contract; "
                "it does not import the official network component."
            ),
            adapter_status="local_adapted" if requested_local else "fallback_adapted",
        )
        return self

    def fit(self, train_loader, val_loader=None):
        return run_supervised_fit(self, train_loader, val_loader)

    def predict(self, batch: PDEBatch):
        base = batch.metadata.get("voronoi_grid")
        if not isinstance(base, torch.Tensor):
            raise ValueError("RecFNO requires batch.metadata['voronoi_grid']; zero-masked input fallback is not allowed")
        expected_channels = int(self.data_spec["input_channels"])
        if base.shape[1] != expected_channels:
            raise ValueError(
                f"RecFNO Voronoi input has {base.shape[1]} channels, expected {expected_channels} observed channels"
            )
        mask = batch.mask
        if mask is None:
            raise ValueError("RecFNO requires an explicit observation mask")
        if tuple(mask.shape) == tuple(base.shape):
            # Per-sample masks already carry the batch dimension.
            mask_grid = mask.to(base.device, base.dtype)
        elif tuple(mask.shape) == tuple(base.shape[1:]):
            mask_grid = mask.unsqueeze(0).expand(base.shape[0], *mask.shape).to(base.device, base.dtype)
        else:
            raise ValueError(
                f"RecFNO mask shape {tuple(mask.shape)} must match input with or without batch "
                f"({tuple(base.shape)} or {tuple(base.shape[1:])})"
            )
        if mask_grid.shape[1] != expected_channels:
            raise ValueError(
                f"RecFNO observation mask has {mask_grid.shape[1]} channels, expected {expected_channels}"
            )
        x = torch.cat([base, mask_grid, grid_channels(base)], dim=1)
        return self.net(x)


def _clamped_modes(config: dict, data_spec: dict) -> tuple[int, int]:
    shape = tuple(data_spec.get("target_shape", ()))
    h = int(shape[-2]) if len(shape) >= 2 else 32
    w = int(shape[-1]) if len(shape) >= 1 else h
    return max(1, min(int(config.get("modes1", 12)), h)), max(1, min(int(config.get("modes2", 12)), w // 2 + 1))
