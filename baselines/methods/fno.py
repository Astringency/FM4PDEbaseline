from __future__ import annotations

import warnings

from baselines.common.data_adapter import PDEBatch

from .base import BaselineModel, run_supervised_fit
from .official import (
    OfficialImportError,
    allows_adapted_fallback,
    get_neuraloperator_fno_class,
    official_source_info,
    requested_implementation_mode,
    wrap_official_adapter_error,
)
from .shared import FNO2dNet, OfficialRecFNOVoronoiFNO2dNet


class FNOBaseline(BaselineModel):
    name = "fno"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        in_channels = int(data_spec["input_channels"])
        out_channels = int(data_spec["target_channels"])
        width = int(self.config.get("width", 24))
        layers = int(self.config.get("layers", 4))
        modes1, modes2 = _clamped_modes(self.config, data_spec)
        backend = str(self.config.get("official_backend", "auto")).lower()
        implementation_mode = requested_implementation_mode(self.config)
        fallback_reasons: list[str] = []
        if implementation_mode == "official" and backend == "recfno":
            raise OfficialImportError("RecFNO VoronoiFNO2d is not an official vanilla FNO backend; use neuraloperator or a recorded vendored zongyi-li FNO")

        if implementation_mode != "adapted" and backend in {"auto", "neuraloperator", "official"}:
            try:
                neuralop_fno = get_neuraloperator_fno_class()
                self.net = neuralop_fno(
                    n_modes=(modes1, modes2),
                    in_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=width,
                    n_layers=layers,
                    positional_embedding="grid",
                )
                self.set_backend(
                    "neuraloperator",
                    "neuraloperator",
                    fallback_used=False,
                    implementation_mode_effective="adapted",
                    implementation_source="neuraloperator_2_component_unified_training",
                    official_import_success=True,
                    official_alignment_level="component",
                    official_alignment_notes=(
                        "Uses the vendored NeuralOperator 2.0 FNO network component with FM4PDE's unified "
                        "normalization, pointwise MSE, optimizer, scheduler, early stopping, and data protocol."
                    ),
                    adapter_status="official_component_unified_training_adapter",
                    **official_source_info("neuraloperator"),
                )
                return self
            except Exception as exc:
                exc = wrap_official_adapter_error("neuraloperator FNO", exc)
                fallback_reasons.append(f"neuraloperator unavailable: {exc}")
                if backend in {"neuraloperator", "official"}:
                    warnings.warn(f"neuraloperator FNO unavailable, using fallback: {exc}", RuntimeWarning, stacklevel=2)

        if implementation_mode == "official" and fallback_reasons:
            raise OfficialImportError("; ".join(fallback_reasons))

        if implementation_mode != "official" and implementation_mode != "adapted" and backend == "recfno":
            try:
                self.net = OfficialRecFNOVoronoiFNO2dNet(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    width=width,
                    modes1=modes1,
                    modes2=modes2,
                )
                self.set_backend(
                    "recfno_component",
                    "recfno_component",
                    fallback_used=True,
                    warning="RecFNO VoronoiFNO2d is an adapted component for FNO and is supplement-only",
                    implementation_mode_effective="adapted",
                    implementation_source="recfno_voronoifno_component_adapted_for_fno",
                    official_import_success=True,
                    official_reimplementation_success=False,
                    official_alignment_level="adapted_component",
                    official_alignment_notes=(
                        "Uses the RecFNO VoronoiFNO2d component as an adapted FNO-style network; "
                        "this is not vanilla FNO official code and is not paper main-table eligible."
                    ),
                    adapter_status="adapted_recfno_component_supplement_only",
                    **official_source_info("recfno"),
                )
                return self
            except Exception as exc:
                exc = wrap_official_adapter_error("RecFNO VoronoiFNO2d", exc)
                fallback_reasons.append(f"recfno unavailable: {exc}")
                if backend == "recfno":
                    warnings.warn(f"RecFNO official FNO unavailable, using local fallback: {exc}", RuntimeWarning, stacklevel=2)

        self.net = FNO2dNet(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes1=modes1,
            modes2=modes2,
            layers=layers,
        )
        requested_local = backend in {"local", "none"} or implementation_mode == "adapted"
        fallback_used = not requested_local
        if not allows_adapted_fallback(self.config) and fallback_used:
            warning = "; ".join(fallback_reasons) or "official FNO backend unavailable"
        else:
            warning = "; ".join(fallback_reasons)
        self.set_backend(
            "local",
            "local" if requested_local else ("official" if backend == "official" else backend),
            fallback_used=fallback_used,
            warning=warning,
            implementation_mode_effective="adapted",
            implementation_source="local_compact_fno",
            official_import_success=False,
            adapter_status="local_adapted" if requested_local else "fallback_adapted",
        )
        return self

    def fit(self, train_loader, val_loader=None):
        return run_supervised_fit(self, train_loader, val_loader)

    def predict(self, batch: PDEBatch):
        return self.net(batch.input_fields)


def _clamped_modes(config: dict, data_spec: dict) -> tuple[int, int]:
    shape = tuple(data_spec.get("target_shape", ()))
    h = int(shape[-2]) if len(shape) >= 2 else 32
    w = int(shape[-1]) if len(shape) >= 1 else h
    modes1 = min(int(config.get("modes1", 12)), h)
    modes2 = min(int(config.get("modes2", 12)), w // 2 + 1)
    return max(modes1, 1), max(modes2, 1)
