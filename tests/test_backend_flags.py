from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

import baselines.methods.fno as fno_module
from baselines.common.data_adapter import build_default_registry
from baselines.methods.fno import FNOBaseline
from baselines.methods.official import OfficialImportError
from baselines.run import build_data_spec, main


def _data_spec():
    registry = build_default_registry()
    raw = registry.synthetic_raw("darcy", n=2, resolution=8)
    batch = registry.make_task(raw, "darcy", "forward")
    return build_data_spec(batch)


def test_fno_explicit_local_backend_not_fallback():
    model = FNOBaseline().build({"official_backend": "local"}, _data_spec())
    assert model.backend_used == "local"
    assert model.official_backend == "local"
    assert model.fallback_used is False
    assert model.backend_warning == ""


def test_fno_auto_unavailable_marks_local_fallback(monkeypatch):
    def missing_neuralop():
        raise OfficialImportError("no neuraloperator")

    def missing_recfno(*_args, **_kwargs):
        raise OfficialImportError("no recfno")

    monkeypatch.setattr(fno_module, "get_neuraloperator_fno_class", missing_neuralop)
    monkeypatch.setattr(fno_module, "OfficialRecFNOVoronoiFNO2dNet", missing_recfno)
    model = FNOBaseline().build({"official_backend": "auto"}, _data_spec())
    assert model.backend_used == "local"
    assert model.fallback_used is True
    assert "unavailable" in model.backend_warning


def test_fno_recfno_component_is_adapted_not_official(monkeypatch):
    def missing_neuralop():
        raise OfficialImportError("no neuraloperator")

    class DummyRecFNO(torch.nn.Module):
        def __init__(self, in_channels, out_channels, width=8, modes1=4, modes2=4, add_coords=True):
            super().__init__()
            self.proj = torch.nn.Conv2d(in_channels + (2 if add_coords else 0), out_channels, 1)
            self.add_coords = add_coords

        def forward(self, x):
            if self.add_coords:
                coords = torch.zeros(x.shape[0], 2, x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)
                x = torch.cat([x, coords], dim=1)
            return self.proj(x)

    monkeypatch.setattr(fno_module, "get_neuraloperator_fno_class", missing_neuralop)
    monkeypatch.setattr(fno_module, "OfficialRecFNOVoronoiFNO2dNet", DummyRecFNO)

    model = FNOBaseline().build({"implementation_mode": "official_or_skip", "official_backend": "recfno"}, _data_spec())
    backend = model.backend_metadata()
    assert backend["backend_used"] == "recfno_component"
    assert backend["implementation_mode_effective"] == "adapted"
    assert backend["fallback_used"] is True
    assert "adapted" in backend["adapter_status"]
    assert "RecFNO" in backend["official_repo"] or "recfno" in backend["official_repo"].lower()


def test_paper_official_unavailable_raises(monkeypatch, tiny_data_root, tmp_path: Path):
    def missing_neuralop():
        raise OfficialImportError("no neuraloperator")

    def missing_recfno(*_args, **_kwargs):
        raise OfficialImportError("no recfno")

    monkeypatch.setattr(fno_module, "get_neuraloperator_fno_class", missing_neuralop)
    monkeypatch.setattr(fno_module, "OfficialRecFNOVoronoiFNO2dNet", missing_recfno)
    config = tmp_path / "paper_official.yaml"
    config.write_text(yaml.safe_dump({"method": {"official_backend": "official"}, "epochs": 1}), encoding="utf-8")
    with pytest.raises(OfficialImportError, match="unavailable"):
        main(
            [
                "--baseline",
                "fno",
                "--pde",
                "poisson",
                "--task",
                "forward",
                "--experiment-mode",
                "paper",
                "--task-protocol-version",
                "backend-test-v1",
                "--sensor-protocol-version",
                "backend-test-v1",
                "--data-root",
                str(tiny_data_root),
                "--config",
                str(config),
                "--train-size",
                "2",
                "--val-size",
                "0",
                "--test-size",
                "1",
                "--batch-size",
                "1",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
