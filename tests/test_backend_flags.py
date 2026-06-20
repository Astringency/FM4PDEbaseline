from __future__ import annotations

from pathlib import Path

import pytest
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
