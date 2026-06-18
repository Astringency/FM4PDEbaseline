from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import baselines.methods.fno as fno_module
from baselines.methods.official import OfficialImportError
from baselines.run import main


def test_paper_official_or_skip_does_not_use_local_fallback(monkeypatch, tiny_data_root, tmp_path: Path):
    def missing_neuralop():
        raise OfficialImportError("no neuraloperator")

    def missing_recfno(*_args, **_kwargs):
        raise OfficialImportError("no recfno")

    monkeypatch.setattr(fno_module, "get_neuraloperator_fno_class", missing_neuralop)
    monkeypatch.setattr(fno_module, "OfficialRecFNOVoronoiFNO2dNet", missing_recfno)
    config = tmp_path / "paper.yaml"
    config.write_text(yaml.safe_dump({"method": {"implementation_mode": "official_or_skip", "official_backend": "auto"}, "epochs": 1}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="requested official backend"):
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
                "--test-size",
                "1",
                "--batch-size",
                "1",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
