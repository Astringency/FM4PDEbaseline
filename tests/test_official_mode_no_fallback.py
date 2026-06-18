from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import baselines.methods.fno as fno_module
from baselines.methods.official import OfficialImportError
from baselines.run import main


def _missing_fno_backends(monkeypatch):
    def missing_neuralop():
        raise OfficialImportError("no neuraloperator")

    def missing_recfno(*_args, **_kwargs):
        raise OfficialImportError("no recfno")

    monkeypatch.setattr(fno_module, "get_neuraloperator_fno_class", missing_neuralop)
    monkeypatch.setattr(fno_module, "OfficialRecFNOVoronoiFNO2dNet", missing_recfno)


def test_paper_strict_official_missing_backend_raises(monkeypatch, tiny_data_root, tmp_path: Path):
    _missing_fno_backends(monkeypatch)
    config = tmp_path / "strict_official.yaml"
    config.write_text(yaml.safe_dump({"method": {"implementation_mode": "official", "official_backend": "auto"}, "epochs": 1}), encoding="utf-8")

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
                str(tmp_path / "strict_out"),
            ]
        )


def test_paper_official_or_skip_missing_backend_writes_skip(monkeypatch, tiny_data_root, tmp_path: Path):
    _missing_fno_backends(monkeypatch)
    config = tmp_path / "paper.yaml"
    config.write_text(yaml.safe_dump({"method": {"implementation_mode": "official_or_skip", "official_backend": "auto"}, "epochs": 1}), encoding="utf-8")
    out = tmp_path / "out"

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
            str(out),
        ]
    )

    skipped = out / "skipped_combinations.jsonl"
    assert skipped.exists()
    row = yaml.safe_load(skipped.read_text(encoding="utf-8").splitlines()[-1])
    assert row["baseline"] == "fno"
    assert row["implementation_mode_requested"] == "official_or_skip"
    assert row["implementation_mode_effective"] == "adapted"
    assert row["fallback_used"] is True
    assert row["paper_table_eligible"] is False
    assert not (out / "results_summary.jsonl").exists()
