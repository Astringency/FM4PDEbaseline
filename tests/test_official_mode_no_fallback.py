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


def _typeerror_neuraloperator(monkeypatch):
    class BrokenFNO:
        def __init__(self, *_args, **_kwargs):
            raise TypeError("signature changed")

    def broken_neuralop():
        return BrokenFNO

    def missing_recfno(*_args, **_kwargs):
        raise OfficialImportError("no recfno")

    monkeypatch.setattr(fno_module, "get_neuraloperator_fno_class", broken_neuralop)
    monkeypatch.setattr(fno_module, "OfficialRecFNOVoronoiFNO2dNet", missing_recfno)


def test_paper_strict_official_missing_backend_raises(monkeypatch, tiny_data_root, tmp_path: Path):
    _missing_fno_backends(monkeypatch)
    config = tmp_path / "strict_official.yaml"
    config.write_text(yaml.safe_dump({"method": {"implementation_mode": "official", "official_backend": "auto"}, "epochs": 1}), encoding="utf-8")

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


def test_paper_official_or_skip_runs_official_aligned_when_allowed(tiny_data_root, tmp_path: Path):
    config = tmp_path / "ifno.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "method": {
                    "implementation_mode": "official_or_skip",
                    "official_backend": "ifno",
                    "width": 8,
                    "modes1": 4,
                    "modes2": 4,
                    "layers": 1,
                    "max_steps": 1,
                },
                "epochs": 1,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "ifno_out"
    main(
        [
            "--baseline",
            "ifno",
            "--pde",
            "darcy",
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
    row = yaml.safe_load((out / "results_summary.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["implementation_mode_requested"] == "official_or_skip"
    assert row["implementation_mode_effective"] == "official_aligned"
    assert row["paper_table_eligible"] is False
    assert row["adapter_status"] == "official_training_ifno_task_adapter"
    assert row["official_alignment_level"] == "algorithm_training"
    assert not (out / "skipped_combinations.jsonl").exists()


def test_official_or_skip_wraps_constructor_typeerror_as_skip(monkeypatch, tiny_data_root, tmp_path: Path):
    _typeerror_neuraloperator(monkeypatch)
    config = tmp_path / "paper.yaml"
    config.write_text(yaml.safe_dump({"method": {"implementation_mode": "official_or_skip", "official_backend": "auto"}, "epochs": 1}), encoding="utf-8")
    out = tmp_path / "out_typeerror"

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

    row = yaml.safe_load((out / "skipped_combinations.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert "signature changed" in row["backend_warning"]
    assert not (out / "results_summary.jsonl").exists()


def test_strict_official_wraps_constructor_typeerror_as_raise(monkeypatch, tiny_data_root, tmp_path: Path):
    _typeerror_neuraloperator(monkeypatch)
    config = tmp_path / "strict.yaml"
    config.write_text(yaml.safe_dump({"method": {"implementation_mode": "official", "official_backend": "auto"}, "epochs": 1}), encoding="utf-8")
    with pytest.raises(OfficialImportError, match="signature changed"):
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
                str(tmp_path / "strict_typeerror"),
            ]
        )
