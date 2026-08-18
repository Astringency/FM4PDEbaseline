from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from baselines.run import main as run_baseline
from scripts.experiments.build_matrix import write_outputs
from scripts.experiments.provenance import HISTORICAL_EXPERIMENT_NAMESPACE, HistoricalExperimentError
from scripts.experiments.run_one import run_one as run_matrix_row
from scripts.run_experiments import quarantine_invalid_output


ROOT = Path(__file__).resolve().parents[1]


def test_unified_runner_rejects_historical_matrix_without_moving_evidence(tmp_path: Path):
    historical = HISTORICAL_EXPERIMENT_NAMESPACE
    output_dir = tmp_path / historical / "runs" / "historical-run"
    output_dir.mkdir(parents=True)
    sentinel = output_dir / "summary.json"
    sentinel.write_text('{"historical": true}\n', encoding="utf-8")
    matrix = tmp_path / historical / "matrices" / f"{historical}.jsonl"
    matrix.parent.mkdir(parents=True)
    matrix.write_text(json.dumps({"run_id": "historical-run", "output_dir": str(output_dir)}) + "\n")

    result = subprocess.run(
        [sys.executable, "scripts/run_experiments.py", str(matrix), "--data-root", str(tmp_path), "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "historical" in result.stderr.lower()
    assert sentinel.exists()
    assert not (output_dir / "quarantine").exists()


def test_quarantine_helper_refuses_historical_artifacts(tmp_path: Path):
    output_dir = tmp_path / HISTORICAL_EXPERIMENT_NAMESPACE / "historical-run"
    output_dir.mkdir(parents=True)
    sentinel = output_dir / "results_raw.jsonl"
    sentinel.write_text("{}\n", encoding="utf-8")
    with pytest.raises(HistoricalExperimentError, match="historical"):
        quarantine_invalid_output({"run_id": "historical-run", "output_dir": str(output_dir)})
    assert sentinel.exists()


def test_all_direct_mutating_entrypoints_refuse_historical_namespace(tmp_path: Path):
    historical = tmp_path / HISTORICAL_EXPERIMENT_NAMESPACE
    row = {"run_id": "historical-run", "output_dir": str(historical / "run")}
    with pytest.raises(HistoricalExperimentError):
        run_matrix_row(row, ["must-not-run"])
    with pytest.raises(HistoricalExperimentError):
        write_outputs([], [], {}, historical, "safe_matrix")
    with pytest.raises(HistoricalExperimentError):
        run_baseline(["--baseline", "fno", "--pde", "poisson", "--output-dir", str(historical / "core")])
    assert not historical.exists()


def test_formal_builder_requires_data_manifest_before_writing(tmp_path: Path):
    output = tmp_path / "formal-output"
    result = subprocess.run(
        [sys.executable, "scripts/build_experiment_matrix.py", "--output-root", str(output)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "--data-manifest is required" in result.stderr
    assert not output.exists()
