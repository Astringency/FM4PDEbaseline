from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from baselines.run import main as run_baseline
from scripts.experiments.build_matrix import write_outputs
from scripts.experiments.run_one import run_one as run_matrix_row
from scripts.run_remaining_plan_v2 import (
    CORRECTED_NAMESPACE,
    HISTORICAL_NAMESPACE,
    MATRIX,
    HistoricalExperimentError,
    load_rows,
    quarantine_invalid_output,
)


ROOT = Path(__file__).resolve().parents[1]


def test_remaining_runner_defaults_to_the_corrected_namespace():
    assert CORRECTED_NAMESPACE == "experiment_plan_v2_corrected"
    assert MATRIX.name == "experiment_plan_v2_corrected.jsonl"
    assert CORRECTED_NAMESPACE in MATRIX.parts
    assert HISTORICAL_NAMESPACE not in MATRIX.parts


def test_remaining_runner_rejects_historical_matrix_without_moving_evidence(tmp_path: Path):
    matrix = (
        tmp_path
        / "outputs"
        / HISTORICAL_NAMESPACE
        / "matrices"
        / f"{HISTORICAL_NAMESPACE}.jsonl"
    )
    output_dir = (
        tmp_path
        / "outputs"
        / HISTORICAL_NAMESPACE
        / "runs"
        / HISTORICAL_NAMESPACE
        / "historical-run"
    )
    output_dir.mkdir(parents=True)
    sentinel = output_dir / "summary.json"
    sentinel.write_text('{"historical": true}\n', encoding="utf-8")
    matrix.parent.mkdir(parents=True)
    matrix.write_text(
        json.dumps(
            {
                "run_id": "historical-run",
                "output_dir": str(output_dir),
                "task_group": "sparse_forward_main_physics",
                "pde": "poisson",
                "baseline": "pde_opt",
                "skip_reason": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_remaining_plan_v2.py",
            "--matrix",
            str(matrix),
            "--dry-run",
        ],
        cwd=ROOT,
        env={**os.environ, "OUTPUT_ROOT": str(tmp_path / "outputs")},
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "historical" in result.stderr.lower()
    assert sentinel.read_text(encoding="utf-8") == '{"historical": true}\n'
    assert not (output_dir / "quarantine").exists()


def test_quarantine_helper_refuses_historical_artifacts(tmp_path: Path):
    output_dir = (
        tmp_path
        / "outputs"
        / HISTORICAL_NAMESPACE
        / "runs"
        / HISTORICAL_NAMESPACE
        / "historical-run"
    )
    output_dir.mkdir(parents=True)
    sentinel = output_dir / "results_raw.jsonl"
    sentinel.write_text('{"historical": true}\n', encoding="utf-8")

    with pytest.raises(HistoricalExperimentError, match="historical"):
        quarantine_invalid_output(
            {
                "run_id": "historical-run",
                "output_dir": str(output_dir),
            }
        )

    assert sentinel.exists()
    assert not (output_dir / "quarantine").exists()


def test_all_direct_mutating_entrypoints_refuse_historical_namespace(tmp_path: Path):
    historical = tmp_path / "outputs" / HISTORICAL_NAMESPACE
    row = {
        "run_id": "historical-run",
        "output_dir": str(historical / "run"),
    }

    with pytest.raises(HistoricalExperimentError, match="historical"):
        run_matrix_row(row, ["must-not-run"])
    with pytest.raises(HistoricalExperimentError, match="historical"):
        write_outputs([], [], {}, historical, "safe_matrix")
    with pytest.raises(HistoricalExperimentError, match="historical"):
        run_baseline(
            [
                "--baseline",
                "fno",
                "--pde",
                "poisson",
                "--output-dir",
                str(historical / "core-run"),
            ]
        )
    assert not historical.exists()


def test_matrix_writer_rejects_path_traversal_name_before_writing(tmp_path: Path):
    output = tmp_path / "safe-output"
    with pytest.raises(ValueError, match="safe basename"):
        write_outputs([], [], {}, output, "../experiment_plan_v2")
    assert not output.exists()


def test_remaining_runner_normalizes_paths_before_accepting_corrected_namespace(tmp_path: Path):
    matrix = (
        tmp_path
        / CORRECTED_NAMESPACE
        / "matrices"
        / f"{CORRECTED_NAMESPACE}.jsonl"
    )
    matrix.parent.mkdir(parents=True)
    escaped_output = matrix.parent / ".." / ".." / "outside" / "run"
    matrix.write_text(
        json.dumps(
            {
                "run_id": "namespace-traversal",
                "output_dir": str(escaped_output),
                "task_group": "full_forward_main",
                "pde": "poisson",
                "baseline": "fno",
                "skip_reason": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside the required"):
        load_rows(matrix)


def test_formal_matrix_builders_require_and_forward_exact_data_manifests():
    build_all = (ROOT / "scripts/experiments/00_build_matrices.sh").read_text(encoding="utf-8")
    build_main = (ROOT / "scripts/experiments/07_build_main_results_matrix.sh").read_text(
        encoding="utf-8"
    )

    assert "DATA_MANIFEST_DIR" in build_all
    assert '${DATA_MANIFEST_DIR}/${matrix}/full/data_protocol_report.json' in build_all
    assert '--data-manifest "$DATA_MANIFEST"' in build_all
    assert 'DATA_MANIFEST="${DATA_MANIFEST:-}"' in build_main
    assert '--data-manifest "$DATA_MANIFEST"' in build_main


def test_build_all_refuses_to_mutate_outputs_without_manifest_directory(tmp_path: Path):
    output_root = tmp_path / "formal-output"
    env = os.environ.copy()
    env.pop("DATA_MANIFEST_DIR", None)
    env["OUT_ROOT"] = str(output_root)

    result = subprocess.run(
        ["bash", "scripts/experiments/00_build_matrices.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "DATA_MANIFEST_DIR is required" in result.stderr
    assert not output_root.exists()


def test_build_main_refuses_to_mutate_outputs_without_data_manifest(tmp_path: Path):
    data_root = tmp_path / "PDEdata"
    data_root.mkdir()
    output_root = tmp_path / "formal-output"
    env = os.environ.copy()
    env.pop("DATA_MANIFEST", None)
    env.update({"DATA_ROOT": str(data_root), "OUT_ROOT": str(output_root)})

    result = subprocess.run(
        ["bash", "scripts/experiments/07_build_main_results_matrix.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "DATA_MANIFEST is required" in result.stderr
    assert not output_root.exists()


def test_parallel_launcher_executes_the_fingerprinted_matrix_row_unchanged():
    launcher = (ROOT / "scripts/experiments/08_run_matrix_2gpu_parallel.sh").read_text(
        encoding="utf-8"
    )

    assert "05_run_one.sh" in launcher
    assert "CUDA_VISIBLE_DEVICES" in launcher
    assert "unset ALLOW_ROW_OVERRIDE" in launcher
    assert "export ALLOW_ROW_OVERRIDE=1" not in launcher
    assert "NUM_WORKERS_PER_RUN" not in launcher
    assert "export NUM_WORKERS=" not in launcher
