from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

import pytest
import torch

from baselines.aggregate_results import aggregate_rows


def test_burger_full_trajectory_metrics_persist_and_reject_legacy_resume(tmp_path: Path):
    out = tmp_path / "burger"
    cmd = [
        sys.executable, "-m", "baselines.run",
        "--baseline", "recfno", "--pde", "burger", "--task", "sparse_solution",
        "--experiment-mode", "smoke", "--dry-run", "--synthetic-data",
        "--synthetic-resolution", "8", "--test-size", "2", "--train-size", "4",
        "--val-size", "0", "--batch-size", "1", "--num-sensors", "4",
        "--output-dir", str(out), "--resume-eval",
    ]
    root = Path(__file__).resolve().parents[1]
    subprocess.run(cmd, check=True, cwd=root, capture_output=True, text=True)

    full_errors = []
    initial_errors = []
    for path in sorted((out / "samples").glob("sample_*.pt")):
        sample = torch.load(path, weights_only=False)
        target = sample["target_fields"]
        pred = sample["prediction"]
        assert target.shape == pred.shape == (1, 8, 8)
        full_errors.append(float(((pred - target).square().sum() / target.square().sum()).sqrt()))
        initial_errors.append(float(
            ((pred[:, 0] - target[:, 0]).square().sum() / target[:, 0].square().sum()).sqrt()
        ))
    assert len(full_errors) == 2
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["relative_l2_solution_scope"] == "full_trajectory"
    assert summary["relative_l2_solution_mean"] == pytest.approx(mean(full_errors))
    assert summary["relative_l2_solution_std"] == pytest.approx(stdev(full_errors), abs=1e-7)
    assert summary["relative_l2_input_or_coeff_mean"] == pytest.approx(mean(initial_errors))

    # Batches evaluated under the current definition remain resumable.
    subprocess.run(cmd, check=True, cwd=root, capture_output=True, text=True)
    resumed = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert resumed["resumed_sample_count"] == 2
    assert resumed["relative_l2_solution_mean"] == summary["relative_l2_solution_mean"]

    # Historical batches have no scope marker; never pool them with new errors.
    raw_path = out / "results_raw.jsonl"
    legacy_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    for row in legacy_rows:
        row.pop("relative_l2_solution_scope")
    legacy_text = "".join(json.dumps(row) + "\n" for row in legacy_rows)
    raw_path.write_text(legacy_text, encoding="utf-8")
    rejected = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "relative_l2_solution_scope" in rejected.stderr
    assert "Cannot resume evaluation" in rejected.stderr
    assert raw_path.read_text(encoding="utf-8") == legacy_text


def test_runner_synthetic_full_test_loader_outputs_raw_and_summary(tmp_path: Path):
    out = tmp_path / "run"
    cmd = [
        sys.executable,
        "-m",
        "baselines.run",
        "--baseline",
        "fno",
        "--pde",
        "heat",
        "--task",
        "forward",
        "--experiment-mode",
        "smoke",
        "--dry-run",
        "--synthetic-data",
        "--scalar-param-mode",
        "metadata",
        "--test-size",
        "4",
        "--train-size",
        "8",
        "--batch-size",
        "2",
        "--output-dir",
        str(out),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    raw_path = out / "results_raw.jsonl"
    summary_path = out / "results_summary.jsonl"
    csv_path = out / "results_summary.csv"
    assert raw_path.exists()
    assert summary_path.exists()
    assert csv_path.exists()
    sample_dir = out / "samples"
    sample_manifest = sample_dir / "manifest.jsonl"
    sample_pdf = out / "samples.pdf"
    assert len(list(sample_dir.glob("sample_*.pt"))) == 4
    assert len(sample_manifest.read_text(encoding="utf-8").splitlines()) == 4
    assert not sample_pdf.exists()
    raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    assert len(raw_rows) == 2
    assert [row["batch_index"] for row in raw_rows] == [0, 1]
    assert sum(row["sample_count"] for row in raw_rows) == 4
    summary = json.loads(summary_path.read_text(encoding="utf-8").splitlines()[-1])
    assert summary["test_size"] == 4
    assert summary["relative_l2_solution_n"] == 4
    assert "backend_used" in summary
    assert "residual_mode_counts" in summary
    assert summary["sample_artifact_count"] == 4
    assert summary["sample_manifest_path"] == str(sample_manifest)
    assert summary["sample_pdf_path"] == ""

    # The default non-resume mode is a clean rerun, not an append into stale files.
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    assert len(raw_path.read_text(encoding="utf-8").splitlines()) == 2
    assert len(summary_path.read_text(encoding="utf-8").splitlines()) == 1


def test_runner_resumes_committed_batches_without_duplicate_raw_rows(tmp_path: Path):
    out = tmp_path / "resume"
    cmd = [
        sys.executable,
        "-m",
        "baselines.run",
        "--baseline",
        "fno",
        "--pde",
        "heat",
        "--task",
        "forward",
        "--experiment-mode",
        "smoke",
        "--dry-run",
        "--synthetic-data",
        "--test-size",
        "4",
        "--train-size",
        "8",
        "--val-size",
        "0",
        "--batch-size",
        "2",
        "--output-dir",
        str(out),
        "--resume-eval",
    ]
    root = Path(__file__).resolve().parents[1]
    subprocess.run(cmd, check=True, cwd=root, capture_output=True, text=True)

    raw_path = out / "results_raw.jsonl"
    first_batch = raw_path.read_text(encoding="utf-8").splitlines()[0]
    raw_path.write_text(first_batch + "\n{\"truncated\":", encoding="utf-8")
    for path in (
        out / "summary.json",
        out / "results_summary.jsonl",
        out / "results_summary.csv",
        out / "results_summary_latest.csv",
    ):
        path.unlink(missing_ok=True)

    resumed = subprocess.run(cmd, check=True, cwd=root, capture_output=True, text=True)
    raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    manifest_rows = (out / "samples" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))

    assert [row["batch_index"] for row in raw_rows] == [0, 1]
    assert len(manifest_rows) == 4
    assert summary["relative_l2_solution_n"] == 4
    assert summary["evaluation_resumed"] is True
    assert summary["resumed_batch_count"] == 1
    assert summary["resumed_sample_count"] == 2
    assert "eval_resume trimmed_invalid_tail retained_batches=1" in resumed.stderr
    assert "eval_resume loaded batches=1 samples=2" in resumed.stderr


def test_runner_resumes_when_sample_artifacts_are_disabled(tmp_path: Path):
    out = tmp_path / "resume-no-samples"
    cmd = [
        sys.executable, "-m", "baselines.run",
        "--baseline", "fno", "--pde", "heat", "--task", "forward",
        "--experiment-mode", "smoke", "--dry-run", "--synthetic-data",
        "--test-size", "3", "--train-size", "4", "--val-size", "0",
        "--batch-size", "1", "--output-dir", str(out),
        "--no-save-sample-artifacts", "--resume-eval",
    ]
    root = Path(__file__).resolve().parents[1]
    subprocess.run(cmd, check=True, cwd=root, capture_output=True, text=True)
    raw_path = out / "results_raw.jsonl"
    first_row = raw_path.read_text(encoding="utf-8").splitlines()[0]
    raw_path.write_text(first_row + "\n", encoding="utf-8")
    for path in (
        out / "summary.json",
        out / "results_summary.jsonl",
        out / "results_summary.csv",
        out / "results_summary_latest.csv",
    ):
        path.unlink(missing_ok=True)

    subprocess.run(cmd, check=True, cwd=root, capture_output=True, text=True)
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert [row["batch_index"] for row in rows] == [0, 1, 2]
    assert summary["relative_l2_solution_n"] == 3
    assert summary["resumed_batch_count"] == 1
    assert summary["sample_artifact_count"] == 0


def test_aggregate_results_mean_std_ci_nan_and_residual_counts():
    rows = [
        {
            "pde": "heat",
            "task": "forward",
            "baseline": "fno",
            "train_size": 8,
            "scalar_param_mode": "metadata",
            "num_sensors": 0,
            "sensor_mode": "none",
            "noise_level": 0.0,
            "backend_used": "local",
            "relative_l2_solution_values": "[1.0, 3.0]",
            "mse_values": "[1.0, NaN]",
            "mae_values": "[2.0, 2.0]",
            "obs_mse": float("nan"),
            "pde_residual": 0.5,
            "bc_residual": 0.0,
            "ic_residual": 0.0,
            "physics_loss": 0.5,
            "residual_mode": "two_level",
            "sample_count": 2,
        }
    ]
    summary = aggregate_rows(rows)[0]
    assert summary["relative_l2_solution_mean"] == 2.0
    assert summary["relative_l2_solution_n"] == 2
    assert summary["mse_nan_count"] == 1
    assert json.loads(summary["residual_mode_counts"]) == {"two_level": 2}


def test_experiment_scripts_do_not_default_to_debug_data_flags():
    root = Path(__file__).resolve().parents[1]
    paper_scripts = [root / "scripts/run_experiments.py"]
    forbidden = ["--prefer-test", "--synthetic-data", "--allow-synthetic-fallback"]
    for script in paper_scripts:
        text = script.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text
