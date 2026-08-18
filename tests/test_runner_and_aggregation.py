from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from baselines.aggregate_results import aggregate_rows


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
    assert sample_pdf.read_bytes().startswith(b"%PDF")
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
    assert summary["sample_pdf_path"] == str(sample_pdf)


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
    paper_scripts = [
        root / "scripts/experiments/01_run_sanity_main.sh",
        root / "scripts/experiments/02_run_main_results_local.sh",
        root / "scripts/experiments/04_run_ablation_local.sh",
        root / "scripts/experiments/05_run_one.sh",
    ]
    forbidden = ["--prefer-test", "--synthetic-data", "--dry-run"]
    for script in paper_scripts:
        text = script.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text
