from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from baselines.common.sample_artifacts import load_evaluation_sample


def test_per_sample_physics_metric_values_and_summary_counts(tmp_path: Path):
    out = tmp_path / "metrics"
    cmd = [
        sys.executable,
        "-m",
        "baselines.run",
        "--baseline",
        "fno",
        "--pde",
        "darcy",
        "--task",
        "forward",
        "--experiment-mode",
        "debug",
        "--dry-run",
        "--synthetic-data",
        "--physics-metric-mode",
        "per_sample",
        "--synthetic-resolution",
        "8",
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
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    raw_rows = [json.loads(line) for line in (out / "results_raw.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(raw_rows) == 2
    assert all(len(json.loads(row["pde_residual_values"])) == 2 for row in raw_rows)
    summary = json.loads((out / "results_summary.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert summary["metric_granularity"] == "per_sample"
    assert summary["relative_l2_solution_n"] == 4
    assert summary["pde_residual_n"] == 4
    assert sum(json.loads(summary["residual_mode_counts"]).values()) == 4


def test_per_batch_summary_still_persists_true_per_sample_physics_metrics(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]

    def run(mode: str) -> Path:
        out = tmp_path / mode
        cmd = [
            sys.executable, "-m", "baselines.run",
            "--baseline", "fno", "--pde", "darcy", "--task", "forward",
            "--experiment-mode", "debug", "--dry-run", "--synthetic-data",
            "--physics-metric-mode", mode, "--synthetic-resolution", "8",
            "--test-size", "2", "--train-size", "4", "--val-size", "0",
            "--batch-size", "2", "--output-dir", str(out),
        ]
        subprocess.run(cmd, check=True, cwd=root)
        return out

    per_batch = run("per_batch")
    per_sample = run("per_sample")

    def artifact_physics(out: Path) -> list[float]:
        rows = [json.loads(line) for line in (out / "samples" / "manifest.jsonl").read_text().splitlines()]
        return [load_evaluation_sample(row["artifact_path"])["metrics"]["physics_loss"] for row in rows]

    assert artifact_physics(per_batch) == pytest.approx(artifact_physics(per_sample))
