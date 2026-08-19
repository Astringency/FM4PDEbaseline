from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_different_noise_levels_have_different_output_dirs(tmp_path: Path):
    out = tmp_path / "large"
    subprocess.run(
        [
            sys.executable,
            "scripts/experiments/build_matrix.py",
            "--config",
            "configs/experiments/noise_ablation.yaml",
            "--output-root",
            str(out),
            "--matrix-name",
            "noise_ablation",
        ],
        cwd=ROOT,
        check=True,
    )
    rows = [
        row
        for row in _read_jsonl(out / "matrices" / "noise_ablation.jsonl")
        if row["task_group"] == "noise_ablation"
        and row["pde"] == "darcy"
        and row["baseline"] == "recfno"
        and row["seed"] == 1
        and row["num_sensors"] == 500
        and row["sensor_mode"] == "random_per_sample"
    ]
    assert len({row["noise_level"] for row in rows}) >= 2
    assert len({row["output_dir"] for row in rows}) == len(rows)


def test_run_id_makes_config_paths_unique_in_shared_output_dir(tmp_path: Path):
    out = tmp_path / "shared"
    base_cmd = [
        sys.executable,
        "-m",
        "baselines.run",
        "--baseline",
        "fno",
        "--pde",
        "heat",
        "--task",
        "sparse_solution",
        "--experiment-mode",
        "smoke",
        "--dry-run",
        "--synthetic-data",
        "--synthetic-resolution",
        "8",
        "--train-size",
        "4",
        "--val-size",
        "0",
        "--test-size",
        "1",
        "--batch-size",
        "1",
        "--epochs",
        "1",
        "--num-sensors",
        "4",
        "--output-dir",
        str(out),
    ]
    subprocess.run(base_cmd + ["--noise-level", "0.0", "--run-id", "noise0"], cwd=ROOT, check=True)
    subprocess.run(base_cmd + ["--noise-level", "0.1", "--run-id", "noise1"], cwd=ROOT, check=True)
    summaries = _read_jsonl(out / "results_summary.jsonl")
    config_paths = [Path(row["config_path"]).name for row in summaries[-2:]]
    assert config_paths == ["noise0_config.json", "noise1_config.json"]
    assert (out / "noise0_results_summary_latest.csv").exists()
    assert (out / "noise1_results_summary_latest.csv").exists()
