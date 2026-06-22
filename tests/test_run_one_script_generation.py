from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_run_one_generates_paper_sparse_command_without_debug_flags(tmp_path: Path):
    out = tmp_path / "large"
    subprocess.run(
        [
            sys.executable,
            "scripts/experiments/build_matrix.py",
            "--config",
            "configs/experiments/sanity_main.yaml",
            "--output-root",
            str(out),
            "--matrix-name",
            "sanity_main",
        ],
        cwd=ROOT,
        check=True,
    )
    rows = _read_jsonl(out / "matrices" / "sanity_main.jsonl")
    index = next(i for i, row in enumerate(rows) if row["task"] == "sparse_solution")
    env = {**os.environ, "PRINT_COMMAND_ONLY": "1", "DATA_ROOT": str(tmp_path / "PDEdata")}
    result = subprocess.run(
        ["bash", "scripts/experiments/05_run_one.sh", str(out / "matrices" / "sanity_main.jsonl"), str(index)],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    command = result.stdout.strip()
    assert "python -m baselines.run" in command
    assert "--experiment-mode paper" in command
    assert "--num-sensors" in command
    assert "--sensor-mode" in command
    assert "--noise-level" in command
    assert "--run-id" in command
    for token in ["--dry-run", "--synthetic-data", "--allow-synthetic-fallback", "--prefer-test"]:
        assert token not in command


def test_run_one_generates_load_full_trajectory_for_time_varying(tmp_path: Path):
    out = tmp_path / "large"
    subprocess.run(
        [
            sys.executable,
            "scripts/experiments/build_matrix.py",
            "--config",
            "configs/experiments/time_varying_sensor_ablation.yaml",
            "--output-root",
            str(out),
            "--matrix-name",
            "time_varying_sensor_ablation",
        ],
        cwd=ROOT,
        check=True,
    )
    env = {**os.environ, "PRINT_COMMAND_ONLY": "1", "DATA_ROOT": str(tmp_path / "PDEdata")}
    result = subprocess.run(
        ["bash", "scripts/experiments/05_run_one.sh", str(out / "matrices" / "time_varying_sensor_ablation.jsonl"), "0"],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    assert "--load-full-trajectory" in result.stdout
    assert "--sensor-mode time_varying" in result.stdout


def test_2gpu_parallel_launcher_uses_row_override_and_existing_runner():
    text = (ROOT / "scripts/experiments/08_run_matrix_2gpu_parallel.sh").read_text(encoding="utf-8")
    for token in [
        "CUDA_VISIBLE_DEVICES",
        "ALLOW_ROW_OVERRIDE=1",
        "NUM_WORKERS",
        "PREFETCH_FACTOR",
        "05_run_one.sh",
        "xargs",
        "JOBS_PER_GPU",
    ]:
        assert token in text
