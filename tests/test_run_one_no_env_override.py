from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from baselines.run import parse_args as parse_baseline_args


ROOT = Path(__file__).resolve().parents[1]


def _row(tmp_path: Path) -> dict:
    return {
        "run_id": "run_one_override_regression",
        "run_name": "run_one_override_regression",
        "experiment_kind": "ablation",
        "ablation_factor": "runtime_budget",
        "task_group": "runtime_budget_ablation",
        "task": "sparse_solution",
        "pde": "poisson",
        "baseline": "pinn_sparse",
        "seed": 1,
        "train_size": 500,
        "val_size": 0,
        "test_size": 10,
        "train_shards": 1,
        "num_sensors": 500,
        "sensor_mode": "random",
        "noise_level": 0.05,
        "scalar_param_mode": "metadata",
        "data_loading_mode": "lazy",
        "load_full_trajectory": False,
        "batch_size": 1,
        "epochs": 1,
        "steps": 50,
        "refine_steps": 0,
        "particles": 0,
        "device": "cpu",
        "config": "baselines/configs/paper.yaml",
        "output_dir": str(tmp_path / "run"),
        "log_dir": str(tmp_path / "log"),
        "status_file": str(tmp_path / "run" / "run.status.json"),
        "skip_reason": "",
    }


def _printed_args(tmp_path: Path, env_updates: dict[str, str]):
    matrix = tmp_path / "matrix.jsonl"
    matrix.write_text(json.dumps(_row(tmp_path), sort_keys=True) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": sys.executable,
            "DATA_ROOT": "/tmp/fake",
            "PRINT_COMMAND_ONLY": "1",
            "TRAIN_SIZE": "50000",
            "NOISE_LEVELS": "0.0",
            "PINN_STEPS": "1000",
        }
    )
    env.update(env_updates)
    proc = subprocess.run(
        [sys.executable, "scripts/experiments/run_one.py", str(matrix), "0"],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    cmd = shlex.split(proc.stdout.strip())
    assert cmd[:3] == [sys.executable, "-m", "baselines.run"]
    return parse_baseline_args(cmd[3:])


def test_run_one_uses_matrix_row_by_default_even_when_env_has_design_values(tmp_path: Path):
    args = _printed_args(tmp_path, {"ALLOW_ROW_OVERRIDE": ""})
    assert args.train_size == 500
    assert args.noise_level == 0.05
    assert args.steps == 50
    assert args.experiment_kind == "ablation"
    assert args.ablation_factor == "runtime_budget"
    assert args.task_group == "runtime_budget_ablation"


def test_run_one_allows_design_env_override_only_when_enabled(tmp_path: Path):
    args = _printed_args(tmp_path, {"ALLOW_ROW_OVERRIDE": "1", "NOISE_LEVEL": "0.0"})
    assert args.train_size == 50000
    assert args.noise_level == 0.0
    assert args.steps == 1000
