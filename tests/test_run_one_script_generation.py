from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.experiments.run_one import build_command


ROOT = Path(__file__).resolve().parents[1]


def test_shell_launcher_default_data_root_uses_expandable_home():
    text = (ROOT / "scripts/run_baseline.sh").read_text(encoding="utf-8")
    assert '${DATA_ROOT:-${HOME}/share/PDEdata}' in text


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _command_generation_config(tmp_path: Path, source: str) -> Path:
    """Use a non-formal protocol for tests that only inspect CLI generation."""
    config = yaml.safe_load((ROOT / source).read_text(encoding="utf-8"))
    config["task_protocol_version"] = "command-generation-test-v1"
    target = tmp_path / Path(source).name
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def test_run_one_generates_paper_sparse_command_without_debug_flags(tmp_path: Path):
    out = tmp_path / "large"
    config = _command_generation_config(tmp_path, "configs/experiments/sanity_main.yaml")
    subprocess.run(
        [
            sys.executable,
            "scripts/experiments/build_matrix.py",
            "--config",
            str(config),
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
        [sys.executable, "scripts/experiments/run_one.py", str(out / "matrices" / "sanity_main.jsonl"), str(index)],
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
    config = _command_generation_config(
        tmp_path,
        "configs/experiments/time_varying_sensor_ablation.yaml",
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/experiments/build_matrix.py",
            "--config",
            str(config),
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
        [sys.executable, "scripts/experiments/run_one.py", str(out / "matrices" / "time_varying_sensor_ablation.jsonl"), "0"],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    assert "--load-full-trajectory" in result.stdout
    assert "--sensor-mode time_varying" in result.stdout


def test_2gpu_parallel_launcher_preserves_fingerprinted_row_and_uses_existing_runner():
    text = (ROOT / "scripts/run_experiments.py").read_text(encoding="utf-8")
    for token in [
        "CUDA_VISIBLE_DEVICES",
        'env.pop("ALLOW_ROW_OVERRIDE", None)',
        "run_one.py",
        "ThreadPoolExecutor",
        "jobs-per-gpu",
    ]:
        assert token in text
    for forbidden in ["ALLOW_ROW_OVERRIDE=1", "NUM_WORKERS_PER_RUN"]:
        assert forbidden not in text


def test_save_checkpoint_zero_disables_checkpoint_flag(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "PDEdata"))
    monkeypatch.setenv("SAVE_CHECKPOINT", "0")

    cmd = build_command(_row(tmp_path, "fno"))

    assert "--save-checkpoint" not in cmd
    assert "--no-save-checkpoint" in cmd


def test_save_checkpoint_one_enables_checkpoint_flag(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "PDEdata"))
    monkeypatch.setenv("SAVE_CHECKPOINT", "1")

    cmd = build_command(_row(tmp_path, "pde_opt"))

    assert "--save-checkpoint" in cmd


def test_save_checkpoint_amortized_enables_neural_baselines(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "PDEdata"))
    monkeypatch.setenv("SAVE_CHECKPOINT", "amortized")

    for baseline in ["fno", "deeponet", "ifno"]:
        cmd = build_command(_row(tmp_path, baseline))
        assert "--save-checkpoint" in cmd


def test_save_checkpoint_amortized_skips_per_instance_baselines(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "PDEdata"))
    monkeypatch.setenv("SAVE_CHECKPOINT", "amortized")

    for baseline in ["pde_opt", "pinn_sparse"]:
        cmd = build_command(_row(tmp_path, baseline))
        assert "--save-checkpoint" not in cmd
        assert "--no-save-checkpoint" in cmd


def _row(tmp_path: Path, baseline: str) -> dict:
    return {
        "baseline": baseline,
        "pde": "poisson",
        "task": "forward",
        "config": "baselines/configs/paper.yaml",
        "train_size": 4,
        "val_size": 2,
        "test_size": 2,
        "train_shards": 1,
        "batch_size": 2,
        "epochs": 1,
        "seed": 1,
        "device": "cpu",
        "data_loading_mode": "eager",
        "num_workers": 0,
        "prefetch_factor": 2,
        "scalar_param_mode": "metadata",
        "output_dir": str(tmp_path / baseline),
        "experiment_kind": "test",
        "ablation_factor": "",
        "task_group": "main",
        "run_id": f"{baseline}_poisson_forward_seed1",
        "run_name": f"{baseline} poisson forward seed1",
        "pin_memory": False,
        "persistent_workers": False,
        "steps": 0,
        "refine_steps": 0,
        "particles": 0,
        "num_sensors": 50,
        "sensor_mode": "random",
        "noise_level": 0.0,
    }
