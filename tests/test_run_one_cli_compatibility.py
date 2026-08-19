from __future__ import annotations

from pathlib import Path

from baselines.run import parse_args as parse_baseline_args
from scripts.experiments.build_matrix import build_matrix, load_config
from scripts.experiments.run_one import build_command


ROOT = Path(__file__).resolve().parents[1]
MATRIX_ENV = ["SEEDS", "SENSOR_COUNTS", "SENSOR_MODES", "NOISE_LEVELS", "TRAIN_SIZES", "FULL_ABLATION_ALL"]


def _rows(name: str, tmp_path: Path, monkeypatch) -> list[dict]:
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config(ROOT / "configs" / "experiments" / f"{name}.yaml")
    rows, _skipped, _summary = build_matrix(cfg, tmp_path / name, name)
    return rows


def _parse_command(row: dict, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "/tmp/PDEdata")
    # This file tests that run_one and baselines.run agree on CLI syntax.  A
    # formal v3 row intentionally requires a real, full data manifest; keep
    # that provenance contract covered by test_data_manifest_provenance.py.
    cli_only_row = dict(row)
    cli_only_row["run_fingerprint"] = ""
    cmd = build_command(cli_only_row)
    assert cmd[1:3] == ["-m", "baselines.run"]
    return cmd, parse_baseline_args(cmd[3:])


def test_run_one_command_is_accepted_by_baselines_run_parse_args(tmp_path: Path, monkeypatch):
    rows = _rows("main_results", tmp_path, monkeypatch)
    row = next(row for row in rows if row["task_group"] == "sparse_inverse_main" and row["baseline"] == "pde_opt")
    cmd, args = _parse_command(row, monkeypatch)
    assert "--run-id" in cmd
    assert "--run-name" in cmd
    assert "--steps" in cmd
    assert args.run_id == row["run_id"]
    assert args.run_name == row["run_name"]
    assert args.steps == row["steps"]


def test_run_one_budget_flags_are_cli_compatible(tmp_path: Path, monkeypatch):
    rows = _rows("runtime_budget_ablation", tmp_path, monkeypatch)
    examples = [
        next(row for row in rows if row["baseline"] == "pde_opt" and row["steps"] == 50),
        next(row for row in rows if row["baseline"] == "pc_bnn" and row["steps"] == 50),
        next(row for row in rows if row["baseline"] == "vivid" and row["refine_steps"] == 50),
    ]
    for row in examples:
        cmd, args = _parse_command(row, monkeypatch)
        assert args.run_id == row["run_id"]
        if row["steps"]:
            assert "--steps" in cmd
            assert args.steps == row["steps"]
        if row["refine_steps"]:
            assert "--refine-steps" in cmd
            assert args.refine_steps == row["refine_steps"]
        if row["particles"]:
            assert "--particles" in cmd
            assert args.particles == row["particles"]
