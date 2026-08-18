from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from scripts.experiments.build_matrix import build_matrix, load_config


ROOT = Path(__file__).resolve().parents[1]
MATRIX_ENV = ["SEEDS", "SENSOR_COUNTS", "SENSOR_MODES", "NOISE_LEVELS", "TRAIN_SIZES", "FULL_ABLATION_ALL"]


def _matrix(name: str, tmp_path: Path, monkeypatch) -> tuple[list[dict], list[dict], dict]:
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config(ROOT / "configs" / "experiments" / f"{name}.yaml")
    return build_matrix(cfg, tmp_path / name, name)


def test_sensor_count_ablation_only_varies_num_sensors(tmp_path: Path, monkeypatch):
    rows, _skipped, summary = _matrix("sensor_count_ablation", tmp_path, monkeypatch)
    assert summary["experiment_kind"] == "ablation"
    assert summary["ablation_factor"] == "sensor_count"
    assert {row["num_sensors"] for row in rows} == {25, 50, 100, 250, 500, 1000}
    assert {row["sensor_mode"] for row in rows} == {"random_per_sample"}
    assert {row["noise_level"] for row in rows} == {0.0}


def test_noise_ablation_only_varies_noise_level(tmp_path: Path, monkeypatch):
    rows, _skipped, _summary = _matrix("noise_ablation", tmp_path, monkeypatch)
    assert {row["noise_level"] for row in rows} == {0.0, 0.01, 0.05, 0.10}
    assert {row["num_sensors"] for row in rows} == {500}
    assert {row["sensor_mode"] for row in rows} == {"random_per_sample"}


def test_sensor_mode_ablation_only_varies_sensor_mode(tmp_path: Path, monkeypatch):
    rows, _skipped, _summary = _matrix("sensor_mode_ablation", tmp_path, monkeypatch)
    assert {row["sensor_mode"] for row in rows} == {"random_per_sample", "fixed", "grid"}
    assert {row["num_sensors"] for row in rows} == {500}
    assert {row["noise_level"] for row in rows} == {0.0}


def test_train_size_ablation_only_varies_train_size(tmp_path: Path, monkeypatch):
    rows, _skipped, _summary = _matrix("train_size_ablation", tmp_path, monkeypatch)
    assert {row["train_size"] for row in rows} == {500, 1000, 5000, 10000, 50000}
    assert {row["num_sensors"] for row in rows} == {500}
    assert {row["sensor_mode"] for row in rows} == {"random_per_sample"}
    assert {row["noise_level"] for row in rows} == {0.0}


def test_time_varying_sensor_ablation_scope_and_full_trajectory(tmp_path: Path, monkeypatch):
    rows, _skipped, _summary = _matrix("time_varying_sensor_ablation", tmp_path, monkeypatch)
    assert {row["pde"] for row in rows} == {"nsnonbounded", "burger", "reaction_diffusion", "shallow_water"}
    assert {row["baseline"] for row in rows} == {"var4d", "vivid", "senseiver"}
    assert {row["sensor_mode"] for row in rows} == {"time_varying"}
    assert {row["load_full_trajectory"] for row in rows} == {True}
    assert {row["noise_level"] for row in rows} == {0.0}


def test_runtime_budget_ablation_varies_one_budget_per_baseline(tmp_path: Path, monkeypatch):
    rows, _skipped, _summary = _matrix("runtime_budget_ablation", tmp_path, monkeypatch)
    expected_steps = {50, 100, 250, 500, 1000}
    by_baseline: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_baseline[row["baseline"]].append(row)

    for baseline in {"pinn_sparse", "pc_bnn", "pde_opt", "var4d"}:
        baseline_rows = by_baseline[baseline]
        assert {row["steps"] for row in baseline_rows} == expected_steps
        assert {row["refine_steps"] for row in baseline_rows} == {0}
        if baseline == "pc_bnn":
            assert {row["particles"] for row in baseline_rows} == {8}
        else:
            assert {row["particles"] for row in baseline_rows} == {0}

    vivid_rows = by_baseline["vivid"]
    assert {row["refine_steps"] for row in vivid_rows} == {50, 100, 250, 500}
    assert {row["steps"] for row in vivid_rows} == {0}
    assert {row["particles"] for row in vivid_rows} == {0}

    for baseline, baseline_rows in by_baseline.items():
        expected = 4 if baseline == "vivid" else 5
        per_problem = defaultdict(set)
        for row in baseline_rows:
            per_problem[(row["pde"], row["seed"])].add((row["steps"], row["refine_steps"], row["particles"]))
        assert all(len(values) == expected for values in per_problem.values())
