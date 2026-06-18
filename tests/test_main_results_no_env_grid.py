from __future__ import annotations

from pathlib import Path

from scripts.experiments.build_matrix import build_matrix, load_config


ROOT = Path(__file__).resolve().parents[1]
MATRIX_ENV = ["SEEDS", "SENSOR_COUNTS", "SENSOR_MODES", "NOISE_LEVELS", "TRAIN_SIZES", "FULL_ABLATION_ALL"]


def test_main_results_sparse_tasks_have_one_default_sensor_noise_setting(tmp_path: Path, monkeypatch):
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config(ROOT / "configs" / "experiments" / "main_results.yaml")
    rows, _skipped, _summary = build_matrix(cfg, tmp_path / "main_results", "main_results")
    sparse_rows = [row for row in rows if str(row["task"]).startswith("sparse")]

    assert {row["num_sensors"] for row in sparse_rows} == {500}
    assert {row["sensor_mode"] for row in sparse_rows} == {"random"}
    assert {row["noise_level"] for row in sparse_rows} == {0.0}

    settings_by_family = {}
    for row in sparse_rows:
        key = (row["baseline"], row["pde"], row["seed"], row["task_group"])
        settings_by_family.setdefault(key, set()).add((row["num_sensors"], row["sensor_mode"], row["noise_level"]))
    assert all(settings == {(500, "random", 0.0)} for settings in settings_by_family.values())
