from __future__ import annotations

from pathlib import Path

from scripts.experiments.build_matrix import build_matrix, load_config


ROOT = Path(__file__).resolve().parents[1]
MATRIX_ENV = ["SEEDS", "SENSOR_COUNTS", "SENSOR_MODES", "NOISE_LEVELS", "TRAIN_SIZES", "FULL_ABLATION_ALL"]


def _rows(name: str, tmp_path: Path, monkeypatch) -> list[dict]:
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config(ROOT / "configs" / "experiments" / f"{name}.yaml")
    rows, _skipped, _summary = build_matrix(cfg, tmp_path / name, name)
    return rows


def test_main_results_sparse_tasks_use_one_standard_sparse_setting(tmp_path: Path, monkeypatch):
    rows = _rows("main_results", tmp_path, monkeypatch)
    sparse_rows = [row for row in rows if str(row["task"]).startswith("sparse")]
    standard_sparse = [row for row in sparse_rows if row["task_group"] != "time_varying_da_main"]
    time_varying = [row for row in sparse_rows if row["task_group"] == "time_varying_da_main"]
    assert sparse_rows
    assert {row["num_sensors"] for row in sparse_rows} == {500}
    assert {row["sensor_mode"] for row in standard_sparse} == {"random_per_sample"}
    assert {row["sensor_mode"] for row in time_varying} == {"time_varying"}
    assert {row["noise_level"] for row in sparse_rows} == {0.0}


def test_main_results_are_not_sensor_noise_mode_full_grid(tmp_path: Path, monkeypatch):
    rows = _rows("main_results", tmp_path, monkeypatch)
    by_run_family: dict[tuple, set[tuple]] = {}
    for row in rows:
        key = (row["task_group"], row["pde"], row["baseline"], row["seed"])
        setting = (row["num_sensors"], row["sensor_mode"], row["noise_level"])
        by_run_family.setdefault(key, set()).add(setting)
    assert by_run_family
    assert all(len(settings) == 1 for settings in by_run_family.values())


def test_main_and_ablation_outputs_are_separated(tmp_path: Path, monkeypatch):
    main_rows = _rows("main_results", tmp_path, monkeypatch)
    ablation_rows = _rows("sensor_count_ablation", tmp_path, monkeypatch)
    assert all("/runs/main_results/" in row["output_dir"] for row in main_rows)
    assert all("/runs/ablations/ablation=sensor_count/" in row["output_dir"] for row in ablation_rows)
