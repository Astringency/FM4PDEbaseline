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


def test_00_build_matrices_does_not_default_to_full_all():
    text = (ROOT / "scripts" / "experiments" / "00_build_matrices.sh").read_text(encoding="utf-8")
    assert "full_all" not in text
    assert "core" not in text
    assert "time_varying)" not in text
    assert "main_results" in text
    assert "sensor_count_ablation" in text


def test_main_results_has_no_sparse_factor_cartesian_product(tmp_path: Path, monkeypatch):
    rows = _rows("main_results", tmp_path, monkeypatch)
    sparse_rows = [row for row in rows if row["task"].startswith("sparse")]
    assert {row["num_sensors"] for row in sparse_rows} == {500}
    assert {row["sensor_mode"] for row in sparse_rows} == {"random"}
    assert {row["noise_level"] for row in sparse_rows} == {0.0}
    expected_families = {(row["task_group"], row["pde"], row["baseline"], row["seed"]) for row in sparse_rows}
    assert len(sparse_rows) == len(expected_families)
