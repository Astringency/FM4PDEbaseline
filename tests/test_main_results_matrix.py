from __future__ import annotations

from pathlib import Path

from scripts.experiments.build_matrix import build_matrix, load_config


ROOT = Path(__file__).resolve().parents[1]
MATRIX_ENV = ["SEEDS", "SENSOR_COUNTS", "SENSOR_MODES", "NOISE_LEVELS", "TRAIN_SIZES", "FULL_ABLATION_ALL"]


def test_main_results_counts_skips_and_unique_run_ids(tmp_path: Path, monkeypatch):
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config(ROOT / "configs" / "experiments" / "main_results.yaml")
    rows, skipped, summary = build_matrix(cfg, tmp_path / "main_results", "main_results")

    assert summary["experiment_kind"] == "main"
    assert summary["ablation_factor"] == ""
    assert summary["run_count"] == 387
    assert summary["by_task_group"] == {
        "full_forward_main": 99,
        "full_inverse_main": 99,
        "sparse_inverse_main": 24,
        "sparse_solution_main_amortized": 99,
        "sparse_solution_main_physics": 66,
    }
    run_ids = [row["run_id"] for row in rows]
    assert len(run_ids) == len(set(run_ids))
    assert len(skipped) == 14
    assert all(row["baseline"] in {"pinn_sparse", "pde_opt"} for row in skipped)
    assert all("time-dependent sparse inverse" in row["reason"] for row in skipped)


def test_main_results_sparse_inverse_uses_physics_baselines_only(tmp_path: Path, monkeypatch):
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config(ROOT / "configs" / "experiments" / "main_results.yaml")
    rows, _skipped, _summary = build_matrix(cfg, tmp_path / "main_results", "main_results")
    sparse_inverse = [row for row in rows if row["task_group"] == "sparse_inverse_main"]
    assert {row["baseline"] for row in sparse_inverse} == {"pinn_sparse", "pde_opt"}
    assert {row["pde"] for row in sparse_inverse} == {"poisson", "helmholtz", "darcy", "steady_heat_conduction"}
