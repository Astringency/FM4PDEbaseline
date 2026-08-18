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
    assert summary["run_count"] == 228
    assert summary["by_task_group"] == {
        "full_forward_main": 36,
        "full_inverse_main": 12,
        "sparse_inverse_main": 63,
        "sparse_solution_main_amortized": 45,
        "sparse_solution_burger_time_slices": 9,
        "sparse_forward_main_amortized": 36,
        "sparse_forward_main_physics": 27,
    }
    run_ids = [row["run_id"] for row in rows]
    assert len(run_ids) == len(set(run_ids))
    assert {row["val_size"] for row in rows} == {1000}
    assert {row["data_loading_mode"] for row in rows} == {"eager"}
    assert {row["num_workers"] for row in rows} == {4}
    assert {row["pin_memory"] for row in rows} == {True}
    assert {row["persistent_workers"] for row in rows} == {True}
    assert {row["prefetch_factor"] for row in rows} == {2}
    assert set(summary["by_task_group"]) == {
        "full_forward_main",
        "full_inverse_main",
        "sparse_solution_main_amortized",
        "sparse_solution_burger_time_slices",
        "sparse_forward_main_amortized",
        "sparse_forward_main_physics",
        "sparse_inverse_main",
    }
    assert len(skipped) == 6
    inverse_rows = [row for row in rows if row["task_group"] == "full_inverse_main"]
    assert {row["baseline"] for row in inverse_rows} == {"ifno"}
    assert {row["capability_status"] for row in inverse_rows} == {"adapted"}
    unsupported_skips = [row for row in skipped if row["task_group"] == "sparse_inverse_main"]
    assert all(row["baseline"] in {"pinn_sparse", "pde_opt", "pc_bnn"} for row in unsupported_skips)
    assert all(
        "time-dependent sparse inverse" in row["reason"] or "Burger sparse_inverse is excluded" in row["reason"]
        or "enabled only for static PDEs" in row["reason"]
        for row in unsupported_skips
    )


def test_main_results_sparse_inverse_uses_all_requested_baselines(tmp_path: Path, monkeypatch):
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config(ROOT / "configs" / "experiments" / "main_results.yaml")
    rows, _skipped, _summary = build_matrix(cfg, tmp_path / "main_results", "main_results")
    sparse_inverse = [row for row in rows if row["task_group"] == "sparse_inverse_main"]
    assert {row["baseline"] for row in sparse_inverse} == {
        "recfno", "senseiver", "voronoicnn", "pinn_sparse", "pde_opt", "pc_bnn"
    }
    assert {row["pde"] for row in sparse_inverse} == {"poisson", "helmholtz", "darcy", "nsnonbounded"}


def test_main_results_includes_burger_complete_time_slice_mode(tmp_path: Path, monkeypatch):
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config(ROOT / "configs" / "experiments" / "main_results.yaml")
    rows, _skipped, _summary = build_matrix(cfg, tmp_path / "main_results", "main_results")
    tv_rows = [row for row in rows if row["task_group"] == "sparse_solution_burger_time_slices"]
    assert tv_rows
    assert {row["baseline"] for row in tv_rows} == {"recfno", "senseiver", "voronoicnn"}
    assert {row["pde"] for row in tv_rows} == {"burger"}
    assert {row["task"] for row in tv_rows} == {"sparse_solution"}
    assert {row["sensor_mode"] for row in tv_rows} == {"time_slices_per_sample"}
    assert {row["num_sensors"] for row in tv_rows} == {5}
    assert {row["load_full_trajectory"] for row in tv_rows} == {True}


def test_main_results_includes_scalar_pcbnn_sparse_forward_and_inverse(tmp_path: Path, monkeypatch):
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config(ROOT / "configs" / "experiments" / "main_results.yaml")
    rows, skipped, _summary = build_matrix(cfg, tmp_path / "main_results", "main_results")
    pcbnn_rows = [row for row in rows if row["baseline"] == "pc_bnn"]
    assert pcbnn_rows
    assert {row["pde"] for row in pcbnn_rows} == {"darcy", "poisson", "helmholtz"}
    assert {row["task"] for row in pcbnn_rows} == {"sparse_forward", "sparse_inverse"}
    assert all(row["pde"] == "nsnonbounded" for row in skipped if row["baseline"] == "pc_bnn")
