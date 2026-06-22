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
    assert summary["run_count"] == 360
    assert summary["by_task_group"] == {
        "full_forward_main": 99,
        "full_inverse_main": 33,
        "sparse_inverse_main": 24,
        "sparse_solution_main_amortized": 99,
        "sparse_solution_main_physics": 69,
        "time_varying_da_main": 36,
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
        "sparse_solution_main_physics",
        "sparse_inverse_main",
        "time_varying_da_main",
    }
    assert len(skipped) == 46
    adapted_skips = [row for row in skipped if row["task_group"] == "full_inverse_main"]
    assert {row["baseline"] for row in adapted_skips} == {"fno", "deeponet"}
    assert all("supplement-only" in row["reason"] for row in adapted_skips)
    unsupported_skips = [row for row in skipped if row["task_group"] == "sparse_inverse_main"]
    assert all(row["baseline"] in {"pinn_sparse", "pde_opt"} for row in unsupported_skips)
    assert all("time-dependent sparse inverse" in row["reason"] for row in unsupported_skips)
    pcbnn_skips = [row for row in skipped if row["task_group"] == "sparse_solution_main_physics" and row["baseline"] == "pc_bnn"]
    assert len(pcbnn_skips) == 10
    assert all("PC-BNN" in row["reason"] for row in pcbnn_skips)


def test_main_results_sparse_inverse_uses_physics_baselines_only(tmp_path: Path, monkeypatch):
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config(ROOT / "configs" / "experiments" / "main_results.yaml")
    rows, _skipped, _summary = build_matrix(cfg, tmp_path / "main_results", "main_results")
    sparse_inverse = [row for row in rows if row["task_group"] == "sparse_inverse_main"]
    assert {row["baseline"] for row in sparse_inverse} == {"pinn_sparse", "pde_opt"}
    assert {row["pde"] for row in sparse_inverse} == {"poisson", "helmholtz", "darcy", "steady_heat_conduction"}


def test_main_results_includes_time_varying_da_main(tmp_path: Path, monkeypatch):
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config(ROOT / "configs" / "experiments" / "main_results.yaml")
    rows, _skipped, _summary = build_matrix(cfg, tmp_path / "main_results", "main_results")
    tv_rows = [row for row in rows if row["task_group"] == "time_varying_da_main"]
    assert tv_rows
    assert {row["baseline"] for row in tv_rows} == {"senseiver", "var4d", "vivid"}
    assert {row["pde"] for row in tv_rows} == {"nsnonbounded", "burger", "reaction_diffusion", "shallow_water"}
    assert {row["task"] for row in tv_rows} == {"sparse_solution"}
    assert {row["sensor_mode"] for row in tv_rows} == {"time_varying"}
    assert {row["load_full_trajectory"] for row in tv_rows} == {True}


def test_main_results_includes_only_matched_pcbnn_sparse_reconstruction(tmp_path: Path, monkeypatch):
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config(ROOT / "configs" / "experiments" / "main_results.yaml")
    rows, skipped, _summary = build_matrix(cfg, tmp_path / "main_results", "main_results")
    pcbnn_rows = [row for row in rows if row["task_group"] == "sparse_solution_main_physics" and row["baseline"] == "pc_bnn"]
    assert pcbnn_rows
    assert {row["pde"] for row in pcbnn_rows} == {"shallow_water"}
    assert {row["task"] for row in pcbnn_rows} == {"sparse_solution"}
    skipped_pdes = {
        row["pde"]
        for row in skipped
        if row["task_group"] == "sparse_solution_main_physics" and row["baseline"] == "pc_bnn"
    }
    assert "shallow_water" not in skipped_pdes
    assert {"darcy", "poisson", "helmholtz"}.issubset(skipped_pdes)
