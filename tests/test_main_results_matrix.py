from __future__ import annotations

from pathlib import Path

from scripts.experiments.build_matrix import build_matrix, load_config


ROOT = Path(__file__).resolve().parents[1]
MATRIX_ENV = [
    "SEEDS",
    "SENSOR_COUNTS",
    "SENSOR_MODES",
    "NOISE_LEVELS",
    "TRAIN_SIZES",
    "FULL_ABLATION_ALL",
    "BASELINES",
]


def test_main_results_counts_skips_and_unique_run_ids(tmp_path: Path, monkeypatch):
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config(ROOT / "configs" / "experiments" / "main_results.yaml")
    rows, skipped, summary = build_matrix(cfg, tmp_path / "main_results", "main_results")

    assert summary["experiment_kind"] == "main"
    assert summary["ablation_factor"] == ""
    assert summary["run_count"] == 80
    assert summary["by_task_group"] == {
        "full_forward_main": 12,
        "full_inverse_main": 4,
        "sparse_inverse_main_amortized": 12,
        "sparse_inverse_main": 9,
        "sparse_solution_main_amortized": 15,
        "sparse_solution_burger_time_slices": 3,
        "time_varying_da_main": 2,
        "time_varying_da_burger_time_slices": 2,
        "sparse_forward_main_amortized": 12,
        "sparse_forward_main_physics": 9,
    }
    run_ids = [row["run_id"] for row in rows]
    assert len(run_ids) == len(set(run_ids))
    assert {row["val_size"] for row in rows} == {5000}
    assert {row["data_loading_mode"] for row in rows} == {"eager"}
    assert {row["num_workers"] for row in rows} == {4}
    assert {row["pin_memory"] for row in rows} == {True}
    assert {row["persistent_workers"] for row in rows} == {True}
    assert {row["prefetch_factor"] for row in rows} == {2}
    assert all(row.get("data_files") for row in rows)
    poisson = next(row for row in rows if row["pde"] == "poisson")
    helmholtz = next(row for row in rows if row["pde"] == "helmholtz")
    assert poisson["data_files"]["test"] == [
        "poisson/poisson_test_10000-128-128.mat"
    ]
    assert helmholtz["data_files"]["test"] == [
        "helmholtz/helmholtz_test_10000-128-128.mat"
    ]
    assert {row["epochs"] for row in rows if row["baseline"] == "ifno"} == {500}
    assert set(summary["by_task_group"]) == {
        "full_forward_main",
        "full_inverse_main",
        "sparse_inverse_main_amortized",
        "sparse_solution_main_amortized",
        "sparse_solution_burger_time_slices",
        "time_varying_da_main",
        "time_varying_da_burger_time_slices",
        "sparse_forward_main_amortized",
        "sparse_forward_main_physics",
        "sparse_inverse_main",
    }
    assert skipped == []
    inverse_rows = [row for row in rows if row["task_group"] == "full_inverse_main"]
    assert {row["baseline"] for row in inverse_rows} == {"ifno"}
    assert {row["capability_status"] for row in inverse_rows} == {"adapted"}
    assert not any(
        row["pde"] == "nsnonbounded" and row["baseline"] in {"pinn_sparse", "pde_opt", "pc_bnn"}
        for row in rows
    )


def test_main_results_can_build_an_isolated_ifno_recovery_matrix(tmp_path: Path, monkeypatch):
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BASELINES", "ifno")
    cfg = load_config(ROOT / "configs" / "experiments" / "main_results.yaml")

    rows, skipped, summary = build_matrix(cfg, tmp_path / "ifno_recovery", "ifno_recovery")

    assert len(rows) == 8
    assert {row["baseline"] for row in rows} == {"ifno"}
    assert {row["task_group"] for row in rows} == {"full_forward_main", "full_inverse_main"}
    assert summary["run_count"] == 8
    assert skipped == []


def test_main_results_sparse_inverse_uses_all_requested_baselines(tmp_path: Path, monkeypatch):
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config(ROOT / "configs" / "experiments" / "main_results.yaml")
    rows, _skipped, _summary = build_matrix(cfg, tmp_path / "main_results", "main_results")
    sparse_inverse = [row for row in rows if row["task"] == "sparse_inverse"]
    assert {row["baseline"] for row in sparse_inverse} == {
        "recfno", "senseiver", "voronoicnn", "pinn_sparse", "pde_opt", "pc_bnn"
    }
    assert {row["pde"] for row in sparse_inverse} == {"poisson", "helmholtz", "darcy", "nsnonbounded"}
    assert {
        row["baseline"] for row in sparse_inverse if row["pde"] == "nsnonbounded"
    } == {"recfno", "senseiver", "voronoicnn"}


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


def test_main_results_assigns_var4d_and_vivid_only_to_burger_sparse_reconstruction(tmp_path: Path, monkeypatch):
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config(ROOT / "configs" / "experiments" / "main_results.yaml")
    rows, _skipped, _summary = build_matrix(cfg, tmp_path / "main_results", "main_results")
    da_rows = [row for row in rows if row["baseline"] in {"var4d", "vivid"}]

    assert len(da_rows) == 4
    assert {row["pde"] for row in da_rows} == {"burger"}
    assert {row["task"] for row in da_rows} == {"sparse_solution"}
    assert {
        (row["num_sensors"], row["sensor_mode"], row["sensor_budget_mode"])
        for row in da_rows
    } == {
        (500, "random_per_sample", "total"),
        (5, "time_slices_per_sample", "total"),
    }
    assert {row["load_full_trajectory"] for row in da_rows} == {True}
    assert {row["capability_status"] for row in da_rows} == {"native", "official_adapter"}
    assert {row["implementation_required"] for row in da_rows} == {"canonical_math", "official"}
    assert {row["paper_table_eligible"] for row in da_rows} == {True}
    assert {row["official_native_eligible"] for row in da_rows} == {False}
    assert not any(row["pde"] == "nsnonbounded" for row in da_rows)


def test_main_results_includes_disclosed_scalar_pcbnn_adaptation(tmp_path: Path, monkeypatch):
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config(ROOT / "configs" / "experiments" / "main_results.yaml")
    rows, skipped, _summary = build_matrix(cfg, tmp_path / "main_results", "main_results")
    pcbnn_rows = [row for row in rows if row["baseline"] == "pc_bnn"]
    assert {row["pde"] for row in pcbnn_rows} == {"darcy", "poisson", "helmholtz"}
    assert {row["task"] for row in pcbnn_rows} == {"sparse_forward", "sparse_inverse"}
    assert all(row["implementation_required"] == "adapted_allowed" for row in pcbnn_rows)
    assert not [row for row in skipped if row["baseline"] == "pc_bnn"]
