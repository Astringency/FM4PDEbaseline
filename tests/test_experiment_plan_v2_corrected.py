from __future__ import annotations

from pathlib import Path

import yaml

from scripts.experiments.build_matrix import build_matrix


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "configs/experiments/experiment_plan_v2.yaml"
CORRECTED = ROOT / "configs/experiments/experiment_plan_v2_corrected.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_corrected_plan_uses_separate_namespace_and_explicit_sensor_protocol():
    original = _load(ORIGINAL)
    corrected = _load(CORRECTED)

    assert original["name"] == "experiment_plan_v2"
    assert corrected["name"] == "experiment_plan_v2_corrected"
    assert corrected["comparison_track"] == "unified_adapted"
    assert corrected["task_protocol_version"] == "fm4pde-task-contract-v3"
    assert corrected["sensor_protocol_version"] == "fm4pde-sensor-contract-v3"

    for name, group in corrected["task_group_overrides"].items():
        if str(group["task"]).startswith("sparse"):
            assert group["sensor_mode"] in {"random_per_sample", "time_slices_per_sample"}, name


def test_corrected_plan_excludes_invalid_sparse_rows():
    corrected = _load(CORRECTED)
    groups = corrected["task_group_overrides"]

    assert "sparse_solution_main_physics" not in corrected["task_groups"]
    assert "sparse_solution_main_physics" not in groups
    assert "burger" not in groups["sparse_inverse_main"]["pdes"]
    assert set(groups["sparse_solution_main_amortized"]["baselines"]) == {
        "recfno",
        "senseiver",
        "voronoicnn",
    }
    assert groups["sparse_solution_main_amortized"]["load_full_trajectory"] is True
    assert groups["sparse_solution_main_amortized"]["sensor_budget_mode"] == "total"
    assert groups["full_forward_main"]["pdes"] == ["poisson", "helmholtz", "darcy", "nsnonbounded"]
    assert groups["full_inverse_main"]["pdes"] == ["poisson", "helmholtz", "darcy", "nsnonbounded"]
    assert groups["sparse_solution_burger_time_slices"]["pdes"] == ["burger"]
    assert groups["sparse_solution_burger_time_slices"]["num_sensors"] == 5
    assert groups["sparse_solution_burger_time_slices"]["sensor_mode"] == "time_slices_per_sample"
    assert "burger" not in groups["sparse_forward_main_amortized"]["pdes"]
    assert "burger" not in groups["sparse_forward_main_physics"]["pdes"]
    assert "pc_bnn" in groups["sparse_inverse_main"]["baselines"]
    assert "pc_bnn" in groups["sparse_forward_main_physics"]["baselines"]


def test_corrected_plan_expands_the_full_three_seed_cohort(tmp_path: Path):
    corrected = _load(CORRECTED)

    assert corrected["seeds"] == [1, 2, 3]

    rows, skipped, summary = build_matrix(
        corrected,
        tmp_path / "experiment_plan_v2_corrected",
        "experiment_plan_v2_corrected",
    )

    assert len(rows) == summary["run_count"] == 228
    assert len(skipped) == summary["skipped_combo_count"] == 6
    assert summary["skipped_expanded_count"] == 18
    assert summary["by_task_group"] == {
        "full_forward_main": 36,
        "full_inverse_main": 12,
        "sparse_forward_main_amortized": 36,
        "sparse_forward_main_physics": 27,
        "sparse_inverse_main": 63,
        "sparse_solution_burger_time_slices": 9,
        "sparse_solution_main_amortized": 45,
    }
