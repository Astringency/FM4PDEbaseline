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
    assert corrected["sensor_protocol_version"] == "fm4pde-sensor-contract-v2"

    for name, group in corrected["task_group_overrides"].items():
        if str(group["task"]).startswith("sparse"):
            assert group["sensor_mode"] == "random_per_sample", name


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
    assert "burger" not in groups["sparse_forward_main_amortized"]["pdes"]
    assert "burger" not in groups["sparse_forward_main_physics"]["pdes"]


def test_corrected_plan_expands_the_full_three_seed_cohort(tmp_path: Path):
    corrected = _load(CORRECTED)

    assert corrected["seeds"] == [1, 2, 3]

    rows, skipped, summary = build_matrix(
        corrected,
        tmp_path / "experiment_plan_v2_corrected",
        "experiment_plan_v2_corrected",
    )

    assert len(rows) == summary["run_count"] == 213
    assert len(skipped) == summary["skipped_combo_count"] == 4
    assert summary["skipped_expanded_count"] == 12
    assert summary["by_task_group"] == {
        "full_forward_main": 45,
        "full_inverse_main": 15,
        "sparse_forward_main_amortized": 36,
        "sparse_forward_main_physics": 18,
        "sparse_inverse_main": 54,
        "sparse_solution_main_amortized": 45,
    }
