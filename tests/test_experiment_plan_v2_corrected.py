from __future__ import annotations

from pathlib import Path

import yaml


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
