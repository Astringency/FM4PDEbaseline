from __future__ import annotations

from baselines.aggregate_results import aggregate_rows


def _raw_row(steps: int, *, experiment_kind: str = "ablation", ablation_factor: str = "runtime_budget") -> dict:
    return {
        "experiment_kind": experiment_kind,
        "ablation_factor": ablation_factor,
        "task_group": f"{ablation_factor}_ablation" if experiment_kind == "ablation" else "sparse_solution_main_physics",
        "pde": "poisson",
        "task": "sparse_solution",
        "baseline": "pde_opt",
        "train_size": 50000,
        "scalar_param_mode": "metadata",
        "num_sensors": 500,
        "sensor_mode": "random",
        "noise_level": 0.0,
        "steps": steps,
        "refine_steps": 0,
        "particles": 0,
        "method_budget_label": f"steps={steps}",
        "backend_used": "local",
        "relative_l2_solution": 1.0,
        "sample_count": 1,
    }


def test_runtime_budget_steps_are_aggregation_keys():
    rows = [_raw_row(50), _raw_row(100)]
    summary = aggregate_rows(rows)
    assert len(summary) == 2
    assert {row["steps"] for row in summary} == {50, 100}
    assert {row["method_budget_label"] for row in summary} == {"steps=50", "steps=100"}
    assert {row["grouping_budget_mode"] for row in summary} == {"budget_grouped"}
    assert {row["budget_variation_warning"] for row in summary} == {""}


def test_main_results_budget_fields_are_recorded_not_grouped():
    rows = [
        _raw_row(50, experiment_kind="main", ablation_factor="none"),
        _raw_row(100, experiment_kind="main", ablation_factor="none"),
    ]
    summary = aggregate_rows(rows)
    assert len(summary) == 1
    assert summary[0]["steps"] == 50
    assert summary[0]["method_budget_label"] == "steps=50"
    assert summary[0]["grouping_budget_mode"] == "budget_recorded_only"
    assert summary[0]["budget_variation_warning"] == "budget fields vary within this non-runtime group"
    assert summary[0]["run_count"] == 2


def test_main_results_same_budget_has_no_variation_warning():
    rows = [
        _raw_row(50, experiment_kind="main", ablation_factor="none"),
        _raw_row(50, experiment_kind="main", ablation_factor="none"),
    ]
    summary = aggregate_rows(rows)
    assert len(summary) == 1
    assert summary[0]["grouping_budget_mode"] == "budget_recorded_only"
    assert summary[0]["budget_variation_warning"] == ""
    assert summary[0]["run_count"] == 2


def test_non_runtime_ablations_do_not_group_by_steps():
    for ablation_factor in ("noise", "sensor_count"):
        rows = [_raw_row(50, ablation_factor=ablation_factor), _raw_row(100, ablation_factor=ablation_factor)]
        summary = aggregate_rows(rows)
        assert len(summary) == 1
        assert summary[0]["grouping_budget_mode"] == "budget_recorded_only"
        assert summary[0]["budget_variation_warning"] == "budget fields vary within this non-runtime group"
        assert summary[0]["run_count"] == 2
