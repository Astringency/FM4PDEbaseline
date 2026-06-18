from __future__ import annotations

from baselines.aggregate_results import aggregate_rows


def _raw_row(steps: int) -> dict:
    return {
        "experiment_kind": "ablation",
        "ablation_factor": "runtime_budget",
        "task_group": "runtime_budget_ablation",
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
