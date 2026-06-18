from __future__ import annotations

import json

from baselines.aggregate_results import aggregate_rows


def _row(assimilation_mode_counts: dict[str, int]) -> dict:
    return {
        "experiment_kind": "main",
        "ablation_factor": "",
        "task_group": "sparse_solution_main_physics",
        "pde": "heat",
        "task": "sparse_solution",
        "baseline": "var4d",
        "train_size": 50000,
        "scalar_param_mode": "metadata",
        "num_sensors": 500,
        "sensor_mode": "random",
        "noise_level": 0.0,
        "steps": 500,
        "refine_steps": 0,
        "particles": 0,
        "method_budget_label": "steps=500",
        "backend_used": "local",
        "relative_l2_solution": 1.0,
        "sample_count": sum(assimilation_mode_counts.values()),
        "assimilation_mode_counts": json.dumps(assimilation_mode_counts, sort_keys=True),
    }


def test_aggregate_results_combines_assimilation_mode_counts():
    summary = aggregate_rows([_row({"filter": 2}), _row({"filter": 1, "smoother": 3})])[0]
    assert json.loads(summary["assimilation_mode_counts"]) == {"filter": 3, "smoother": 3}
