from __future__ import annotations

from baselines.common.data_adapter import build_default_registry
from baselines.experiment_matrix import compatibility_reason
from baselines.methods.var4d import Var4DBaseline
from baselines.run import build_data_spec


def test_var4d_full_trajectory_mode_on_shallow_water():
    registry = build_default_registry()
    raw = registry.synthetic_raw("shallow_water", n=1, resolution=8)
    batch = registry.make_task(raw, "shallow_water", "sparse_solution", num_sensors=4, seed=1)
    model = Var4DBaseline().build({"steps": 0}, build_data_spec(batch))
    model.predict(batch)
    assert batch.metadata["assimilation_mode"] == "full_trajectory"


def test_var4d_two_level_surrogate_on_heat_endpoint_only():
    registry = build_default_registry()
    raw = registry.synthetic_raw("heat", n=1, resolution=8)
    batch = registry.make_task(raw, "heat", "sparse_solution", num_sensors=4, seed=1)
    model = Var4DBaseline().build({"steps": 0}, build_data_spec(batch))
    model.predict(batch)
    assert batch.metadata["assimilation_mode"] == "two_level_surrogate"


def test_static_pde_var4d_skipped_by_matrix():
    assert "time-varying" in compatibility_reason("var4d", "poisson", "sparse_solution")
