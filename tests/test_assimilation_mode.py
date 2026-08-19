from __future__ import annotations

import torch
import pytest

from baselines.common.data_adapter import build_default_registry
from baselines.common.metrics import physics_loss_metric
from baselines.experiment_matrix import compatibility_reason
from baselines.methods.var4d import Var4DBaseline, _initial_trajectory
from baselines.run import build_data_spec


def test_var4d_burgers_full_trajectory_mode_and_shape():
    registry = build_default_registry()
    raw = registry.synthetic_raw("burger", n=1, resolution=8)
    batch = registry.make_task(
        raw,
        "burger",
        "sparse_solution",
        num_sensors=4,
        sensor_mode="random_per_sample",
        sensor_budget_mode="total",
        experiment_mode="debug",
    )
    model = Var4DBaseline().build({"steps": 0}, build_data_spec(batch))
    pred = model.predict(batch)
    losses = physics_loss_metric(pred, "burger", batch.metadata)
    assert tuple(pred.shape) == tuple(batch.target_fields.shape)
    assert batch.metadata["assimilation_mode"] == "full_trajectory"
    assert losses["mode"] == "full_trajectory"


def test_var4d_burgers_initialization_uses_only_sparse_observation_background():
    registry = build_default_registry()
    raw = registry.synthetic_raw("burger", n=1, resolution=8)
    batch = registry.make_task(
        raw,
        "burger",
        "sparse_solution",
        num_sensors=4,
        sensor_mode="random_per_sample",
        sensor_budget_mode="total",
        experiment_mode="debug",
        build_voronoi_grid=True,
    )

    state_before, _view, background_before, dyn_meta = _initial_trajectory(batch)
    batch.full_tensor.fill_(1234.0)
    batch.metadata["initial_1d"] = torch.full(
        (batch.target_fields.shape[0], batch.target_fields.shape[-1]),
        -4321.0,
        dtype=batch.target_fields.dtype,
    )
    batch.metadata["background_fields"] = torch.full_like(batch.metadata["background_fields"], 999.0)
    state_after, _view, background_after, dyn_meta_after = _initial_trajectory(batch)

    assert torch.equal(state_before, batch.metadata["voronoi_grid"])
    assert torch.equal(state_after, state_before)
    assert torch.equal(background_after, background_before)
    assert dyn_meta["assimilation_background_source"] == "voronoi_grid_from_sparse_observations"
    assert dyn_meta["assimilation_uses_hidden_truth"] is False
    assert torch.equal(dyn_meta["initial_1d"], state_before[:, 0, 0, :])
    assert torch.equal(dyn_meta_after["initial_1d"], state_before[:, 0, 0, :])
    for private_key in {"full_tensor", "full_trajectory", "original_input_fields", "background_fields"}:
        assert private_key not in dyn_meta


def test_static_pde_var4d_skipped_by_matrix():
    assert "time-varying" in compatibility_reason("var4d", "poisson", "sparse_solution")


def test_var4d_rejects_removed_non_burgers_entry():
    registry = build_default_registry()
    raw = registry.synthetic_raw("shallow_water", n=1, resolution=8)
    batch = registry.make_task(raw, "shallow_water", "sparse_solution", num_sensors=4)
    with pytest.raises(ValueError, match="only to Burgers"):
        Var4DBaseline().build({"steps": 0}, build_data_spec(batch))
