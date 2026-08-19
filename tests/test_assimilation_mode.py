from __future__ import annotations

import torch
import pytest

from baselines.common.data_adapter import build_default_registry
from baselines.common.metrics import physics_loss_metric
from baselines.experiment_matrix import compatibility_reason
from baselines.methods.var4d import Var4DBaseline, _initial_background, propagate_burgers_trajectory
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
    assert batch.metadata["assimilation_mode"] == "strong_constraint_full_window"
    assert batch.metadata["optimization_variable"] == "decorrelated_initial_state_increment"
    assert batch.metadata["optimization_optimizer"] == "scipy_L-BFGS-B"
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

    background_before = _initial_background(batch)
    batch.full_tensor.fill_(1234.0)
    batch.metadata["initial_1d"] = torch.full(
        (batch.target_fields.shape[0], batch.target_fields.shape[-1]),
        -4321.0,
        dtype=batch.target_fields.dtype,
    )
    batch.metadata["background_fields"] = torch.full_like(batch.metadata["background_fields"], 999.0)
    background_after = _initial_background(batch)

    assert torch.equal(background_after, background_before)
    assert torch.equal(background_before, batch.metadata["voronoi_grid"][:, :, 0, :])


def test_burgers_strong_constraint_propagator_has_finite_gradient_and_preserves_constant_state():
    initial = torch.ones(1, 1, 16, requires_grad=True)
    trajectory = propagate_burgers_trajectory(
        initial,
        time_steps=12,
        final_time=1.0,
        viscosity=0.01,
        substeps=2,
    )
    assert tuple(trajectory.shape) == (1, 1, 12, 16)
    assert torch.allclose(trajectory, torch.ones_like(trajectory), atol=1e-6)
    trajectory.square().sum().backward()
    assert initial.grad is not None
    assert torch.isfinite(initial.grad).all()


def test_static_pde_var4d_skipped_by_matrix():
    assert "time-varying" in compatibility_reason("var4d", "poisson", "sparse_solution")


def test_var4d_rejects_removed_non_burgers_entry():
    registry = build_default_registry()
    raw = registry.synthetic_raw("shallow_water", n=1, resolution=8)
    batch = registry.make_task(raw, "shallow_water", "sparse_solution", num_sensors=4)
    with pytest.raises(ValueError, match="only to Burgers"):
        Var4DBaseline().build({"steps": 0}, build_data_spec(batch))
