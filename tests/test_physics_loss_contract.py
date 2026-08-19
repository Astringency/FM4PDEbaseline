from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from baselines.common.metrics import physics_loss_metric
from baselines.common.physics import _as_ns_trajectory, dirichlet_zero_bc_loss, physics_losses
from baselines.methods.pc_bnn import _boundary_observation_count, _physics_residual_count
from baselines.run import _batch_metric_payload, _joint_physics_views


def test_static_joint_physics_uses_both_predicted_fields():
    pred = torch.zeros(1, 2, 8, 8)
    target = torch.zeros_like(pred)
    pred[:, 0] = 2.0
    pred[:, 1] = 3.0
    target[:, 0] = 9.0

    solution, coefficient = _joint_physics_views(
        pred,
        {"joint_reconstruction": True, "joint_input_channels": 1},
        torch.zeros(1, 1, 8, 8),
        pde_name="poisson",
    )

    torch.testing.assert_close(solution, pred[:, 1:])
    torch.testing.assert_close(coefficient, pred[:, :1])


def test_ns_joint_physics_uses_both_predicted_endpoints():
    pred = torch.zeros(1, 2, 8, 8)
    target = torch.zeros_like(pred)
    pred[:, 0] = 2.0
    pred[:, 1] = 3.0
    target[:, 0] = 9.0

    trajectory, _ = _joint_physics_views(
        pred,
        {"joint_reconstruction": True, "joint_input_channels": 1},
        torch.zeros(1, 1, 8, 8),
        pde_name="nsnonbounded",
    )

    assert trajectory.shape == (1, 1, 2, 8, 8)
    torch.testing.assert_close(trajectory[:, 0, 0], pred[:, 0])
    torch.testing.assert_close(trajectory[:, 0, 1], pred[:, 1])


@pytest.mark.parametrize("task", ["inverse", "sparse_inverse"])
def test_ns_inverse_physics_orders_predicted_initial_before_given_terminal(task):
    predicted_initial = torch.full((1, 1, 8, 8), 2.0)
    given_terminal = torch.full((1, 1, 8, 8), 9.0)
    sparse_terminal = torch.zeros_like(given_terminal)
    private_true_trajectory = torch.zeros(1, 1, 11, 8, 8)
    private_true_trajectory[:, :, 0] = 1.0
    private_true_trajectory[:, :, -1] = 9.0

    trajectory, mode = _as_ns_trajectory(
        predicted_initial,
        {
            "task": task,
            "input_fields": sparse_terminal,
            "original_input_fields": given_terminal,
            "full_tensor": private_true_trajectory,
        },
    )

    assert mode == "two_level_midpoint_approx"
    torch.testing.assert_close(trajectory[:, :, 0], predicted_initial)
    torch.testing.assert_close(trajectory[:, :, 1], given_terminal)


def test_ns_inverse_total_does_not_add_hidden_true_initial_error():
    predicted_initial = torch.full((1, 1, 8, 8), 2.0)
    given_terminal = torch.full((1, 1, 8, 8), 9.0)
    private_true_trajectory = torch.zeros(1, 1, 11, 8, 8)
    private_true_trajectory[:, :, 0] = 1.0
    private_true_trajectory[:, :, -1] = 9.0

    losses = physics_losses(
        predicted_initial,
        "nsnonbounded",
        {
            "task": "inverse",
            "input_fields": given_terminal,
            "full_tensor": private_true_trajectory,
            "forcing_field": torch.zeros(8, 8),
        },
    )

    assert losses["ic"].item() == pytest.approx(0.0)


def test_static_joint_batch_metric_does_not_fall_back_to_true_full_coefficient():
    class Batch:
        metadata = {"joint_reconstruction": True, "joint_input_channels": 1}
        input_fields = torch.zeros(1, 2, 8, 8)
        full_tensor = torch.zeros(1, 2, 8, 8)
        full_tensor[:, 0] = 9.0
        task = "sparse_solution"
        mask = None
        obs_values = None

    pred = torch.zeros(1, 2, 8, 8)
    pred[:, 0] = 2.0
    payload = _batch_metric_payload(
        pred,
        Batch.full_tensor,
        Batch(),
        Namespace(physics_metric_mode="per_batch", pde="poisson"),
    )

    assert payload["pde_residual"] == pytest.approx(4.0)


def test_metric_wrapper_preserves_raw_residual_for_likelihood_counting():
    pred = torch.zeros(1, 1, 8, 8)
    losses = physics_loss_metric(
        pred,
        "poisson",
        {"task": "forward", "input_fields": torch.zeros_like(pred)},
        strict=True,
    )

    assert losses["residual"].shape == (1, 1, 6, 6)
    assert _physics_residual_count(losses, pred) == 36


def test_dirichlet_boundary_loss_weights_unique_boundary_nodes_uniformly():
    field = torch.zeros(1, 1, 4, 4)
    field[..., 0, 0] = 1.0

    assert dirichlet_zero_bc_loss(field).item() == pytest.approx(1.0 / 12.0)
    assert _boundary_observation_count("poisson", field) == 12


def test_strict_physics_metric_raises_instead_of_returning_nan():
    pred = torch.ones(1, 1, 8, 8)

    with pytest.raises(ValueError, match="Missing required scalar PDE parameter"):
        physics_loss_metric(
            pred,
            "heat",
            {"task": "forward", "input_fields": pred, "final_time": 1.0},
            strict=True,
        )


def test_formal_batch_metric_path_is_strict(monkeypatch):
    class Batch:
        pde_name = "poisson"
        metadata = {}
        input_fields = torch.zeros(1, 1, 8, 8)
        full_tensor = torch.zeros(1, 2, 8, 8)
        task = "forward"
        mask = None
        obs_values = None

    def invalid_metric(*args, **kwargs):
        assert kwargs["strict"] is True
        raise RuntimeError("invalid PDE metric")

    monkeypatch.setattr("baselines.run.physics_loss_metric", invalid_metric)
    with pytest.raises(RuntimeError, match="invalid PDE metric"):
        _batch_metric_payload(
            torch.zeros(1, 1, 8, 8),
            torch.zeros(1, 1, 8, 8),
            Batch(),
            Namespace(physics_metric_mode="per_batch", pde="poisson"),
        )
