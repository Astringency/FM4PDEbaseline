from __future__ import annotations

import math

import pytest
import torch

from baselines.run import _joint_reconstruction_relative_l2_values, _relative_l2_input_or_coeff_values


def test_inverse_relative_l2_input_or_coeff_has_one_value_per_sample():
    target = torch.ones(4, 1, 8, 8)
    pred = target * 2.0
    values = _relative_l2_input_or_coeff_values("inverse", pred, target)
    assert len(values) == 4
    assert all(v > 0 for v in values)


def test_forward_relative_l2_input_or_coeff_is_documented_nan_per_sample():
    target = torch.ones(4, 1, 8, 8)
    pred = target
    values = _relative_l2_input_or_coeff_values("forward", pred, target)
    assert len(values) == 4
    assert all(math.isnan(v) for v in values)


def test_joint_sparse_solution_reports_input_and_solution_errors_separately():
    target = torch.tensor([[[[2.0]], [[4.0]]]])
    pred = torch.tensor([[[[1.0]], [[3.0]]]])

    solution, input_or_coeff = _joint_reconstruction_relative_l2_values(
        pred,
        target,
        {"joint_reconstruction": True, "joint_input_channels": 1},
    )

    assert solution == [0.25]
    assert input_or_coeff == [0.5]


@pytest.mark.parametrize(
    "errors",
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 3.0],
        [1.0, 2.0, 2.0, 2.0],
    ],
)
def test_burger_solution_error_covers_every_time_and_keeps_initial_error(errors):
    target = torch.tensor([1.0, 2.0, 4.0, 8.0]).reshape(1, 1, 4, 1).expand(1, 1, 4, 2)
    pred = target + torch.tensor(errors).reshape(1, 1, 4, 1)

    solution, input_or_coeff = _joint_reconstruction_relative_l2_values(
        pred,
        target,
        {
            "joint_reconstruction": True,
            "joint_split_axis": 2,
            "joint_input_extent": 1,
        },
    )

    # One norm over all time-space points, including the initial slice;
    # unequal target magnitudes distinguish it from a mean of per-time errors.
    assert solution == pytest.approx([math.sqrt(sum(error**2 for error in errors) / 85.0)])
    assert input_or_coeff == pytest.approx([abs(errors[0])])
