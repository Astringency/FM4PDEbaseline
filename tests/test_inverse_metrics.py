from __future__ import annotations

import math

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
