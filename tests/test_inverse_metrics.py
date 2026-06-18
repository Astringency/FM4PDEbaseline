from __future__ import annotations

import math

import torch

from baselines.run import _relative_l2_input_or_coeff_values


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
