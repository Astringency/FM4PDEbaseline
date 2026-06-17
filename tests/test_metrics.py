from __future__ import annotations

import torch

from baselines.common.metrics import mae, mse, obs_mse, pde_residual_metric, relative_l2


def test_basic_metrics():
    target = torch.ones(2, 1, 4, 4)
    pred = target * 2
    mask = torch.ones(1, 4, 4)
    assert relative_l2(pred, target).item() == 1.0
    assert mse(pred, target).item() == 1.0
    assert mae(pred, target).item() == 1.0
    assert obs_mse(pred, target, mask).item() == 1.0


def test_shallow_water_residual_with_metadata_is_finite():
    pred = torch.randn(1, 3, 4, 4)
    pred[:, 0] = pred[:, 0].abs() + 1.0
    meta = {"input_fields": torch.ones(1, 3, 4, 4), "task": "forward", "final_time": 1.0}
    value = pde_residual_metric(pred, "shallow_water", meta)
    assert torch.isfinite(value)
