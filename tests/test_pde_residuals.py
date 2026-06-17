from __future__ import annotations

import pytest
import torch

from baselines.common.metrics import NotImplementedWarning, pde_residual_metric


@pytest.mark.parametrize("pde", ["poisson", "helmholtz", "darcy", "burger"])
def test_supported_residuals_do_not_crash(pde):
    pred = torch.randn(1, 1, 8, 8)
    meta = {"input_fields": torch.randn(1, 1, 8, 8)}
    value = pde_residual_metric(pred, pde, meta)
    assert value.ndim == 0


@pytest.mark.parametrize("pde,channels", [("nsnonbounded", 10), ("reaction_diffusion", 2), ("shallow_water", 3)])
def test_time_dependent_residuals_are_implemented(pde, channels):
    pred = torch.randn(1, channels, 8, 8)
    if pde == "shallow_water":
        pred[:, 0] = pred[:, 0].abs() + 1.0
    meta = {
        "input_fields": torch.randn(1, channels if pde != "nsnonbounded" else 1, 8, 8),
        "full_tensor": torch.randn(1, 1 if pde == "nsnonbounded" else channels, 11, 8, 8),
        "task": "forward",
    }
    if pde == "shallow_water":
        meta["full_tensor"][:, 0] = meta["full_tensor"][:, 0].abs() + 1.0
    value = pde_residual_metric(pred, pde, meta)
    assert value.ndim == 0
    assert torch.isfinite(value)
