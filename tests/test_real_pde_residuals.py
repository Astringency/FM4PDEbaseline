"""Opt-in integration checks against an actual FM4PDE dataset mount.

Run with, for example:
    FM4PDE_REAL_DATA_ROOT=/path/to/PDEdata python -m pytest -q -s tests/test_real_pde_residuals.py

The checks require an explicit environment variable so the normal unit suite
never substitutes synthetic data for this real-data audit.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from baselines.common.data_adapter import build_default_registry
from baselines.common.metrics import physics_loss_metric


REAL_DATA_ROOT = os.environ.get("FM4PDE_REAL_DATA_ROOT")
PDES = (
    "poisson",
    "helmholtz",
    "darcy",
    "burger",
    "nsnonbounded",
    "reaction_diffusion",
    "shallow_water",
    "heat",
    "wave",
    "advection_diffusion",
    "steady_heat_conduction",
)


@pytest.mark.skipif(not REAL_DATA_ROOT, reason="set FM4PDE_REAL_DATA_ROOT to enable the real-data audit")
@pytest.mark.parametrize("pde", PDES)
def test_exact_real_dataset_pairs_produce_finite_physics_diagnostics(pde):
    root = Path(REAL_DATA_ROOT)
    registry = build_default_registry()
    try:
        raw = registry.load_raw(
            pde,
            root,
            split="test",
            max_samples=2,
            load_full_trajectory=True,
            strict_size=False,
        )
    except FileNotFoundError as exc:
        pytest.skip(f"{pde} is absent from this dataset mount: {exc}")

    batch = registry.make_task(raw, pde, "forward", num_sensors=16, seed=0)
    metadata = {
        "input_fields": batch.input_fields,
        "full_tensor": batch.full_tensor,
        "task": batch.task,
        **batch.metadata,
    }
    stored_trajectory = batch.metadata.get("full_trajectory")
    if isinstance(stored_trajectory, torch.Tensor) and stored_trajectory.ndim == 5:
        physics_field = stored_trajectory
    elif batch.full_tensor.ndim == 5:
        physics_field = batch.full_tensor
    else:
        physics_field = batch.target_fields
    metrics = physics_loss_metric(physics_field, pde, metadata)
    for name in ("interior", "bc", "ic", "total"):
        assert torch.isfinite(metrics[name]), f"{pde} produced a non-finite {name} diagnostic"
    print(
        f"{pde}: interior={float(metrics['interior']):.6g}, "
        f"bc={float(metrics['bc']):.6g} ({metrics['bc_status']}), "
        f"ic={float(metrics['ic']):.6g}, mode={metrics['mode']}"
    )

    if pde in {"poisson", "helmholtz"}:
        sign = float(batch.metadata["elliptic_operator_sign"])
        alternatives = batch.metadata["elliptic_operator_inference_mse"]
        selected_key = "positive" if sign > 0 else "negative"
        assert float(alternatives[selected_key]) == pytest.approx(min(float(value) for value in alternatives.values()))
