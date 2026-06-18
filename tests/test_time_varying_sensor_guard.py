from __future__ import annotations

import pytest
import torch

from baselines.common.data_adapter import build_default_registry


def test_time_varying_sensor_rejects_final_state_target_in_paper_mode():
    registry = build_default_registry()
    raw = registry.synthetic_raw("heat", n=2, resolution=8)
    with pytest.raises(ValueError, match="time_varying"):
        registry.make_task(raw, "heat", "sparse_solution", num_sensors=4, sensor_mode="time_varying", experiment_mode="paper")


def test_time_varying_sensor_full_trajectory_target_has_time_masks():
    registry = build_default_registry()
    raw = registry.synthetic_raw("reaction_diffusion", n=2, resolution=8)
    batch = registry.make_task(raw, "reaction_diffusion", "sparse_solution", num_sensors=4, sensor_mode="time_varying", experiment_mode="paper")
    assert batch.target_fields.ndim == 5
    assert batch.mask.ndim == 4
    assert batch.metadata["time_varying_sensor_valid"] is True
    assert any(not torch.equal(batch.mask[:, t], batch.mask[:, 0]) for t in range(1, batch.mask.shape[1]))


def test_burgers_time_varying_sensor_keeps_tx_trajectory():
    registry = build_default_registry()
    raw = registry.synthetic_raw("burger", n=2, resolution=8)
    batch = registry.make_task(raw, "burger", "sparse_solution", num_sensors=4, sensor_mode="time_varying", experiment_mode="paper")
    assert batch.target_fields.ndim == 4
    assert tuple(batch.target_fields.shape[2:]) == (8, 8)
    assert batch.mask.ndim == 3
    assert batch.metadata["time_varying_sensor_valid"] is True
    assert any(not torch.equal(batch.mask[:, t], batch.mask[:, 0]) for t in range(1, batch.mask.shape[1]))
