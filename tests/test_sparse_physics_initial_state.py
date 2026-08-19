from __future__ import annotations

import torch

from baselines.common.data_adapter import build_default_registry
from baselines.common.physics import _as_rd_trajectory, _as_swe_trajectory, physics_losses


def test_reaction_diffusion_sparse_background_uses_true_initial_state():
    registry = build_default_registry()
    raw = registry.synthetic_raw("reaction_diffusion", n=1, resolution=8, seed=7)
    batch = registry.make_task(raw, "reaction_diffusion", "sparse_solution", num_sensors=4, seed=3)
    input_idx = int(batch.metadata["input_time_index"])
    true_initial = batch.full_tensor[:, :, input_idx]
    assert torch.allclose(batch.metadata["background_fields"], true_initial)
    assert torch.allclose(batch.metadata["original_input_fields"], true_initial)
    assert not torch.allclose(batch.input_fields[:, :2], true_initial)

    meta = {"input_fields": batch.input_fields, "full_tensor": batch.full_tensor, "task": batch.task, **batch.metadata}
    traj, mode = _as_rd_trajectory(batch.target_fields, meta)
    assert mode == "two_level_midpoint_approx"
    assert torch.allclose(traj[:, :2, 0], true_initial)
    losses = physics_losses(batch.target_fields, "reaction_diffusion", meta)
    assert torch.isfinite(losses["ic"])


def test_shallow_water_sparse_background_uses_true_initial_state():
    registry = build_default_registry()
    raw = registry.synthetic_raw("shallow_water", n=1, resolution=8, seed=9)
    batch = registry.make_task(raw, "shallow_water", "sparse_solution", num_sensors=4, seed=4)
    true_initial = batch.full_tensor[:, :, 0]
    assert torch.allclose(batch.metadata["background_fields"], true_initial)
    assert torch.allclose(batch.metadata["original_input_fields"], true_initial)
    assert not torch.allclose(batch.input_fields[:, :3], true_initial)

    meta = {"input_fields": batch.input_fields, "full_tensor": batch.full_tensor, "task": batch.task, **batch.metadata}
    traj, mode = _as_swe_trajectory(batch.target_fields, meta)
    assert mode == "two_level_midpoint_approx"
    assert torch.allclose(traj[:, :3, 0], true_initial)
    losses = physics_losses(batch.target_fields, "shallow_water", meta)
    assert torch.isfinite(losses["ic"])
