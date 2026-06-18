from __future__ import annotations

import torch

from baselines.common.data_adapter import build_default_registry
from baselines.experiment_matrix import compatibility_reason


def test_sparse_inverse_observes_solution_side_and_targets_coefficient():
    registry = build_default_registry()
    coeff = torch.zeros(2, 1, 8, 8)
    solution = torch.ones(2, 1, 8, 8)
    raw = {
        "full_tensor": torch.cat([coeff, solution], dim=1),
        "channel_names": ["f", "phi"],
        "metadata": {"canonical_layout": "NCHW", "split": "train"},
        "split": "train",
    }
    batch = registry.make_task(raw, "poisson", "sparse_inverse", num_sensors=6, noise_level=0.0, seed=3)
    assert torch.allclose(batch.target_fields, coeff)
    assert torch.all(batch.obs_values == 1.0)
    assert torch.all(batch.input_fields[batch.input_fields != 0.0] == 1.0)
    assert "observation_source_fields" in batch.metadata


def test_static_per_instance_sparse_inverse_is_supported():
    reason = compatibility_reason("pde_opt", "poisson", "sparse_inverse")
    assert reason == ""
    assert "disabled" in compatibility_reason("pde_opt", "heat", "sparse_inverse")
