from __future__ import annotations

import pytest

from baselines.common.data_adapter import build_default_registry
from baselines.methods.pinn_sparse import observation_loss_from_batch
from baselines.run import _obs_mse_clean, _obs_mse_noisy


def test_obs_mse_clean_and_noisy_match_without_noise():
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=1, resolution=8)
    batch = registry.make_task(raw, "poisson", "sparse_solution", num_sensors=8, noise_level=0.0, seed=1)
    pred = batch.target_fields.clone()
    assert _obs_mse_clean(pred, batch.target_fields, batch) == pytest.approx(0.0)
    assert _obs_mse_noisy(pred, batch) == pytest.approx(0.0)


def test_obs_mse_clean_and_noisy_diverge_with_noise_and_loss_uses_noisy_obs():
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=1, resolution=8)
    batch = registry.make_task(raw, "poisson", "sparse_solution", num_sensors=8, noise_level=0.1, seed=1)
    pred = batch.target_fields.clone()
    assert _obs_mse_clean(pred, batch.target_fields, batch) == pytest.approx(0.0)
    assert _obs_mse_noisy(pred, batch) > 0.0
    assert observation_loss_from_batch(pred, batch).item() == pytest.approx(_obs_mse_noisy(pred, batch))
