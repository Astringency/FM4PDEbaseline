from __future__ import annotations

import torch

from baselines.common.sensors import add_noise, build_observation_tensors, make_sensor_mask


def test_sensor_modes_and_noise():
    for mode in ["random", "fixed", "grid"]:
        mask = make_sensor_mask((2, 8, 8), 10, mode, seed=1)
        assert mask.shape == (2, 8, 8)
        assert mask[0].sum() <= 10
        assert torch.equal(mask[0], mask[1])

    tv = make_sensor_mask((1, 4, 8, 8), 5, "time_varying", seed=1, time_dim=0)
    assert tv.shape == (1, 4, 8, 8)
    assert tv[0].sum() == 20

    x = torch.ones(2, 3)
    noisy = add_noise(x, 0.1, seed=2)
    assert noisy.shape == x.shape


def test_build_observation_tensors_shapes():
    target = torch.randn(2, 1, 8, 8)
    obs = build_observation_tensors(target, num_sensors=7, mode="random", seed=4, noise_level=0.01)
    assert obs["mask"].shape == (1, 8, 8)
    assert obs["masked_grid"].shape == target.shape
    assert obs["voronoi_grid"].shape == target.shape
    assert obs["obs_values"].shape == (2, 7, 1)
    assert obs["obs_coords"].shape == (2, 7, 2)

