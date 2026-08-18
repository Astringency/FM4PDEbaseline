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
    tv_total = make_sensor_mask((1, 4, 8, 8), 5, "time_varying", seed=1, time_dim=0, sensor_budget_mode="total")
    assert tv_total[0].sum() <= 5

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
    assert obs["num_observations_total"] == 7
    assert obs["sensor_budget_mode"] == "per_time"


def test_time_varying_sensor_budget_modes():
    target = torch.randn(2, 1, 4, 8, 8)
    per_time = build_observation_tensors(target, num_sensors=5, mode="time_varying", seed=4, sensor_budget_mode="per_time")
    total = build_observation_tensors(target, num_sensors=5, mode="time_varying", seed=4, sensor_budget_mode="total")
    assert per_time["num_observations_total"] == 20
    assert sum(per_time["num_sensors_per_time"]) == 20
    assert total["num_observations_total"] <= 5
    assert sum(total["num_sensors_per_time"]) <= 5


def test_random_per_sample_temporal_budget_modes_are_honored():
    target = torch.randn(2, 1, 4, 8)
    common = {
        "num_sensors": 3,
        "mode": "random_per_sample",
        "seed": 4,
        "time_dim": 0,
        "sample_ids": ["burger:test:0", "burger:test:1"],
        "split": "test",
    }
    per_time = build_observation_tensors(
        target, sensor_budget_mode="per_time", **common
    )
    total = build_observation_tensors(
        target, sensor_budget_mode="total", **common
    )

    assert per_time["num_observations_total"] == 12
    assert per_time["num_sensors_per_time"] == [3, 3, 3, 3]
    assert total["num_observations_total"] == 3
    assert sum(total["num_sensors_per_time"]) == 3


def test_random_per_sample_masks_are_deterministic_and_sample_specific():
    target = torch.randn(3, 1, 8, 8)
    kwargs = {
        "num_sensors": 7,
        "mode": "random_per_sample",
        "seed": 11,
        "sample_ids": ["train:10", "train:11", "train:12"],
        "split": "train",
        "epoch": 2,
    }
    first = build_observation_tensors(target, **kwargs)
    repeat = build_observation_tensors(target, **kwargs)
    next_epoch = build_observation_tensors(target, **{**kwargs, "epoch": 3})

    assert first["mask"].shape == target.shape
    assert torch.equal(first["mask"], repeat["mask"])
    assert not torch.equal(first["mask"][0], first["mask"][1])
    assert not torch.equal(first["mask"], next_epoch["mask"])
    assert len(first["mask_ids"]) == len(target)


def test_random_per_sample_validation_mask_does_not_depend_on_epoch():
    target = torch.randn(2, 1, 8, 8)
    common = {
        "num_sensors": 5,
        "mode": "random_per_sample",
        "seed": 7,
        "sample_ids": ["val:1", "val:2"],
        "split": "val",
    }
    epoch_zero = build_observation_tensors(target, epoch=0, **common)
    epoch_late = build_observation_tensors(target, epoch=99, **common)
    assert torch.equal(epoch_zero["mask"], epoch_late["mask"])
