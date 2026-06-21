from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from baselines.common.data_adapter import build_default_registry


CURRENT_PDES = [
    "darcy",
    "poisson",
    "helmholtz",
    "nsnonbounded",
    "burger",
    "reaction_diffusion",
    "shallow_water",
    "heat",
    "wave",
    "advection_diffusion",
    "steady_heat_conduction",
]


@pytest.mark.parametrize("pde", CURRENT_PDES)
def test_adapter_loads_tiny_native_formats(tiny_data_root, pde):
    registry = build_default_registry()
    raw = registry.load_raw(pde, tiny_data_root, split="train", max_samples=2)
    batch = registry.make_task(raw, pde, "forward", num_sensors=5, seed=3)
    assert batch.input_fields.shape[0] == 2
    assert batch.target_fields.shape[0] == 2
    assert batch.full_tensor.shape[0] == 2
    assert batch.mask is not None
    assert batch.obs_values is not None
    assert batch.obs_coords is not None


@pytest.mark.parametrize("pde", ["heat", "wave", "advection_diffusion", "steady_heat_conduction"])
def test_hdf5_pdes_warn_and_synthetic_fallback_when_files_missing(tmp_path, pde):
    registry = build_default_registry()
    with pytest.warns(RuntimeWarning):
        raw = registry.load_raw(pde, tmp_path, synthetic_if_missing=True, max_samples=2, synthetic_resolution=8)
    batch = registry.make_task(raw, pde, "forward")
    assert batch.target_fields.shape[-2:] == (8, 8)


def test_future_hdf5_scalar_params_default_to_metadata(tiny_data_root):
    registry = build_default_registry()
    raw = registry.load_raw("heat", tiny_data_root, split="train", max_samples=2, scalar_param_mode="metadata")
    batch = registry.make_task(raw, "heat", "forward")
    assert batch.input_fields.shape[1] == 1
    assert batch.target_fields.shape[1] == 1
    assert "alpha" in batch.pde_params
    assert "alpha" in batch.metadata["pde_params"]
    assert "full_trajectory" in batch.metadata
    assert batch.input_channel_names == ["u0"]
    assert batch.target_channel_names == ["uT"]


def test_future_hdf5_materialize_scalar_params_only_when_requested(tiny_data_root):
    registry = build_default_registry()
    raw = registry.load_raw("heat", tiny_data_root, split="train", max_samples=2, scalar_param_mode="materialize")
    batch = registry.make_task(raw, "heat", "forward")
    assert batch.input_fields.shape[1] == 2
    assert batch.target_fields.shape[1] == 2
    assert batch.input_channel_names == ["u0", "alpha"]
    assert torch.allclose(batch.input_fields[:, 1], batch.pde_params["alpha"].reshape(-1, 1, 1).expand_as(batch.input_fields[:, 1]))


def test_reaction_diffusion_grf_shard_name_supported_in_lazy_mode(tmp_path):
    rd = tmp_path / "reaction_diffusion"
    rd.mkdir()
    path = rd / "reaction_diffusion_grf_50000-128-128-T1-steps10_shard000.h5"
    rng = np.random.default_rng(123)
    with h5py.File(path, "w") as f:
        for i in range(2):
            g = f.create_group(str(i))
            g["data"] = rng.normal(size=(10, 8, 8, 2)).astype("float32")

    registry = build_default_registry()
    raw = registry.load_raw(
        "reaction_diffusion",
        tmp_path,
        split="train",
        max_samples=1,
        train_shards=1,
        load_full_trajectory=False,
    )
    assert raw["file_paths"] == [str(path)]
    assert raw["full_tensor"].shape == (1, 4, 8, 8)

    dataset = registry.make_dataset(
        "reaction_diffusion",
        tmp_path,
        "sparse_solution",
        split="train",
        max_samples=2,
        train_shards=1,
        data_loading_mode="lazy",
        load_full_trajectory=False,
        num_sensors=4,
        seed=5,
    )

    assert len(dataset) == 2
    assert dataset.file_paths == [str(path)]
    assert dataset.batch.metadata["data_loading_mode"] == "lazy"
    assert dataset.batch.full_tensor.shape == (1, 4, 8, 8)


def test_train_val_test_are_distinct_splits(tiny_data_root):
    registry = build_default_registry()
    train = registry.make_dataset("poisson", tiny_data_root, "forward", split="train", max_samples=2)
    val = registry.make_dataset("poisson", tiny_data_root, "forward", split="val", max_samples=1, sample_offset=2, val_from_train_offset=2)
    test = registry.make_dataset("poisson", tiny_data_root, "forward", split="test", max_samples=2)
    assert train.batch.split == "train"
    assert val.batch.split == "val"
    assert val.batch.metadata["split_source"] == "deterministic_train_subset"
    assert test.batch.split == "test"
    assert all("test" not in p.rsplit("/", 1)[-1] for p in val.batch.file_paths)
    assert any("test" in p.rsplit("/", 1)[-1] for p in test.batch.file_paths)


def test_sensor_budget_metadata_static_and_time_varying():
    registry = build_default_registry()
    raw_static = registry.synthetic_raw("poisson", n=1, resolution=8)
    static = registry.make_task(raw_static, "poisson", "sparse_solution", num_sensors=5, sensor_mode="random", seed=1)
    assert static.metadata["num_observations_total"] == 5
    assert static.metadata["num_sensors_per_time"] == 5
    assert static.metadata["sensor_budget_mode"] == "per_time"

    raw_tv = registry.synthetic_raw("reaction_diffusion", n=1, resolution=8)
    per_time = registry.make_task(
        raw_tv,
        "reaction_diffusion",
        "sparse_solution",
        num_sensors=4,
        sensor_mode="time_varying",
        sensor_budget_mode="per_time",
        seed=1,
    )
    total = registry.make_task(
        raw_tv,
        "reaction_diffusion",
        "sparse_solution",
        num_sensors=4,
        sensor_mode="time_varying",
        sensor_budget_mode="total",
        seed=1,
    )
    assert per_time.metadata["num_observations_total"] == 40
    assert total.metadata["num_observations_total"] <= 4


def test_missing_file_error_lists_pde_and_candidates(tmp_path):
    registry = build_default_registry()
    with pytest.raises(FileNotFoundError) as exc:
        registry.load_raw("heat", tmp_path, split="test", max_samples=1)
    message = str(exc.value)
    assert "heat" in message
    assert "Candidate paths" in message
