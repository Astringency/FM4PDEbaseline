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


def test_reaction_diffusion_eager_preserves_physical_metadata(tiny_data_root):
    registry = build_default_registry()
    raw = registry.load_raw("reaction_diffusion", tiny_data_root, split="train", max_samples=2)
    batch = registry.make_task(raw, "reaction_diffusion", "forward")
    for key in ("init_mode", "boundary_condition", "dx", "dy", "D_u", "D_v", "k", "T"):
        assert key in batch.metadata
    assert batch.metadata["bc"] == "periodic"
    assert set(batch.pde_params) == {"D_u", "D_v", "k"}


def test_reaction_diffusion_eager_uses_initial_endpoint_and_sample_specific_params(tmp_path):
    rd = tmp_path / "reaction_diffusion"
    rd.mkdir()
    path = rd / "reaction_diffusion-128-128-10_0.h5"
    rng = np.random.default_rng(9)
    with h5py.File(path, "w") as f:
        f.attrs["D_u"] = 1e-3
        f.attrs["D_v"] = 5e-3
        f.attrs["k"] = 5e-3
        f.attrs["T"] = 5.0
        f.attrs["dx"] = 0.125
        f.attrs["dy"] = 0.125
        f.attrs["x_range"] = np.asarray([-1.0, 1.0], dtype="float32")
        f.attrs["y_range"] = np.asarray([-1.0, 1.0], dtype="float32")
        f.attrs["init_mode"] = "grf"
        f.attrs["boundary_condition"] = "neumann"
        for i in range(3):
            data = rng.normal(size=(10, 8, 8, 2)).astype("float32")
            data[0, :, :, :] = float(i + 1)
            data[-1, :, :, :] = float(i + 11)
            g = f.create_group(str(i))
            g.attrs["D_u"] = 0.01 + i
            g.attrs["D_v"] = 0.02 + i
            g.attrs["k"] = 0.03 + i
            g.attrs["sample_seed"] = 100 + i
            g["data"] = data

    registry = build_default_registry()
    raw = registry.load_raw("reaction_diffusion", tmp_path, split="train", max_samples=3, load_full_trajectory=False)

    assert raw["metadata"]["input_time_index"] == 0
    assert torch.allclose(raw["full_tensor"][:, 0], torch.tensor([1.0, 2.0, 3.0]).reshape(3, 1, 1).expand(3, 8, 8))
    assert torch.allclose(raw["full_tensor"][:, 2], torch.tensor([11.0, 12.0, 13.0]).reshape(3, 1, 1).expand(3, 8, 8))
    assert torch.allclose(raw["pde_params"]["D_u"], torch.tensor([0.01, 1.01, 2.01], dtype=torch.float32))
    assert torch.allclose(raw["pde_params"]["D_v"], torch.tensor([0.02, 1.02, 2.02], dtype=torch.float32))
    assert torch.allclose(raw["pde_params"]["k"], torch.tensor([0.03, 1.03, 2.03], dtype=torch.float32))
    assert torch.allclose(raw["metadata"]["sample_seed"], torch.tensor([100.0, 101.0, 102.0]))
    assert raw["metadata"]["x_range"] == [-1.0, 1.0]
    assert raw["metadata"]["boundary_condition"] == "neumann"

    full = registry.load_raw("reaction_diffusion", tmp_path, split="train", max_samples=1, load_full_trajectory=True)
    batch = registry.make_task(full, "reaction_diffusion", "forward")
    assert batch.metadata["input_time_index"] == 0
    assert torch.allclose(batch.input_fields, batch.full_tensor[:, :, 0])
    assert torch.allclose(batch.target_fields, batch.full_tensor[:, :, -1])


def test_reaction_diffusion_grf_shard_name_supported_in_eager_mode(tmp_path):
    rd = tmp_path / "reaction_diffusion"
    rd.mkdir()
    path = rd / "reaction_diffusion_grf_50000-128-128-T1-steps10_shard000.h5"
    rng = np.random.default_rng(123)
    with h5py.File(path, "w") as f:
        f["t"] = np.linspace(0.0, 1.0, 10).astype("float32")
        meta = f.create_group("metadata")
        meta.attrs["source"] = "unit-test"
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
        data_loading_mode="eager",
        load_full_trajectory=False,
        num_sensors=4,
        seed=5,
    )

    assert len(dataset) == 2
    assert dataset.batch.file_paths == [str(path)]
    assert dataset.batch.metadata["data_loading_mode"] == "eager"
    assert dataset.batch.full_tensor.shape == (2, 4, 8, 8)


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
