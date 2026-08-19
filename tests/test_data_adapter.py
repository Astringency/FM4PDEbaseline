from __future__ import annotations

import h5py
import numpy as np
import pytest
import scipy.io
import torch
from torch.utils.data import DataLoader

from baselines.common.data_adapter import build_default_registry, pde_collate


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


@pytest.mark.parametrize("operator_sign", [1.0, -1.0])
def test_poisson_loader_infers_equation_sign_from_exact_dataset_pairs(tmp_path, operator_sign):
    pde_dir = tmp_path / "poisson"
    pde_dir.mkdir()
    n = 16
    x = np.linspace(0.0, 1.0, n)
    yy, xx = np.meshgrid(x, x, indexing="ij")
    solution = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    h = 1.0 / (n - 1)
    lap = np.zeros_like(solution)
    lap[1:-1, 1:-1] = (
        solution[1:-1, :-2]
        + solution[1:-1, 2:]
        + solution[:-2, 1:-1]
        + solution[2:, 1:-1]
        - 4.0 * solution[1:-1, 1:-1]
    ) / h**2
    source = operator_sign * lap
    scipy.io.savemat(
        pde_dir / "poisson_10000-128-128_1.mat",
        {"f_data": source[None].astype("float32"), "phi_data": solution[None].astype("float32")},
    )
    raw = build_default_registry().load_raw("poisson", tmp_path, split="train", max_samples=1)
    assert raw["metadata"]["elliptic_operator_sign"] == operator_sign


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
        f.attrs["Du"] = 0.10
        f.attrs["Dv"] = 0.20
        f.attrs["k"] = 5e-3
        f.attrs["total_time"] = 7.0
        f.attrs["dx"] = 0.125
        f.attrs["dy"] = 0.125
        f.attrs["x_min"] = -9.0
        f.attrs["x_max"] = 9.0
        f.attrs["y_min"] = -8.0
        f.attrs["y_max"] = 8.0
        f.attrs["init_mode"] = "grf"
        f.attrs["bc"] = "periodic"
        meta = f.create_group("metadata")
        meta.attrs["x_range"] = np.asarray([-1.0, 1.0], dtype="float32")
        meta.attrs["y_range"] = np.asarray([-2.0, 2.0], dtype="float32")
        meta.attrs["boundary_condition_kind"] = "neumann"
        for i in range(3):
            data = rng.normal(size=(10, 8, 8, 2)).astype("float32")
            data[0, :, :, :] = float(i + 1)
            data[-1, :, :, :] = float(i + 11)
            g = f.create_group(str(i))
            if i in {0, 2}:
                g.attrs["D_u"] = 0.01 + i
            if i in {1, 2}:
                g.attrs["D_v"] = 0.02 + i
                g.attrs["k"] = 0.03 + i
            g.attrs["seed"] = 100 + i
            g["data"] = data

    registry = build_default_registry()
    raw = registry.load_raw("reaction_diffusion", tmp_path, split="train", max_samples=3, load_full_trajectory=False)

    assert raw["metadata"]["input_time_index"] == 0
    assert torch.allclose(raw["full_tensor"][:, 0], torch.tensor([1.0, 2.0, 3.0]).reshape(3, 1, 1).expand(3, 8, 8))
    assert torch.allclose(raw["full_tensor"][:, 2], torch.tensor([11.0, 12.0, 13.0]).reshape(3, 1, 1).expand(3, 8, 8))
    assert torch.allclose(raw["pde_params"]["D_u"], torch.tensor([0.01, 0.10, 2.01], dtype=torch.float32))
    assert torch.allclose(raw["pde_params"]["D_v"], torch.tensor([0.20, 1.02, 2.02], dtype=torch.float32))
    assert torch.allclose(raw["pde_params"]["k"], torch.tensor([0.005, 1.03, 2.03], dtype=torch.float32))
    assert raw["pde_params"]["D_u"].dtype == torch.float32
    assert raw["pde_params"]["D_u"].shape == (3,)
    assert raw["metadata"]["T"] == pytest.approx(7.0)
    assert raw["metadata"]["final_time"] == pytest.approx(7.0)
    assert raw["metadata"]["boundary_condition"] == "neumann"
    assert raw["metadata"]["bc"] == "neumann"
    assert raw["metadata"]["x_left"] == pytest.approx(-1.0)
    assert raw["metadata"]["x_right"] == pytest.approx(1.0)
    assert raw["metadata"]["y_bottom"] == pytest.approx(-2.0)
    assert raw["metadata"]["y_top"] == pytest.approx(2.0)
    assert torch.allclose(raw["metadata"]["sample_seed"], torch.tensor([100.0, 101.0, 102.0]))

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


def test_cell_centered_layout_maps_grid_and_sensor_coordinates_to_cell_centers():
    registry = build_default_registry()
    raw = registry.synthetic_raw("darcy", n=1, resolution=4)
    raw["metadata"]["grid_layout"] = "cell_centered"
    batch = registry.make_task(raw, "darcy", "sparse_forward", num_sensors=4, sensor_mode="fixed", seed=1)

    assert torch.isclose(batch.coords.min(), torch.tensor(0.125))
    assert torch.isclose(batch.coords.max(), torch.tensor(0.875))
    assert batch.obs_coords is not None
    assert bool(((batch.obs_coords >= 0.125) & (batch.obs_coords <= 0.875)).all())


def test_legacy_random_sensor_mode_is_rejected_in_paper_mode():
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=2, resolution=8)
    with pytest.raises(ValueError, match="ambiguous"):
        registry.make_task(
            raw,
            "poisson",
            "sparse_solution",
            num_sensors=5,
            sensor_mode="random",
            experiment_mode="paper",
        )


def test_random_per_sample_dataset_refreshes_training_masks_by_epoch():
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=3, resolution=8, split="train")
    batch = registry.make_task(
        raw,
        "poisson",
        "sparse_solution",
        num_sensors=5,
        sensor_mode="random_per_sample",
        seed=3,
        experiment_mode="paper",
    )
    dataset = registry.make_dataset_from_batch(batch)
    loader = DataLoader(dataset, batch_size=3, collate_fn=pde_collate, shuffle=False)
    first = next(iter(loader))
    dataset.set_epoch(1)
    second = next(iter(loader))

    assert first.mask.shape == first.input_fields.shape
    assert not torch.equal(first.mask[0], first.mask[1])
    assert not torch.equal(first.mask, second.mask)


def test_random_per_sample_epoch_reaches_persistent_workers():
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=3, resolution=8, split="train")
    batch = registry.make_task(
        raw,
        "poisson",
        "sparse_solution",
        num_sensors=5,
        sensor_mode="random_per_sample",
        seed=3,
        experiment_mode="paper",
    )
    dataset = registry.make_dataset_from_batch(batch)
    loader = DataLoader(
        dataset,
        batch_size=3,
        collate_fn=pde_collate,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
    )
    first = next(iter(loader))
    dataset.set_epoch(1)
    second = next(iter(loader))

    assert not torch.equal(first.mask, second.mask)


def test_burger_full_inverse_is_terminal_state_to_initial_state():
    registry = build_default_registry()
    raw = registry.synthetic_raw("burger", n=2, resolution=8)
    raw["full_tensor"][:, :, 0, :] = 1.0
    raw["full_tensor"][:, :, -1, :] = 9.0
    raw["metadata"]["initial_1d"] = torch.ones(2, 8)
    batch = registry.make_task(raw, "burger", "inverse")

    assert tuple(batch.input_fields.shape) == (2, 1, 1, 8)
    assert tuple(batch.target_fields.shape) == (2, 1, 1, 8)
    assert torch.all(batch.input_fields == 9.0)
    assert torch.all(batch.target_fields == 1.0)
    assert batch.input_channel_names == ["uT"]
    assert batch.target_channel_names == ["u0"]


def test_burger_nctx_is_reported_as_a_loaded_full_trajectory():
    registry = build_default_registry()
    raw = registry.synthetic_raw("burger", n=2, resolution=8)
    batch = registry.make_task(raw, "burger", "sparse_solution", num_sensors=5, sensor_mode="fixed")
    dataset = registry.make_dataset_from_batch(batch)

    assert raw["metadata"]["loaded_full_trajectory"] is True
    assert dataset.loaded_full_trajectory is True


def test_burger_loader_reports_nctx_trajectory_even_when_request_flag_is_false(tiny_data_root):
    registry = build_default_registry()
    raw = registry.load_raw(
        "burger",
        tiny_data_root,
        split="train",
        max_samples=1,
        load_full_trajectory=False,
    )

    assert raw["metadata"]["canonical_layout"] == "NCTX"
    assert raw["metadata"]["load_full_trajectory"] is False
    assert raw["metadata"]["loaded_full_trajectory"] is True


@pytest.mark.parametrize("pde", ["poisson", "helmholtz", "darcy", "nsnonbounded"])
def test_sparse_solution_jointly_observes_and_reconstructs_input_and_terminal_solution(pde):
    registry = build_default_registry()
    raw = registry.synthetic_raw(pde, n=2, resolution=8, split="train")
    batch = registry.make_task(
        raw,
        pde,
        "sparse_solution",
        num_sensors=5,
        sensor_mode="fixed",
        experiment_mode="paper",
    )

    assert batch.target_fields.shape[1] == 2
    assert batch.input_fields.shape == batch.target_fields.shape
    assert batch.obs_values.shape == (2, 5, 2)
    assert len(batch.target_channel_names) == 2
    assert batch.metadata["joint_reconstruction"] is True


def test_navier_stokes_full_tasks_use_terminal_state_not_flattened_trajectory():
    registry = build_default_registry()
    raw = registry.synthetic_raw("nsnonbounded", n=2, resolution=8)
    raw["full_tensor"][:, :, 0] = 1.0
    raw["full_tensor"][:, :, -1] = 9.0

    forward = registry.make_task(raw, "nsnonbounded", "forward")
    inverse = registry.make_task(raw, "nsnonbounded", "inverse")

    assert forward.input_fields.shape == forward.target_fields.shape == (2, 1, 8, 8)
    assert torch.all(forward.input_fields == 1.0)
    assert torch.all(forward.target_fields == 9.0)
    assert torch.all(inverse.input_fields == 9.0)
    assert torch.all(inverse.target_fields == 1.0)


def test_burger_time_slice_mode_reconstructs_the_full_trajectory():
    registry = build_default_registry()
    raw = registry.synthetic_raw("burger", n=2, resolution=8, split="train")
    batch = registry.make_task(
        raw,
        "burger",
        "sparse_solution",
        num_sensors=3,
        sensor_mode="time_slices_per_sample",
        experiment_mode="paper",
    )
    dataset = registry.make_dataset_from_batch(batch)
    materialized = pde_collate([dataset[0], dataset[1]])

    assert materialized.target_fields.shape == (2, 1, 8, 8)
    assert materialized.obs_values.shape == (2, 24, 1)
    assert materialized.metadata["num_observations_total"] == 24
    assert materialized.metadata["joint_split_axis"] == 2
    assert materialized.metadata["joint_input_extent"] == 1
    assert materialized.metadata["joint_solution_extent"] == 7


def test_missing_file_error_lists_pde_and_candidates(tmp_path):
    registry = build_default_registry()
    with pytest.raises(FileNotFoundError) as exc:
        registry.load_raw("heat", tmp_path, split="test", max_samples=1)
    message = str(exc.value)
    assert "heat" in message
    assert "Candidate paths" in message
