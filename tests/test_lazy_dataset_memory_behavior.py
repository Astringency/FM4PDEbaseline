from __future__ import annotations

from argparse import Namespace

import pytest

from baselines.common.data_adapter import PDEBatchDataset, build_default_registry
from baselines.run import build_pde_dataloader
from baselines.run import _split_mask_manifest


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
def test_make_dataset_always_returns_eager_pde_batch_dataset(tiny_data_root, pde):
    registry = build_default_registry()
    ds = registry.make_dataset(
        pde,
        tiny_data_root,
        "forward",
        split="train",
        max_samples=2,
        data_loading_mode="eager",
        load_full_trajectory=pde in {"nsnonbounded", "reaction_diffusion", "shallow_water"},
    )
    assert isinstance(ds, PDEBatchDataset)
    assert ds.data_loading_mode == "eager"
    assert ds.batch.metadata["data_loading_mode"] == "eager"
    assert ds.batch.metadata["loaded_count"] == len(ds)
    assert len(ds) == 2
    assert ds.batch.full_tensor.shape[0] == 2
    assert ds.batch.input_fields.shape[0] == 2
    assert ds.batch.target_fields.shape[0] == 2
    assert ds.loaded_in_memory_samples == 2


def test_lazy_mode_is_forced_to_eager_outside_paper(tiny_data_root):
    registry = build_default_registry()
    with pytest.warns(RuntimeWarning, match="lazy.*disabled"):
        ds = registry.make_dataset(
            "reaction_diffusion",
            tiny_data_root,
            "forward",
            split="train",
            max_samples=2,
            data_loading_mode="lazy",
            experiment_mode="debug",
            load_full_trajectory=False,
        )
    assert isinstance(ds, PDEBatchDataset)
    assert ds.batch.metadata["data_loading_mode"] == "eager"
    assert ds.loaded_in_memory_samples == 2


def test_lazy_mode_raises_in_paper(tiny_data_root):
    registry = build_default_registry()
    with pytest.raises(ValueError, match="Regenerate.*eager"):
        registry.make_dataset(
            "darcy",
            tiny_data_root,
            "forward",
            split="train",
            max_samples=1,
            data_loading_mode="lazy",
            experiment_mode="paper",
        )


def test_paper_random_per_sample_defers_observation_tensors_until_getitem(tiny_data_root):
    registry = build_default_registry()
    ds = registry.make_dataset(
        "nsnonbounded",
        tiny_data_root,
        "sparse_forward",
        split="train",
        max_samples=2,
        num_sensors=5,
        sensor_mode="random_per_sample",
        load_full_trajectory=True,
        experiment_mode="paper",
    )

    assert ds.batch.mask is None
    assert ds.batch.obs_values is None
    assert ds.batch.obs_coords is None
    assert "masked_grid" not in ds.batch.metadata
    assert "voronoi_grid" not in ds.batch.metadata
    assert ds.batch.metadata["deferred_dynamic_sensors"] is True
    assert ds.batch.coords.shape[0] == 1

    item = ds[0]

    assert item.mask is not None
    assert item.obs_values is not None
    assert item.obs_coords is not None
    assert item.metadata["masked_grid"].shape == item.input_fields.shape
    assert item.metadata["voronoi_grid"].shape == item.input_fields.shape
    assert item.metadata["mask_id"] == ds.batch.metadata["mask_ids"][0]
    assert item.metadata["mask_tensor_sha1"]
    assert ds.batch.mask is None
    manifest = _split_mask_manifest(ds, None, ds)
    assert manifest["train_epoch0"]["sample_mask_count"] == len(ds)
    assert manifest["train_epoch0"]["mask_ids_sha256"]


def test_deferred_random_per_sample_epoch_reaches_persistent_workers(tiny_data_root):
    registry = build_default_registry()
    ds = registry.make_dataset(
        "poisson",
        tiny_data_root,
        "sparse_solution",
        split="train",
        max_samples=3,
        num_sensors=5,
        sensor_mode="random_per_sample",
        seed=3,
        experiment_mode="paper",
    )
    args = Namespace(
        batch_size=3,
        device="cpu",
        num_workers=1,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )
    loader = build_pde_dataloader(ds, args, shuffle=False)
    first = next(iter(loader))
    ds.set_epoch(1)
    second = next(iter(loader))

    assert not first.metadata["mask_id"] == second.metadata["mask_id"]


def test_deferred_sensors_skip_voronoi_when_baseline_does_not_require_it(tiny_data_root):
    registry = build_default_registry()
    ds = registry.make_dataset(
        "poisson",
        tiny_data_root,
        "sparse_solution",
        split="train",
        max_samples=2,
        num_sensors=5,
        sensor_mode="random_per_sample",
        experiment_mode="paper",
        build_voronoi_grid=False,
    )

    item = ds[0]

    assert item.mask is not None
    assert "masked_grid" in item.metadata
    assert "voronoi_grid" not in item.metadata


def test_static_eager_train_tail_validation_offset(tiny_data_root):
    registry = build_default_registry()
    ds = registry.make_dataset(
        "poisson",
        tiny_data_root,
        "forward",
        split="val",
        max_samples=1,
        val_from_train_offset=2,
        data_loading_mode="eager",
        strict_size=True,
    )
    assert isinstance(ds, PDEBatchDataset)
    assert ds.batch.metadata["split_source"] == "deterministic_train_subset"
    assert ds.batch.metadata["val_from_train_offset"] == 2
    assert ds.batch.metadata["data_loading_mode"] == "eager"


def test_build_pde_dataloader_worker_zero_omits_worker_only_options(tiny_data_root):
    ds = build_default_registry().make_dataset("darcy", tiny_data_root, "forward", split="train", max_samples=2)
    args = Namespace(batch_size=2, device="cpu", num_workers=0, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    loader = build_pde_dataloader(ds, args, shuffle=True)
    assert loader.num_workers == 0
    assert loader.pin_memory is False
    assert loader.persistent_workers is False
    assert loader.prefetch_factor is None


def test_build_pde_dataloader_worker_options_and_pin_memory(tiny_data_root):
    ds = build_default_registry().make_dataset("darcy", tiny_data_root, "forward", split="train", max_samples=2)
    args = Namespace(batch_size=2, device="cuda:0", num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=3)
    loader = build_pde_dataloader(ds, args, shuffle=False)
    assert loader.num_workers == 2
    assert loader.pin_memory is True
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 3
