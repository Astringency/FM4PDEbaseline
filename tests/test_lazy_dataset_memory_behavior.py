from __future__ import annotations

from baselines.common.data_adapter import LazyPDEBatchDataset, PDEBatch, build_default_registry


def test_lazy_dataset_len_and_getitem_without_eager_sample_reads(tiny_data_root):
    registry = build_default_registry()
    ds = registry.make_dataset(
        "reaction_diffusion",
        tiny_data_root,
        "forward",
        split="train",
        max_samples=3,
        data_loading_mode="lazy",
        load_full_trajectory=False,
    )
    assert isinstance(ds, LazyPDEBatchDataset)
    assert len(ds) == 3
    assert ds.samples_read == 0

    item = ds[0]
    assert isinstance(item, PDEBatch)
    assert ds.samples_read == 1
    assert item.full_tensor.ndim == 4
    assert item.input_fields.ndim == 4
    assert item.target_fields.ndim == 4
    assert item.metadata["loaded_full_trajectory"] is False


def test_lazy_dataset_can_read_full_trajectory_when_requested(tiny_data_root):
    registry = build_default_registry()
    ds = registry.make_dataset(
        "shallow_water",
        tiny_data_root,
        "forward",
        split="train",
        max_samples=2,
        data_loading_mode="lazy",
        load_full_trajectory=True,
    )
    item = ds[0]
    assert item.full_tensor.ndim == 5
    assert item.metadata["loaded_full_trajectory"] is True


def test_static_pdes_lazy_dataset_avoids_eager_load(tiny_data_root):
    registry = build_default_registry()
    for pde in ("darcy", "poisson", "helmholtz"):
        ds = registry.make_dataset(
            pde,
            tiny_data_root,
            "forward",
            split="train",
            max_samples=2,
            data_loading_mode="lazy",
        )
        assert isinstance(ds, LazyPDEBatchDataset)
        assert ds.loaded_in_memory_samples == 0
        assert ds.samples_read == 0
        item = ds[0]
        assert isinstance(item, PDEBatch)
        assert tuple(item.input_fields.shape[:2]) == (1, 1)
        assert tuple(item.target_fields.shape[:2]) == (1, 1)
        assert item.metadata["data_loading_mode"] == "lazy"
        assert item.metadata["train_size_loaded_in_memory"] == 0


def test_static_lazy_train_tail_validation_offset(tiny_data_root):
    registry = build_default_registry()
    ds = registry.make_dataset(
        "poisson",
        tiny_data_root,
        "forward",
        split="val",
        max_samples=1,
        val_from_train_offset=2,
        data_loading_mode="lazy",
        strict_size=True,
    )
    assert isinstance(ds, LazyPDEBatchDataset)
    assert ds.batch.metadata["split_source"] == "deterministic_train_subset"
    assert ds.batch.metadata["val_from_train_offset"] == 2
