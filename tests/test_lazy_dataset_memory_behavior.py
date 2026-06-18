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
