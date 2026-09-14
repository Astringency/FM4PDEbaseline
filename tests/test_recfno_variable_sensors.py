from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader

from baselines.common.data_adapter import PDEBatchDataset, build_default_registry, pde_collate
from baselines.common.normalization import estimate_normalization_stats, normalize_batch_input_target
from baselines.methods.recfno import RecFNOBaseline
from baselines.run import build_data_spec
from scripts.run_recfno_variable_sensors import make_training_loader, parse_counts


def source_loader(workers=0, shuffle=False):
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=19, resolution=8, split="train", seed=9)
    batch = registry.make_task(raw, "poisson", "sparse_solution_multicondition", num_sensors=5,
                              sensor_mode="random_per_sample", sensor_budget_mode="total",
                              condition_mode="mixed", seed=1)
    return DataLoader(PDEBatchDataset(batch), batch_size=4, shuffle=shuffle, collate_fn=pde_collate,
                      num_workers=workers, persistent_workers=workers > 0)


def snapshot(loader, epoch):
    loader.dataset.set_epoch(epoch)
    return [(b.global_sample_ids, b.metadata["condition_modes"], b.obs_values.shape[1],
             b.metadata["base_mask_ids"]) for b in loader]


def test_counts_reproducible_across_epochs_and_persistent_workers():
    expected = make_training_loader(source_loader(), (2, 5, 11), seed=1)
    workers = make_training_loader(source_loader(2), (2, 5, 11), seed=1)
    try:
        first = snapshot(expected, 0)
        assert first == snapshot(workers, 0)
        second = snapshot(expected, 1)
        assert second == snapshot(workers, 1)
        assert first != second
        assert len({row[2] for row in first + second}) == 3
        assert first == snapshot(workers, 0)
    finally:
        if workers._iterator is not None:
            workers._iterator._shutdown_workers()


def test_fixed_count_exactly_matches_original_inputs_and_shuffle():
    original = source_loader(shuffle=True)
    wrapped = make_training_loader(source_loader(shuffle=True), (5,), seed=1)
    torch.manual_seed(43)
    old = list(original)
    torch.manual_seed(43)
    new = list(wrapped)
    for left, right in zip(old, new):
        assert left.global_sample_ids == right.global_sample_ids
        for name in ("input_fields", "target_fields", "mask", "obs_values", "obs_coords"):
            assert torch.equal(getattr(left, name), getattr(right, name))
        assert torch.equal(left.metadata["voronoi_grid"], right.metadata["voronoi_grid"])


def test_count_sampling_does_not_change_sample_order_or_conditions():
    fixed = make_training_loader(source_loader(shuffle=True), (5,), seed=1)
    mixed = make_training_loader(source_loader(shuffle=True), (2, 5, 11), seed=1)
    torch.manual_seed(71)
    first = snapshot(fixed, 4)
    torch.manual_seed(71)
    second = snapshot(mixed, 4)
    assert [(x[0], x[1]) for x in first] == [(x[0], x[1]) for x in second]


def test_batch_budget_presence_normalization_and_model_gradients():
    original = source_loader()
    loader = make_training_loader(original, (2, 5, 11), seed=1)
    stats = estimate_normalization_stats(loader)
    for batch in loader:
        n = batch.metadata["train_sensor_count"]
        assert torch.all(batch.metadata["base_mask"][:, 0].sum((1, 2)) == n)
        for index, condition in enumerate(batch.metadata["condition_modes"]):
            expected = {"a_only": [n, 0], "u_only": [0, n], "both": [n, n]}[condition]
            assert batch.mask[index].sum((1, 2)).tolist() == expected
        normalized = normalize_batch_input_target(batch, stats)
        observed = normalized.mask.bool()
        assert torch.allclose(normalized.input_fields[observed], normalized.target_fields[observed])
        model = RecFNOBaseline().build(dict(implementation_mode="adapted", official_backend="local",
                                           width=4, modes1=2, modes2=2), build_data_spec(batch))
        loss = (model.predict(normalized) - normalized.target_fields).abs().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert original.dataset.batch.metadata["num_sensors"] == 5


def test_unobserved_truth_never_enters_variable_count_inputs():
    loader = make_training_loader(source_loader(), (5,), seed=1)
    key = (0, 5, 3, 0)
    first = loader.dataset[key]
    changed = loader.dataset.batch.target_fields.clone()
    changed[0][~first.metadata["base_mask"][0].bool()] += 12345
    replacement = PDEBatchDataset(replace(loader.dataset.batch, target_fields=changed))
    second = make_training_loader(DataLoader(replacement, batch_size=4, collate_fn=pde_collate), (5,), 1).dataset[key]
    for name in ("input_fields", "obs_values", "mask"):
        assert torch.equal(getattr(first, name), getattr(second, name))
    assert torch.equal(first.metadata["voronoi_grid"], second.metadata["voronoi_grid"])


@pytest.mark.parametrize("value", ["0,5", "5,5", "-1,2"])
def test_invalid_counts(value):
    with pytest.raises(ValueError):
        parse_counts(value)
