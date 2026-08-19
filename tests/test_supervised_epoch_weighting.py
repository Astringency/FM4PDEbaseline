from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from baselines.common.data_adapter import PDEBatchDataset, build_default_registry, pde_collate
from baselines.methods.base import BaselineModel, run_supervised_fit
from baselines.run import build_data_spec


class _ConstantModel(BaselineModel):
    name = "constant"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        self.value = torch.nn.Parameter(torch.tensor(0.0))
        return self

    def predict(self, batch):
        return self.value.expand_as(batch.target_fields)


def test_epoch_losses_are_weighted_by_sample_count_for_short_last_batch():
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=3, resolution=4, seed=1)
    batch = registry.make_task(raw, "poisson", "forward")
    batch.target_fields = torch.zeros_like(batch.target_fields)
    batch.target_fields[2].fill_(3.0)
    loader = DataLoader(PDEBatchDataset(batch), batch_size=2, shuffle=False, collate_fn=pde_collate)
    model = _ConstantModel().build(
        {"epochs": 1, "lr": 0.0, "normalize": False, "device": "cpu"},
        build_data_spec(batch),
    )

    history = run_supervised_fit(model, loader, loader)

    # Per-sample losses are [0, 0, 9], so the correct epoch mean is 3,
    # not the unweighted mean of batch means (0 + 9) / 2 = 4.5.
    assert history["train_loss"] == pytest.approx([3.0])
    assert history["val_loss"] == pytest.approx([3.0])
