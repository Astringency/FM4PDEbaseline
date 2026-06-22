from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from torch.utils.data import DataLoader

from baselines.common.data_adapter import PDEBatch, PDEBatchDataset, pde_collate
from baselines.methods.base import BaselineModel, restore_state_dict, run_supervised_fit, snapshot_state_dict


def test_run_supervised_fit_handles_non_tensor_state_dict_entries():
    model = _NonTensorStateModel().build({"epochs": 1, "lr": 0.01}, {})
    loader = _loader()

    with pytest.warns(RuntimeWarning, match="Non-tensor keys skipped"):
        history = run_supervised_fit(model, loader, loader)

    assert history["best_epoch"] == 1
    assert history["best_val_loss"] is not None


def test_run_supervised_fit_logs_epochs_and_incremental_history(tmp_path, capsys):
    history_json = tmp_path / "run_train_history.json"
    history_jsonl = tmp_path / "run_train_history.jsonl"
    model = _StrictRecordingModel().build(
        {
            "epochs": 2,
            "lr": 0.01,
            "train_history_json_path": str(history_json),
            "train_history_jsonl_path": str(history_jsonl),
        },
        {"pde": "poisson", "task": "forward"},
    )
    loader = _loader()

    history = run_supervised_fit(model, loader, loader)
    captured = capsys.readouterr()

    assert "[fit epoch]" in captured.err
    assert "epoch=1/2" in captured.err
    assert "epoch=2/2" in captured.err
    assert "train_loss=" in captured.err
    assert "val_loss=" in captured.err
    assert history_json.exists()
    assert history_jsonl.exists()
    assert len(history_jsonl.read_text(encoding="utf-8").strip().splitlines()) == 2
    assert history["best_val_loss"] is not None


def test_run_supervised_fit_uses_strict_restore_for_tensor_only_models():
    model = _StrictRecordingModel().build({"epochs": 1, "lr": 0.01}, {})
    loader = _loader()

    run_supervised_fit(model, loader, loader)

    assert model.load_strict_values == [True]


def test_snapshot_state_dict_deepcopies_non_tensor_entries_without_detach():
    model = _NonTensorStateModel()
    snapshot = snapshot_state_dict(model)

    assert isinstance(snapshot, OrderedDict)
    assert torch.equal(snapshot["weight"], model.weight.detach().cpu())
    assert snapshot["metadata"]["payload"].deepcopy_count == 1
    assert snapshot["metadata"]["payload"].detach_count == 0


def test_restore_state_dict_fallback_warns_and_skips_non_tensor_entries():
    model = _NonTensorStateModel()
    state = snapshot_state_dict(model)

    with pytest.warns(RuntimeWarning, match="tensor-only strict=False restore"):
        restore_state_dict(model, state)


class _DetachTrap:
    def __init__(self) -> None:
        self.detach_count = 0
        self.deepcopy_count = 0

    def detach(self):
        self.detach_count += 1
        raise AssertionError("detach should not be called for non-tensor state_dict values")

    def __deepcopy__(self, memo):
        copied = type(self)()
        copied.deepcopy_count = self.deepcopy_count + 1
        memo[id(self)] = copied
        return copied


class _NonTensorStateModel(BaselineModel):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.payload = _DetachTrap()

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["metadata"] = {"payload": self.payload}
        return state

    def predict(self, batch: PDEBatch):
        return batch.input_fields * self.weight


class _StrictRecordingModel(BaselineModel):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.load_strict_values: list[bool] = []

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.load_strict_values.append(strict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def predict(self, batch: PDEBatch):
        return batch.input_fields * self.weight


def _loader() -> DataLoader:
    batch = _batch()
    return DataLoader(PDEBatchDataset(batch), batch_size=2, collate_fn=pde_collate)


def _batch() -> PDEBatch:
    input_fields = torch.ones(2, 1, 2, 2)
    target_fields = torch.full((2, 1, 2, 2), 2.0)
    coords = torch.stack(torch.meshgrid(torch.linspace(0, 1, 2), torch.linspace(0, 1, 2), indexing="ij"), dim=-1)
    coords = coords.reshape(1, -1, 2).repeat(2, 1, 1)
    return PDEBatch(
        pde_name="poisson",
        task="forward",
        full_tensor=torch.cat([input_fields, target_fields], dim=1),
        input_fields=input_fields,
        target_fields=target_fields,
        coords=coords,
        mask=None,
        obs_values=None,
        obs_coords=None,
        channel_names=["input", "target"],
        input_channel_names=["input"],
        target_channel_names=["target"],
        metadata={},
        pde_params={},
        split="train",
        sample_indices=torch.arange(2),
        global_sample_ids=["0", "1"],
        file_paths=[],
    )
