from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baselines.common.data_adapter import PDEBatch, PDEBatchDataset, pde_collate
from baselines.methods.ifno import IFNOBaseline


def test_ifno_fit_writes_history_and_uses_one_based_best_epoch(tmp_path: Path, capsys):
    history_json = tmp_path / "ifno_train_history.json"
    history_jsonl = tmp_path / "ifno_train_history.jsonl"
    model = IFNOBaseline().build(
        {
            "implementation_mode": "adapted",
            "official_backend": "local",
            "device": "cpu",
            "epochs": 2,
            "lr": 1e-3,
            "width": 4,
            "layers": 1,
            "modes1": 2,
            "modes2": 2,
            "cycle_weight": 0.0,
            "max_steps": 1,
            "max_val_steps": 1,
            "normalize": True,
            "lr_scheduler": "reduce_on_plateau",
            "scheduler_patience": 1,
            "early_stopping": False,
            "grad_clip_norm": 1.0,
            "train_history_json_path": str(history_json),
            "train_history_jsonl_path": str(history_jsonl),
        },
        {
            "pde": "poisson",
            "task": "forward",
            "input_channels": 1,
            "target_channels": 1,
        },
    )
    loader = _loader()

    history = model.fit(loader, loader)
    captured = capsys.readouterr()

    assert "[normalization] baseline=ifno pde=poisson task=forward start" in captured.err
    assert "[normalization] baseline=ifno pde=poisson task=forward done" in captured.err
    assert "[fit epoch] baseline=ifno" in captured.err
    assert 1 <= history["best_epoch"] <= 2
    assert history["stop_epoch"] == 2
    assert len(history["lr_history"]) == 2
    assert history_json.exists()
    assert history_jsonl.exists()
    saved = json.loads(history_json.read_text(encoding="utf-8"))
    assert saved["completed_epochs"] == 2


def _loader() -> DataLoader:
    batch = _batch()
    return DataLoader(PDEBatchDataset(batch), batch_size=2, collate_fn=pde_collate)


def _batch() -> PDEBatch:
    input_fields = torch.ones(2, 1, 4, 4)
    target_fields = torch.full((2, 1, 4, 4), 0.5)
    coords = torch.stack(torch.meshgrid(torch.linspace(0, 1, 4), torch.linspace(0, 1, 4), indexing="ij"), dim=-1)
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
