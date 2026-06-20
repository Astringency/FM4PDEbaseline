from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from baselines.common.data_adapter import PDEBatchDataset, build_default_registry, pde_collate
from baselines.common.normalization import denormalize_prediction, estimate_normalization_stats, normalize_batch_input_target
from baselines.run import main


def test_normalize_batch_keeps_sparse_observation_shapes():
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=3, resolution=8)
    batch = registry.make_task(raw, "poisson", "sparse_solution", num_sensors=5, seed=1)
    loader = DataLoader(PDEBatchDataset(batch), batch_size=2, collate_fn=pde_collate)
    stats = estimate_normalization_stats(loader)
    norm_batch = normalize_batch_input_target(batch, stats)
    restored = denormalize_prediction(norm_batch.target_fields, stats)
    assert tuple(norm_batch.obs_values.shape) == tuple(batch.obs_values.shape)
    assert tuple(norm_batch.mask.shape) == tuple(batch.mask.shape)
    assert torch.isfinite(norm_batch.obs_values).all()
    assert tuple(restored.shape) == tuple(batch.target_fields.shape)


def test_normalized_fno_tiny_run_saves_stats_and_best_val(tmp_path: Path):
    config = tmp_path / "norm.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "method": {
                    "implementation_mode": "adapted",
                    "official_backend": "local",
                    "normalize": True,
                    "width": 8,
                    "modes1": 4,
                    "modes2": 4,
                    "layers": 1,
                    "max_steps": 1,
                    "max_val_steps": 1,
                },
                "epochs": 1,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    main(
        [
            "--baseline",
            "fno",
            "--pde",
            "poisson",
            "--task",
            "forward",
            "--experiment-mode",
            "debug",
            "--synthetic-data",
            "--synthetic-resolution",
            "8",
            "--config",
            str(config),
            "--train-size",
            "4",
            "--val-size",
            "2",
            "--test-size",
            "2",
            "--batch-size",
            "2",
            "--epochs",
            "1",
            "--save-checkpoint",
            "--output-dir",
            str(out),
        ]
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["normalize"] is True
    assert summary["best_val_loss"] is not None
    assert Path(summary["normalization_stats_path"]).exists()
    ckpt = torch.load(summary["checkpoint_path"], map_location="cpu")
    assert ckpt["uses_normalization"] is True
    assert ckpt["normalization_stats"]["input_mean"].numel() == 1
