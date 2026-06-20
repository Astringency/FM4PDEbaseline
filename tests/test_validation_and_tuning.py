from __future__ import annotations

import json
from pathlib import Path

from baselines.aggregate_results import main as aggregate_main
from baselines.common.data_adapter import LazyPDEBatchDataset, build_default_registry


def test_deterministic_train_tail_validation_is_not_test(tiny_data_root):
    registry = build_default_registry()
    val = registry.make_dataset(
        "poisson",
        tiny_data_root,
        "forward",
        split="val",
        max_samples=1,
        val_from_train_offset=2,
        data_loading_mode="lazy",
        strict_size=True,
    )
    test = registry.make_dataset(
        "poisson",
        tiny_data_root,
        "forward",
        split="test",
        max_samples=1,
        data_loading_mode="lazy",
    )
    assert isinstance(val, LazyPDEBatchDataset)
    assert val.batch.metadata["split_source"] == "deterministic_train_subset"
    assert "test" not in val.batch.global_sample_ids[0].lower()
    assert "test" in test.batch.file_paths[0].lower()


def test_aggregator_writes_tuning_summary(tmp_path: Path):
    raw = tmp_path / "results_raw.jsonl"
    rows = [
        {
            "pde": "poisson",
            "task": "forward",
            "baseline": "fno",
            "sensor_mode": "none",
            "sensor_budget_mode": "none",
            "num_sensors": 0,
            "noise_level": 0.0,
            "implementation_mode_effective": "adapted",
            "best_val_loss": 0.5,
            "best_epoch": 1,
            "config_hash": "a",
            "config_path": "a.json",
            "relative_l2_solution": 1.0,
            "mse": 1.0,
            "mae": 1.0,
        },
        {
            "pde": "poisson",
            "task": "forward",
            "baseline": "fno",
            "sensor_mode": "none",
            "sensor_budget_mode": "none",
            "num_sensors": 0,
            "noise_level": 0.0,
            "implementation_mode_effective": "adapted",
            "best_val_loss": 0.25,
            "best_epoch": 2,
            "config_hash": "b",
            "config_path": "b.json",
            "relative_l2_solution": 0.8,
            "mse": 0.8,
            "mae": 0.8,
        },
    ]
    raw.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    out = tmp_path / "agg"
    aggregate_main([str(raw), "--output-dir", str(out)])
    tuning = json.loads((out / "tuning_summary.json").read_text(encoding="utf-8"))
    assert tuning[0]["config_hash"] == "b"
    assert tuning[0]["selected_config_path"] == "b.json"
