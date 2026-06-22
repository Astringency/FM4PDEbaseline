from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest
import yaml

from baselines.common.data_adapter import PDEBatchDataset, build_default_registry
from baselines.run import (
    _check_loaded_dataset_memory_limit,
    dataset_memory_result_fields,
    dataset_tensor_memory_summary,
    main,
    print_dataset_memory_summary,
)


def test_dataset_memory_summary_fields_are_non_negative(capsys):
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=2, resolution=8, seed=3)
    dataset = PDEBatchDataset(registry.make_task(raw, "poisson", "forward"))

    summary = dataset_tensor_memory_summary(dataset)
    fields = dataset_memory_result_fields(dataset, None, dataset)
    print_dataset_memory_summary("train", dataset)
    captured = capsys.readouterr()

    assert "[dataset memory] split=train" in captured.err
    for key in ("full_tensor_gb", "input_fields_gb", "target_fields_gb", "tensor_memory_gb"):
        assert summary[key] >= 0.0
    for key in (
        "train_tensor_memory_gb",
        "val_tensor_memory_gb",
        "test_tensor_memory_gb",
        "train_full_tensor_memory_gb",
        "train_input_tensor_memory_gb",
        "train_target_tensor_memory_gb",
    ):
        assert key in fields
        assert fields[key] >= 0.0


def test_dataset_memory_limit_is_optional_but_enforced_when_set():
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=2, resolution=8, seed=4)
    dataset = PDEBatchDataset(registry.make_task(raw, "poisson", "forward"))
    fields = dataset_memory_result_fields(dataset, None, dataset)

    _check_loaded_dataset_memory_limit(Namespace(max_loaded_dataset_gb=None), fields)
    with pytest.raises(RuntimeError, match="exceeds max_loaded_dataset_gb"):
        _check_loaded_dataset_memory_limit(Namespace(max_loaded_dataset_gb=1e-12), fields)


def test_run_writes_dataset_memory_fields_and_enforces_cli_limit(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "epochs": 1,
                "method": {
                    "implementation_mode": "adapted",
                    "official_backend": "local",
                    "hidden": 8,
                    "basis": 4,
                    "max_steps": 1,
                    "max_val_steps": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    args = [
        "--baseline",
        "deeponet",
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
        "2",
        "--val-size",
        "1",
        "--test-size",
        "1",
        "--batch-size",
        "1",
        "--num-workers",
        "0",
        "--output-dir",
        str(out),
    ]

    main(args)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    raw = json.loads((out / "results_raw.jsonl").read_text(encoding="utf-8").splitlines()[0])
    config_snapshot = json.loads(Path(summary["config_path"]).read_text(encoding="utf-8"))
    for row in (summary, raw, config_snapshot):
        for key in (
            "train_tensor_memory_gb",
            "val_tensor_memory_gb",
            "test_tensor_memory_gb",
            "train_full_tensor_memory_gb",
            "train_input_tensor_memory_gb",
            "train_target_tensor_memory_gb",
        ):
            assert key in row
            assert row[key] >= 0.0

    with pytest.raises(RuntimeError, match="exceeds max_loaded_dataset_gb"):
        main(args[:-2] + ["--output-dir", str(tmp_path / "limited"), "--max-loaded-dataset-gb", "1e-12"])
