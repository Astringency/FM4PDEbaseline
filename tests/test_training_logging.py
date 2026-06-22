from __future__ import annotations

import json
from pathlib import Path

import yaml

from baselines.run import main


def test_synthetic_debug_run_logs_epoch_losses_and_completed_history(tmp_path: Path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "epochs": 2,
                "learning_rate": 0.001,
                "method": {
                    "implementation_mode": "adapted",
                    "official_backend": "local",
                    "hidden": 8,
                    "basis": 4,
                    "max_steps": 1,
                    "max_val_steps": 1,
                    "normalize": False,
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    main(
        [
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
            "4",
            "--val-size",
            "2",
            "--test-size",
            "1",
            "--batch-size",
            "2",
            "--epochs",
            "2",
            "--num-workers",
            "0",
            "--output-dir",
            str(out),
        ]
    )
    captured = capsys.readouterr()

    assert "[fit epoch]" in captured.err
    assert "train_loss=" in captured.err
    assert "val_loss=" in captured.err
    history_json = next(out.glob("*_train_history.json"))
    history_jsonl = next(out.glob("*_train_history.jsonl"))
    history = json.loads(history_json.read_text(encoding="utf-8"))
    assert history["completed_epochs"] == 2
    assert history_jsonl.exists()
    assert len(history_jsonl.read_text(encoding="utf-8").strip().splitlines()) == 2
