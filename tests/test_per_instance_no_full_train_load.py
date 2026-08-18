from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_pde_opt_sparse_forward_does_not_load_full_train_for_per_instance_spec(tmp_path: Path):
    out = tmp_path / "per_instance"
    cmd = [
        sys.executable,
        "-m",
        "baselines.run",
        "--baseline",
        "pde_opt",
        "--pde",
        "darcy",
        "--task",
        "sparse_forward",
        "--experiment-mode",
        "debug",
        "--synthetic-data",
        "--synthetic-resolution",
        "8",
        "--train-size",
        "50000",
        "--val-size",
        "0",
        "--test-size",
        "1",
        "--batch-size",
        "2",
        "--num-sensors",
        "4",
        "--sensor-mode",
        "random_per_sample",
        "--epochs",
        "1",
        "--output-dir",
        str(out),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    summary = json.loads((out / "results_summary.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert summary["train_requested_size"] == 50000
    assert summary["train_size_loaded_for_spec"] <= 4
    assert summary["train_size_loaded_for_fit"] == 0
    assert summary["train_size"] == 0
    assert summary["test_time_optimization"] is True
    assert summary["inference_optimization_time_total"] > 0.0
