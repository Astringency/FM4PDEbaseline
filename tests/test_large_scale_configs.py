from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ALL_PDES = {
    "darcy",
    "poisson",
    "helmholtz",
    "nsnonbounded",
    "burger",
}


def test_main_results_config_covers_expected_pdes_and_resources():
    cfg = yaml.safe_load((ROOT / "configs" / "experiments" / "main_results.yaml").read_text(encoding="utf-8"))
    assert set(cfg["pdes"]) == ALL_PDES
    assert cfg["data_loading_mode"] == "eager"
    assert cfg["num_workers"] == 4
    assert cfg["pin_memory"] is True
    assert cfg["persistent_workers"] is True
    assert cfg["prefetch_factor"] == 2
    assert cfg["train_size"] == 50000
    assert cfg["val_size"] == 1000
    assert cfg["test_size"] == 1000
    assert "DiffusionPDE" not in cfg["pdes"]
    configured_baselines = {
        baseline
        for group in cfg["task_group_overrides"].values()
        for baseline in group.get("baselines", [])
    }
    assert "fm4pde" not in configured_baselines
    resources = cfg["resources"]
    assert resources["amortized_default"]["batch_size"] == 16
    assert resources["amortized_default"]["epochs"] == 200
    assert resources["per_instance_default"]["batch_size"] == 1
    assert resources["pinn_sparse"]["steps"] == 1000
    assert resources["pc_bnn"]["particles"] == 5
    assert resources["pc_bnn"]["steps"] == 2000
    assert resources["pde_opt"]["steps"] == 500


def test_paper_baseline_config_uses_eager_multi_worker_loading():
    cfg = yaml.safe_load((ROOT / "baselines" / "configs" / "paper.yaml").read_text(encoding="utf-8"))
    assert cfg["data_loading_mode"] == "eager"
    assert cfg["num_workers"] == 4
    assert cfg["pin_memory"] is True
    assert cfg["persistent_workers"] is True
    assert cfg["prefetch_factor"] == 2


def test_aggregate_script_finds_results_files(tmp_path: Path):
    out_root = tmp_path / "large"
    result_dir = out_root / "runs" / "main_results" / "task_group=full_forward_main" / "dummy"
    result_dir.mkdir(parents=True)
    row = {
        "pde": "heat",
        "task": "forward",
        "baseline": "fno",
        "train_size": 128,
        "scalar_param_mode": "materialize",
        "num_sensors": 0,
        "sensor_mode": "none",
        "noise_level": 0.0,
        "backend_used": "local",
        "relative_l2_solution_mean": 1.0,
        "relative_l2_solution_std": 0.0,
        "relative_l2_solution_n": 1,
        "mse_mean": 1.0,
        "mse_std": 0.0,
        "mse_n": 1,
        "residual_mode_counts": "{}",
    }
    (result_dir / "results_summary.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "baselines.aggregate_results",
            str(result_dir),
            "--output-dir",
            str(out_root / "aggregate" / "main_results"),
        ],
        cwd=ROOT,
        check=True,
    )
    assert (out_root / "aggregate" / "main_results" / "summary.csv").exists()
    all_summary = json.loads((out_root / "aggregate" / "main_results" / "summary.json").read_text(encoding="utf-8"))
    assert any(row["pde"] == "heat" and row["baseline"] == "fno" for row in all_summary)
