from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ALL_PDES = {
    "darcy",
    "poisson",
    "helmholtz",
    "nsnonbounded",
    "burger",
    "reaction_diffusion",
    "shallow_water",
    "heat",
    "wave",
    "advection_diffusion",
    "steady_heat_conduction",
}


def test_main_results_config_covers_expected_pdes_and_resources():
    cfg = yaml.safe_load((ROOT / "configs" / "experiments" / "main_results.yaml").read_text(encoding="utf-8"))
    assert set(cfg["pdes"]) == ALL_PDES
    assert cfg["data_loading_mode"] == "eager"
    assert cfg["num_workers"] == 4
    assert cfg["pin_memory"] is True
    assert cfg["persistent_workers"] is True
    assert cfg["prefetch_factor"] == 2
    assert "DiffusionPDE" not in cfg["pdes"]
    assert "fm4pde" not in json.dumps(cfg).lower()
    resources = cfg["resources"]
    assert resources["amortized_default"]["batch_size"] == 16
    assert resources["amortized_default"]["epochs"] == 200
    assert resources["per_instance_default"]["batch_size"] == 1
    assert resources["pinn_sparse"]["steps"] == 1000
    assert resources["pc_bnn"]["particles"] == 8
    assert resources["var4d"]["load_full_trajectory"] is True
    assert resources["vivid"]["refine_steps"] == 300


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
    env = {**os.environ, "OUT_ROOT": str(out_root)}
    subprocess.run(["bash", "scripts/experiments/06_aggregate_main_results.sh"], cwd=ROOT, env=env, check=True)
    assert (out_root / "aggregate" / "main_results" / "summary.csv").exists()
    all_summary = json.loads((out_root / "aggregate" / "main_results" / "summary.json").read_text(encoding="utf-8"))
    assert any(row["pde"] == "heat" and row["baseline"] == "fno" for row in all_summary)
