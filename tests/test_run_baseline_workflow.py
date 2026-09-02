from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_full_workflow_dry_run_lists_steps_in_order(tmp_path: Path):
    env = {
        **os.environ,
        "DATA_ROOT": str(tmp_path),
        "OUT_ROOT": str(tmp_path / "outputs"),
        "GPUS": "2,3",
        "JOBS_PER_GPU": "2",
        "DRY_RUN": "1",
    }

    result = subprocess.run(
        ["bash", "scripts/run_baseline.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    steps = [
        "scripts/verify_data_protocol.py",
        "scripts/build_experiment_matrix.py",
        "scripts/run_experiments.py",
        "scripts/collect_results.py",
        "scripts/plot_results.py",
    ]
    positions = [output.index(step) for step in steps]
    assert positions == sorted(positions)
    assert "--gpus 2\\,3" in output or "--gpus 2,3" in output
    assert "--jobs-per-gpu 2" in output
    assert "--rerun-running" in output
    assert "--max-samples 100" in output


def test_multicondition_ablation_workflow_dry_run_is_two_phase(tmp_path: Path):
    env = {
        **os.environ,
        "DATA_ROOT": str(tmp_path),
        "OUT_ROOT": str(tmp_path / "ablation"),
        "GPUS": "2,3",
        "JOBS_PER_GPU": "2",
        "DRY_RUN": "1",
    }
    env.pop("PDE_LIST", None)

    result = subprocess.run(
        ["bash", "scripts/run_baseline_ablations.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "sparse_solution_multicondition_ablation.yaml" in output
    assert "--indices 0-11" in output
    assert "--indices 12-47" in output
    assert output.count("scripts/build_experiment_matrix.py") == 2
    assert "scripts/build_sparse_solution_multicondition_report.py" in output
    assert output.index("--indices 0-11") < output.index("--indices 12-47")


def test_multicondition_ablation_workflow_accepts_single_pde(tmp_path: Path):
    env = {
        **os.environ,
        "DATA_ROOT": str(tmp_path),
        "OUT_ROOT": str(tmp_path / "ablation"),
        "PDE_LIST": "poisson",
        "DRY_RUN": "1",
    }

    result = subprocess.run(
        ["bash", "scripts/run_baseline_ablations.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "PDE_LIST=poisson" in output
    assert "workflow=3 training rows -> checkpoint binding -> 9 eval-only rows" in output
    assert "--pde poisson" in output
    assert "--indices 0-2" in output
    assert "--indices 3-11" in output
    assert output.index("--indices 0-2") < output.index("--indices 3-11")
