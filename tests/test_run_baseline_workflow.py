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
