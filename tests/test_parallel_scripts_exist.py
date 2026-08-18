from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


SCRIPTS = [
    "scripts/experiments/07_build_main_results_matrix.sh",
    "scripts/experiments/08_run_matrix_2gpu_parallel.sh",
    "scripts/experiments/09_run_main_results_2gpu.sh",
    "scripts/experiments/10_run_first8_2gpu_smoke.sh",
    "scripts/experiments/11_progress_main_results.sh",
    "scripts/experiments/12_clean_stale_run_locks.sh",
]


def test_parallel_scripts_exist_and_pass_bash_syntax_check():
    for script in SCRIPTS:
        assert (ROOT / script).exists(), script
        subprocess.run(["bash", "-n", script], cwd=ROOT, check=True)


def test_2gpu_readme_mentions_key_controls():
    readme = ROOT / "scripts/experiments/README_2gpu_parallel.md"
    assert readme.exists()
    text = readme.read_text(encoding="utf-8")
    for token in ["JOBS_PER_GPU", "CUDA_VISIBLE_DEVICES", "fingerprint", "SAVE_CHECKPOINT"]:
        assert token in text


def test_2gpu_scripts_expose_save_checkpoint_control():
    for script in [
        "scripts/experiments/08_run_matrix_2gpu_parallel.sh",
        "scripts/experiments/09_run_main_results_2gpu.sh",
    ]:
        text = (ROOT / script).read_text(encoding="utf-8")
        assert "SAVE_CHECKPOINT" in text
