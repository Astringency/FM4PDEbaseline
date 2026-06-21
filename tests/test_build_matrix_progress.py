from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_build_matrix_cli_progress_is_on_stderr_and_summary_stays_stdout(tmp_path: Path):
    out = tmp_path / "large"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/build_matrix.py",
            "--config",
            "configs/experiments/sanity_main.yaml",
            "--output-root",
            str(out),
            "--matrix-name",
            "sanity_main",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    summary = json.loads(proc.stdout)
    rows = _read_jsonl(out / "matrices" / "sanity_main.jsonl")
    skipped = _read_jsonl(out / "skipped_combinations.jsonl")
    assert summary["run_count"] == len(rows)
    assert summary["skipped_combo_count"] == len(skipped)
    assert "[matrix start]" in proc.stderr
    assert "[matrix group]" in proc.stderr
    assert "[matrix complete]" in proc.stderr
    assert "[matrix outputs]" in proc.stderr
    assert "[matrix start]" not in proc.stdout
