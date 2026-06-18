from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines import experiment_matrix


def test_inverse_paper_scripts_exist_and_call_runner():
    root = Path(__file__).resolve().parents[1]
    full = root / "scripts/baselines/run_paper_full_inverse.sh"
    sparse = root / "scripts/baselines/run_paper_sparse_inverse.sh"
    assert full.exists()
    assert sparse.exists()
    assert "--task inverse" in full.read_text(encoding="utf-8")
    assert "--task sparse_inverse" in sparse.read_text(encoding="utf-8")
    assert "python -m baselines.run" in full.read_text(encoding="utf-8")
    assert "python -m baselines.run" in sparse.read_text(encoding="utf-8")


def test_sparse_inverse_unsupported_combo_records_skip(tmp_path):
    skipped = tmp_path / "skipped_combinations.jsonl"
    with pytest.raises(SystemExit):
        experiment_matrix.main(
            [
                "--baseline",
                "pde_opt",
                "--pde",
                "poisson",
                "--task",
                "sparse_inverse",
                "--skipped-path",
                str(skipped),
            ]
        )
    row = json.loads(skipped.read_text(encoding="utf-8").strip())
    assert row["task"] == "sparse_inverse"
    assert "forward solve" in row["reason"]
