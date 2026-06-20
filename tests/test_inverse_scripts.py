from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines import experiment_matrix
from scripts.experiments.build_matrix import build_matrix, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_main_results_matrix_includes_inverse_task_families(tmp_path):
    cfg = load_config(ROOT / "configs" / "experiments" / "main_results.yaml")
    rows, _skipped, _summary = build_matrix(cfg, tmp_path / "main_results", "main_results")
    assert any(row["task"] == "inverse" for row in rows)
    assert any(row["task"] == "sparse_inverse" for row in rows)


def test_sparse_inverse_unsupported_combo_records_skip(tmp_path):
    skipped = tmp_path / "skipped_combinations.jsonl"
    with pytest.raises(SystemExit):
        experiment_matrix.main(
            [
                "--baseline",
                "fno",
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
    assert row["capability_status"] == "unsupported"
    assert "sparse-sensor" in row["reason"]
