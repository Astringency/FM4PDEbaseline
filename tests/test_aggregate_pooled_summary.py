from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from baselines.aggregate_results import _expand_inputs, _read_jsonl, aggregate_rows


def _summary_row(mean: float, std: float, n: int) -> dict:
    return {
        "pde": "heat",
        "task": "forward",
        "baseline": "fno",
        "train_size": 8,
        "scalar_param_mode": "metadata",
        "num_sensors": 0,
        "sensor_mode": "none",
        "noise_level": 0.0,
        "backend_used": "local",
        "pde_residual_mean": mean,
        "pde_residual_std": std,
        "pde_residual_n": n,
        "pde_residual_nan_count": 0,
    }


def test_summary_only_uses_pooled_statistics():
    summary = aggregate_rows([_summary_row(1.0, 1.0, 2), _summary_row(3.0, 2.0, 3)])[0]
    assert summary["aggregation_mode"] == "pooled_summary"
    assert summary["pde_residual_mean"] == pytest.approx(2.2)
    expected_std = math.sqrt(13.8 / 4.0)
    assert summary["pde_residual_std"] == pytest.approx(expected_std)
    assert summary["pde_residual_n"] == 5


def test_directory_input_prefers_raw_over_summary(tmp_path: Path):
    raw = {
        "pde": "heat",
        "task": "forward",
        "baseline": "fno",
        "train_size": 8,
        "scalar_param_mode": "metadata",
        "num_sensors": 0,
        "sensor_mode": "none",
        "noise_level": 0.0,
        "backend_used": "local",
        "relative_l2_solution_values": "[1.0, 3.0]",
        "sample_count": 2,
    }
    summary = dict(raw)
    summary.pop("relative_l2_solution_values")
    summary.update({"relative_l2_solution_mean": 100.0, "relative_l2_solution_std": 0.0, "relative_l2_solution_n": 100})
    (tmp_path / "results_raw.jsonl").write_text(json.dumps(raw) + "\n", encoding="utf-8")
    (tmp_path / "results_summary.jsonl").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    paths = _expand_inputs([str(tmp_path)])
    assert paths == [tmp_path / "results_raw.jsonl"]
    rows = []
    for path in paths:
        rows.extend(_read_jsonl(path))
    aggregated = aggregate_rows(rows)[0]
    assert aggregated["aggregation_mode"] == "raw"
    assert aggregated["relative_l2_solution_mean"] == pytest.approx(2.0)
    assert aggregated["relative_l2_solution_n"] == 2
