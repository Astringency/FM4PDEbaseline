from __future__ import annotations

from pathlib import Path


def test_matrix_runner_rejects_debug_paper_flags():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/experiments/run_one.py").read_text(encoding="utf-8")
    forbidden = ["--dry-run", "--synthetic-data", "--allow-synthetic-fallback", "--prefer-test"]
    for token in forbidden:
        assert token in text
    assert "FORBIDDEN_PAPER_FLAGS.intersection" in text


def test_paper_config_defaults_validation_and_sensor_budget():
    root = Path(__file__).resolve().parents[1]
    text = (root / "baselines/configs/paper.yaml").read_text(encoding="utf-8")
    assert "val_size: 1000" in text
    assert "sensor_budget_mode: per_time" in text
    assert "normalize: true" in text
    assert "implementation_mode: official_or_skip" in text
