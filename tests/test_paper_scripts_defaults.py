from __future__ import annotations

from pathlib import Path

from baselines.run import parse_args as parse_baseline_args
from scripts.verify_data_protocol import parse_args as parse_verifier_args


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
    assert "val_size: 5000" in text
    assert "sensor_budget_mode: per_time" in text
    assert "normalize: true" in text
    assert "implementation_mode: official_or_skip" in text


def test_direct_entrypoint_defaults_follow_the_formal_experiment():
    baseline = parse_baseline_args(["--baseline", "fno", "--pde", "poisson"])
    verifier = parse_verifier_args([])

    assert baseline.train_size == 50000
    assert baseline.val_size == 5000
    assert baseline.test_size == 1000
    assert baseline.seed == 1
    assert baseline.task_protocol_version == "fm4pde-task-contract-v3"
    assert baseline.sensor_protocol_version == "fm4pde-sensor-contract-v3"
    assert verifier.config == "configs/experiments/main_results.yaml"
