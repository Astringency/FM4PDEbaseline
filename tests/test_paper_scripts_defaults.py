from __future__ import annotations

from pathlib import Path


def test_paper_scripts_defaults_are_formal_and_validation_enabled():
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "scripts/baselines/run_paper_native_full_operator.sh",
        root / "scripts/baselines/run_paper_native_sparse_reconstruction.sh",
        root / "scripts/baselines/run_paper_static_sparse_inverse.sh",
        root / "scripts/baselines/run_paper_time_varying_da.sh",
        root / "scripts/baselines/run_paper_all_native.sh",
        root / "scripts/baselines/run_paper_full_operator.sh",
        root / "scripts/baselines/run_paper_sparse_reconstruction.sh",
        root / "scripts/baselines/run_paper_physics_da.sh",
        root / "scripts/baselines/run_paper_plan_lightweight.sh",
    ]
    forbidden = ["--dry-run", "--synthetic-data", "--allow-synthetic-fallback", "--prefer-test"]
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        if script.name != "run_paper_all_native.sh":
            assert 'VAL_SIZE="${VAL_SIZE:-1000}"' in text
        if script.name != "run_paper_all_native.sh":
            assert "SENSOR_BUDGET_MODE" in text
        for token in forbidden:
            assert token not in text


def test_paper_config_defaults_validation_and_sensor_budget():
    root = Path(__file__).resolve().parents[1]
    text = (root / "baselines/configs/paper.yaml").read_text(encoding="utf-8")
    assert "val_size: 1000" in text
    assert "sensor_budget_mode: per_time" in text
    assert "normalize: true" in text
    assert "implementation_mode: official_or_skip" in text
