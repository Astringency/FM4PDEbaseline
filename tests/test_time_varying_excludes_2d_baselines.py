from __future__ import annotations

from pathlib import Path

from scripts.experiments.build_matrix import build_matrix, load_config


ROOT = Path(__file__).resolve().parents[1]
MATRIX_ENV = ["SEEDS", "SENSOR_COUNTS", "SENSOR_MODES", "NOISE_LEVELS", "TRAIN_SIZES", "FULL_ABLATION_ALL"]


def test_time_varying_sensor_ablation_valid_rows_exclude_2d_only_baselines(tmp_path: Path, monkeypatch):
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config(ROOT / "configs" / "experiments" / "time_varying_sensor_ablation.yaml")
    rows, _skipped, _summary = build_matrix(cfg, tmp_path / "time_varying_sensor_ablation", "time_varying_sensor_ablation")

    valid_baselines = {row["baseline"] for row in rows}
    assert valid_baselines == {"var4d", "vivid", "senseiver"}
    assert {"fno", "deeponet", "recfno", "voronoicnn"}.isdisjoint(valid_baselines)
