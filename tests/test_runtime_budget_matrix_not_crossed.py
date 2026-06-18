from __future__ import annotations

from pathlib import Path

from scripts.experiments.build_matrix import build_matrix, load_config


ROOT = Path(__file__).resolve().parents[1]
MATRIX_ENV = ["SEEDS", "SENSOR_COUNTS", "SENSOR_MODES", "NOISE_LEVELS", "TRAIN_SIZES", "FULL_ABLATION_ALL"]


def _rows(tmp_path: Path, monkeypatch) -> list[dict]:
    for key in MATRIX_ENV:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config(ROOT / "configs" / "experiments" / "runtime_budget_ablation.yaml")
    rows, _skipped, _summary = build_matrix(cfg, tmp_path / "runtime_budget_ablation", "runtime_budget_ablation")
    return rows


def test_runtime_budget_ablation_does_not_cross_steps_refine_steps_particles(tmp_path: Path, monkeypatch):
    rows = _rows(tmp_path, monkeypatch)

    pde_opt_rows = [row for row in rows if row["baseline"] == "pde_opt"]
    assert {row["steps"] for row in pde_opt_rows} == {50, 100, 250, 500, 1000}
    assert {row["refine_steps"] for row in pde_opt_rows} == {0}
    assert {row["particles"] for row in pde_opt_rows} == {0}

    vivid_rows = [row for row in rows if row["baseline"] == "vivid"]
    assert {row["steps"] for row in vivid_rows} == {0}
    assert {row["refine_steps"] for row in vivid_rows} == {50, 100, 250, 500}
    assert {row["particles"] for row in vivid_rows} == {0}

    pc_bnn_rows = [row for row in rows if row["baseline"] == "pc_bnn"]
    assert {row["steps"] for row in pc_bnn_rows} == {50, 100, 250, 500, 1000}
    assert {row["refine_steps"] for row in pc_bnn_rows} == {0}
    assert {row["particles"] for row in pc_bnn_rows} == {8}

    per_problem = {}
    for row in pc_bnn_rows:
        key = (row["pde"], row["seed"])
        per_problem.setdefault(key, set()).add((row["steps"], row["particles"]))
    assert all(len(values) == 5 for values in per_problem.values())
