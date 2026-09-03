from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import summary as result_summary


def _write_result(
    root: Path,
    *,
    run_id: str,
    task_group: str,
    baseline: str,
    rel_a: float | None,
    rel_u: float | None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "status": "success",
                "run_id": run_id,
                "task_group": task_group,
                "pde": "poisson",
                "task": "sparse_inverse",
                "baseline": baseline,
                "seed": 1,
                "relative_l2_input_or_coeff_mean": rel_a,
                "relative_l2_solution_mean": rel_u,
            }
        ),
        encoding="utf-8",
    )


def _write_matrix(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_summary_keeps_all_main_results_and_leaves_missing_evaluations_blank(
    tmp_path: Path,
) -> None:
    main_root = tmp_path / "runs" / "main_results"
    eval_root = tmp_path / "runs" / "evaluations"
    trained_dir = main_root / "trained"
    physics_dir = main_root / "physics"
    _write_matrix(
        tmp_path / "matrices" / "main_results.jsonl",
        [
            {
                "run_id": "trained",
                "task_group": "sparse_inverse_main_amortized",
                "pde": "poisson",
                "task": "sparse_inverse",
                "baseline": "recfno",
                "seed": 1,
                "output_dir": str(trained_dir),
            },
            {
                "run_id": "physics",
                "task_group": "sparse_inverse_main",
                "pde": "poisson",
                "task": "sparse_inverse",
                "baseline": "pde_opt",
                "seed": 1,
                "output_dir": str(physics_dir),
            },
        ],
    )
    _write_result(
        trained_dir,
        run_id="trained",
        task_group="sparse_inverse_main_amortized",
        baseline="recfno",
        rel_a=0.1,
        rel_u=0.2,
    )
    _write_result(
        physics_dir,
        run_id="physics",
        task_group="sparse_inverse_main",
        baseline="pde_opt",
        rel_a=0.3,
        rel_u=None,
    )
    _write_result(
        eval_root
        / "id/task_group=sparse_inverse_main_amortized/pde=poisson"
        / "baseline=recfno/seed=1/run=eval_id_trained",
        run_id="eval_id_trained",
        task_group="sparse_inverse_main_amortized",
        baseline="recfno",
        rel_a=0.4,
        rel_u=0.5,
    )
    _write_result(
        eval_root
        / "rough/task_group=sparse_inverse_main_amortized/pde=poisson"
        / "baseline=recfno/seed=1/run=eval_rough_trained",
        run_id="eval_rough_trained",
        task_group="sparse_inverse_main_amortized",
        baseline="recfno",
        rel_a=0.6,
        rel_u=0.7,
    )

    output = result_summary.summary(tmp_path)

    assert output == tmp_path / "summary" / "results.csv"
    assert {path.name for path in output.parent.iterdir()} == {"results.csv"}
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    trained = next(row for row in rows if row["baseline"] == "recfno")
    physics = next(row for row in rows if row["baseline"] == "pde_opt")
    assert float(trained["relative_l2_a_smooth_pct"]) == pytest.approx(10.0)
    assert float(trained["relative_l2_a_id_pct"]) == pytest.approx(40.0)
    assert float(trained["relative_l2_a_rough_pct"]) == pytest.approx(60.0)
    assert float(physics["relative_l2_a_smooth_pct"]) == pytest.approx(30.0)
    assert physics["relative_l2_a_id_pct"] == ""
    assert physics["relative_l2_a_rough_pct"] == ""


def test_summary_uses_out_root_environment_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "runs" / "main_results" / "run"
    _write_matrix(
        tmp_path / "matrices" / "main_results.jsonl",
        [
            {
                "run_id": "run",
                "task_group": "full_forward_main",
                "pde": "poisson",
                "task": "sparse_inverse",
                "baseline": "fno",
                "seed": 1,
                "output_dir": str(run_dir),
            }
        ],
    )
    _write_result(
        run_dir,
        run_id="run",
        task_group="full_forward_main",
        baseline="fno",
        rel_a=None,
        rel_u=0.125,
    )
    monkeypatch.setenv("OUT_ROOT", str(tmp_path))

    assert result_summary.summary() == tmp_path / "summary" / "results.csv"


def test_summary_ignores_quarantined_and_historical_results(tmp_path: Path) -> None:
    current = (
        tmp_path
        / "runs/main_results/task_group=full_forward_main/pde=poisson"
        / "baseline=fno/seed=1/run=current"
    )
    _write_matrix(
        tmp_path / "matrices" / "main_results.jsonl",
        [
            {
                "run_id": "current",
                "task_group": "full_forward_main",
                "pde": "poisson",
                "task": "sparse_inverse",
                "baseline": "fno",
                "seed": 1,
                "output_dir": str(current),
            }
        ],
    )
    _write_result(
        current,
        run_id="current",
        task_group="full_forward_main",
        baseline="fno",
        rel_a=None,
        rel_u=0.1,
    )
    _write_result(
        current / "quarantine" / "invalid_old",
        run_id="old",
        task_group="full_forward_main",
        baseline="fno",
        rel_a=None,
        rel_u=9.9,
    )
    _write_result(
        current.parent / "run=historical",
        run_id="historical",
        task_group="full_forward_main",
        baseline="fno",
        rel_a=None,
        rel_u=8.8,
    )

    rows = result_summary.collect_summary_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["relative_l2_u_smooth_pct"] == pytest.approx(10.0)
