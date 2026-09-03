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


def test_summary_keeps_all_main_results_and_leaves_missing_evaluations_blank(
    tmp_path: Path,
) -> None:
    main_root = tmp_path / "runs" / "main_results"
    eval_root = tmp_path / "runs" / "evaluations"
    _write_result(
        main_root / "trained",
        run_id="trained",
        task_group="sparse_inverse_main_amortized",
        baseline="recfno",
        rel_a=0.1,
        rel_u=0.2,
    )
    _write_result(
        main_root / "physics",
        run_id="physics",
        task_group="sparse_inverse_main",
        baseline="pde_opt",
        rel_a=0.3,
        rel_u=None,
    )
    _write_result(
        eval_root / "id" / "trained",
        run_id="eval_id_trained",
        task_group="sparse_inverse_main_amortized",
        baseline="recfno",
        rel_a=0.4,
        rel_u=0.5,
    )
    _write_result(
        eval_root / "rough" / "trained",
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
    _write_result(
        tmp_path / "runs" / "main_results" / "run",
        run_id="run",
        task_group="full_forward_main",
        baseline="fno",
        rel_a=None,
        rel_u=0.125,
    )
    monkeypatch.setenv("OUT_ROOT", str(tmp_path))

    assert result_summary.summary() == tmp_path / "summary" / "results.csv"
