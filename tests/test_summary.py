from __future__ import annotations

import json
from pathlib import Path

import pytest
from openpyxl import load_workbook

from scripts import summary as result_summary


def _write_result(
    root: Path,
    *,
    run_id: str,
    task_group: str,
    baseline: str,
    rel_a: float | None,
    rel_u: float | None,
    task: str = "sparse_inverse",
    sensor_mode: str = "random_per_sample",
    pde_loss: float | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "status": "success",
                "run_id": run_id,
                "task_group": task_group,
                "pde": "poisson",
                "task": task,
                "baseline": baseline,
                "seed": 1,
                "sensor_mode": sensor_mode,
                "relative_l2_input_or_coeff_mean": rel_a,
                "relative_l2_solution_mean": rel_u,
                "pde_residual_mean": pde_loss,
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
                "task_group": "sparse_solution_main_amortized",
                "pde": "poisson",
                "task": "sparse_solution",
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
        task_group="sparse_solution_main_amortized",
        baseline="recfno",
        rel_a=0.1,
        rel_u=0.2,
        task="sparse_solution",
        pde_loss=0.01,
    )
    _write_result(
        physics_dir,
        run_id="physics",
        task_group="sparse_inverse_main",
        baseline="pde_opt",
        rel_a=0.3,
        rel_u=None,
        pde_loss=0.9,
    )
    _write_result(
        eval_root
        / "id/task_group=sparse_solution_main_amortized/pde=poisson"
        / "baseline=recfno/seed=1/run=eval_id_trained",
        run_id="eval_id_trained",
        task_group="sparse_solution_main_amortized",
        baseline="recfno",
        rel_a=0.4,
        rel_u=0.5,
        task="sparse_solution",
        pde_loss=0.02,
    )
    _write_result(
        eval_root
        / "rough/task_group=sparse_solution_main_amortized/pde=poisson"
        / "baseline=recfno/seed=1/run=eval_rough_trained",
        run_id="eval_rough_trained",
        task_group="sparse_solution_main_amortized",
        baseline="recfno",
        rel_a=0.6,
        rel_u=0.7,
        task="sparse_solution",
        sensor_mode="time_slices_per_sample",
        pde_loss=0.03,
    )

    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    (summary_dir / "results.csv").write_text("legacy", encoding="utf-8")

    output = result_summary.summary(tmp_path)

    assert output == tmp_path / "summary" / "results.xlsx"
    assert {path.name for path in output.parent.iterdir()} == {"results.xlsx"}
    workbook = load_workbook(output, read_only=True, data_only=True)
    assert workbook.sheetnames == ["Results"]
    worksheet = workbook["Results"]
    assert [cell.value for cell in worksheet[1]] == result_summary.SUMMARY_COLUMNS
    rows = [
        dict(zip(result_summary.SUMMARY_COLUMNS, values))
        for values in worksheet.iter_rows(min_row=2, values_only=True)
    ]
    workbook.close()
    assert len(rows) == 6
    trained_smooth = next(
        row for row in rows if row["Method"] == "RecFNO" and row["DIST"] == "Smooth"
    )
    trained_rough = next(
        row for row in rows if row["Method"] == "RecFNO" and row["DIST"] == "Rough"
    )
    physics_smooth = next(
        row for row in rows if row["Method"] == "PDE-Opt" and row["DIST"] == "Smooth"
    )
    physics_id = next(
        row for row in rows if row["Method"] == "PDE-Opt" and row["DIST"] == "ID"
    )
    assert trained_smooth["TASK"] == "both"
    assert trained_smooth["SENSOR"] == "random"
    assert trained_smooth["rel L2(a)"] == pytest.approx(0.1)
    assert trained_smooth["rel L2(u)"] == pytest.approx(0.2)
    assert trained_smooth["pde L"] == pytest.approx(0.01)
    assert trained_rough["SENSOR"] == "sersor_col"
    assert trained_rough["rel L2(a)"] == pytest.approx(0.6)
    assert physics_smooth["TASK"] == "inverse"
    assert physics_smooth["rel L2(a)"] == pytest.approx(0.3)
    assert physics_smooth["rel L2(u)"] is None
    assert physics_smooth["pde L"] is None
    assert physics_id["rel L2(a)"] is None
    assert str(physics_id["Remark"]).startswith("missing: runs/evaluations/id/")


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
                "task": "forward",
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
        task="forward",
        sensor_mode="none",
    )
    monkeypatch.setenv("OUT_ROOT", str(tmp_path))

    assert result_summary.summary() == tmp_path / "summary" / "results.xlsx"


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
                "task": "forward",
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
        task="forward",
        sensor_mode="none",
    )
    _write_result(
        current / "quarantine" / "invalid_old",
        run_id="old",
        task_group="full_forward_main",
        baseline="fno",
        rel_a=None,
        rel_u=9.9,
        task="forward",
        sensor_mode="none",
    )
    _write_result(
        current.parent / "run=historical",
        run_id="historical",
        task_group="full_forward_main",
        baseline="fno",
        rel_a=None,
        rel_u=8.8,
        task="forward",
        sensor_mode="none",
    )

    rows = result_summary.collect_summary_rows(tmp_path)

    assert len(rows) == 3
    smooth = next(row for row in rows if row["DIST"] == "Smooth")
    assert smooth["rel L2(u)"] == pytest.approx(0.1)
