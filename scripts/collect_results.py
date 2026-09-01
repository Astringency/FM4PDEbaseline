#!/usr/bin/env python
"""Collect a completed experiment matrix into compact main-result tables.

The main-results matrix intentionally contains both training rows and a small
number of checkpoint-backed ``eval_only`` rows. Those execution modes remain
separate provenance cohorts, but they belong in the same main-results table.
This collector validates every row independently and records cohort ids without
requiring the whole matrix to have one cohort id.

The compact tables are built directly from each run's ``summary.json``. Raw
batch JSONL files are deliberately not re-aggregated here: the run summary is
the authoritative, per-sample-pooled result and keeps this publication output
focused on experiment settings plus relative errors for ``a`` and ``u``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_one import load_matrix_rows
from scripts.export_results_xlsx import collect_results as collect_validated_results


COMPACT_COLUMNS = [
    "experiment_type",
    "pde",
    "task",
    "baseline",
    "seed",
    "execution_mode",
    "train_size",
    "train_size_requested",
    "test_size",
    "num_sensors",
    "sensor_mode",
    "sensor_budget_mode",
    "noise_level",
    "epochs",
    "method_budget_label",
    "relative_l2_a_mean",
    "relative_l2_a_std",
    "relative_l2_a_ci95",
    "relative_l2_a_n",
    "relative_l2_u_mean",
    "relative_l2_u_std",
    "relative_l2_u_ci95",
    "relative_l2_u_n",
    "pde_residual_mean",
    "obs_mse_clean_mean",
    "train_time",
    "inference_time_per_sample",
    "num_params",
    "run_id",
    "cohort_id",
]

LATEX_COLUMNS = [
    "experiment_type",
    "pde",
    "baseline",
    "relative_l2_a_mean",
    "relative_l2_u_mean",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-xlsx", action="store_true")
    parser.add_argument("--latex", action="store_true")
    return parser.parse_args(argv)


def load_rows(matrix: Path) -> list[dict[str, Any]]:
    return [row for row in load_matrix_rows(matrix) if not row.get("skip_reason")]


def _finite_or_blank(value: Any) -> Any:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if math.isfinite(number) else ""


def _metric_count(summary: dict[str, Any], metric: str) -> int | str:
    mean = _finite_or_blank(summary.get(f"{metric}_mean"))
    if mean == "":
        return ""
    try:
        return int(summary.get(f"{metric}_n", 0) or 0)
    except (TypeError, ValueError):
        return ""


def compact_result_rows(
    rows: list[dict[str, Any]],
    cohort_by_run_id: dict[str, str],
) -> list[dict[str, Any]]:
    """Read validated run summaries and retain only decision-useful columns."""
    compact: list[dict[str, Any]] = []
    for row in rows:
        summary_path = Path(str(row["output_dir"])) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_id = str(row.get("run_id", ""))
        item = {
            "experiment_type": summary.get("task_group", row.get("task_group", "")),
            "pde": summary.get("pde", row.get("pde", "")),
            "task": summary.get("task", row.get("task", "")),
            "baseline": summary.get("baseline", row.get("baseline", "")),
            "seed": summary.get("seed", row.get("seed", "")),
            "execution_mode": summary.get("execution_mode", row.get("execution_mode", "")),
            "train_size": summary.get("train_size", ""),
            "train_size_requested": summary.get("train_size_requested", row.get("train_size", "")),
            "test_size": summary.get("test_size", row.get("test_size", "")),
            "num_sensors": summary.get("num_sensors", row.get("num_sensors", "")),
            "sensor_mode": summary.get("sensor_mode", row.get("sensor_mode", "")),
            "sensor_budget_mode": summary.get(
                "sensor_budget_mode", row.get("sensor_budget_mode", "")
            ),
            "noise_level": summary.get("noise_level", row.get("noise_level", "")),
            "epochs": summary.get("epochs", row.get("epochs", "")),
            "method_budget_label": summary.get("method_budget_label", ""),
            "relative_l2_a_mean": _finite_or_blank(
                summary.get("relative_l2_input_or_coeff_mean")
            ),
            "relative_l2_a_std": _finite_or_blank(
                summary.get("relative_l2_input_or_coeff_std")
            ),
            "relative_l2_a_ci95": _finite_or_blank(
                summary.get("relative_l2_input_or_coeff_ci95")
            ),
            "relative_l2_a_n": _metric_count(summary, "relative_l2_input_or_coeff"),
            "relative_l2_u_mean": _finite_or_blank(summary.get("relative_l2_solution_mean")),
            "relative_l2_u_std": _finite_or_blank(summary.get("relative_l2_solution_std")),
            "relative_l2_u_ci95": _finite_or_blank(summary.get("relative_l2_solution_ci95")),
            "relative_l2_u_n": _metric_count(summary, "relative_l2_solution"),
            "pde_residual_mean": _finite_or_blank(summary.get("pde_residual_mean")),
            "obs_mse_clean_mean": _finite_or_blank(summary.get("obs_mse_clean_mean")),
            "train_time": _finite_or_blank(summary.get("train_time")),
            "inference_time_per_sample": _finite_or_blank(
                summary.get("inference_time_per_sample")
            ),
            "num_params": summary.get("num_params", ""),
            "run_id": run_id,
            "cohort_id": cohort_by_run_id.get(run_id, ""),
        }
        compact.append({column: item.get(column, "") for column in COMPACT_COLUMNS})
    return sorted(
        compact,
        key=lambda item: (
            str(item["experiment_type"]),
            str(item["pde"]),
            str(item["baseline"]),
            str(item["seed"]),
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPACT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(value: Any) -> str:
    text = str(value)
    for source, replacement in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(source, replacement)
    return text


def _latex_value(value: Any) -> str:
    if value == "":
        return "--"
    if isinstance(value, float):
        return f"{value:.6g}"
    return _latex_escape(value)


def _write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = {
        "experiment_type": "Experiment",
        "pde": "PDE",
        "baseline": "Baseline",
        "relative_l2_a_mean": r"Rel. $L_2(a)$",
        "relative_l2_u_mean": r"Rel. $L_2(u)$",
    }
    lines = [
        r"\begin{tabular}{lllrr}",
        r"\toprule",
        " & ".join(labels[column] for column in LATEX_COLUMNS) + r" \\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(_latex_value(row[column]) for column in LATEX_COLUMNS) + r" \\" for row in rows
    )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_compact_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    matrix: Path,
    cohort_counts: dict[str, int],
    write_xlsx: bool,
    write_latex: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "matrix": str(matrix),
        "matrix_rows": len(rows),
        "result_rows": len(rows),
        "column_count": len(COMPACT_COLUMNS),
        "cohort_count": len(cohort_counts),
        "cohort_counts": cohort_counts,
        "cohort_policy": "validated_per_run; mixed execution_mode cohorts allowed",
        "relative_l2_a_source": "relative_l2_input_or_coeff_*",
        "relative_l2_u_source": "relative_l2_solution_*",
        "blank_metric_semantics": "not applicable or non-finite",
    }
    outputs: dict[str, Any] = {
        "summary_csv": str(csv_path),
        "summary_json": str(json_path),
    }
    if write_xlsx:
        workbook = output_dir / "results.xlsx"
        workbook_manifest = {
            **manifest,
            "cohort_counts": json.dumps(cohort_counts, sort_keys=True),
        }
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            pd.DataFrame.from_records(rows, columns=COMPACT_COLUMNS).to_excel(
                writer, sheet_name="results", index=False
            )
            pd.DataFrame.from_records([workbook_manifest]).to_excel(
                writer, sheet_name="manifest", index=False
            )
        outputs["workbook"] = str(workbook)
    if write_latex:
        latex_path = output_dir / "latex_table.tex"
        _write_latex(latex_path, rows)
        outputs["latex"] = str(latex_path)
    return {**manifest, **outputs}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.matrix)
    output_dir = args.output_dir or args.matrix.parent.parent / "aggregate" / args.matrix.stem
    collection = collect_validated_results(rows)
    validation_errors = []
    if collection.missing_run_ids:
        validation_errors.append(f"missing summaries: {len(collection.missing_run_ids)}")
    if collection.quarantine:
        validation_errors.append(f"invalid provenance: {len(collection.quarantine)}")
    if not rows:
        validation_errors.append("matrix has no runnable rows")
    if len(collection.records) != len(rows):
        validation_errors.append(f"validated summaries: {len(collection.records)}/{len(rows)}")
    if validation_errors:
        print(
            json.dumps(
                {
                    "matrix": str(args.matrix),
                    "publication_blocked": True,
                    "errors": validation_errors,
                    "missing_run_ids": collection.missing_run_ids,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    cohort_by_run_id = {
        str(record.get("run_id", "")): str(record.get("cohort_id", ""))
        for record in collection.records
    }
    cohort_counts = dict(Counter(cohort_by_run_id.values()))
    compact_rows = compact_result_rows(rows, cohort_by_run_id)
    report: dict[str, Any] = {
        "matrix": str(args.matrix),
        "matrix_rows": len(rows),
        "result_rows": len(compact_rows),
        "output_dir": str(output_dir),
        "publication_blocked": False,
        **write_compact_outputs(
            compact_rows,
            output_dir,
            matrix=args.matrix,
            cohort_counts=cohort_counts,
            write_xlsx=not args.no_xlsx,
            write_latex=args.latex,
        ),
    }
    (output_dir / "collection_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
