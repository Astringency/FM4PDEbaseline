#!/usr/bin/env python
"""Aggregate a completed matrix into publication tables and a workbook."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.aggregate_results import main as aggregate_main
from scripts.experiments.run_one import load_matrix_rows
from scripts.export_results_xlsx import (
    ExportValidationError,
    collect_results as collect_validated_results,
    export_results,
    require_single_cohort,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-xlsx", action="store_true")
    parser.add_argument("--latex", action="store_true")
    return parser.parse_args(argv)


def load_rows(matrix: Path) -> list[dict[str, Any]]:
    return [row for row in load_matrix_rows(matrix) if not row.get("skip_reason")]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.matrix)
    output_dir = args.output_dir or args.matrix.parent.parent / "aggregate" / args.matrix.stem
    collection = collect_validated_results(rows)
    result_files = [Path(str(row["output_dir"])) / "results_raw.jsonl" for row in rows]
    result_files = [path for path in result_files if path.is_file()]
    validation_errors = []
    if collection.missing_run_ids:
        validation_errors.append(f"missing summaries: {len(collection.missing_run_ids)}")
    if collection.quarantine:
        validation_errors.append(f"invalid provenance: {len(collection.quarantine)}")
    if len(result_files) != len(rows):
        validation_errors.append(f"missing raw results: {len(rows) - len(result_files)}")
    if not rows:
        validation_errors.append("matrix has no runnable rows")
    if not validation_errors:
        try:
            require_single_cohort(collection.records)
        except ExportValidationError as exc:
            validation_errors.append(str(exc))
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
    aggregate_args = [*(str(path) for path in result_files), "--output-dir", str(output_dir)]
    if args.latex:
        aggregate_args.append("--latex-tex")
    aggregate_main(aggregate_args)

    report: dict[str, Any] = {
        "matrix": str(args.matrix),
        "matrix_rows": len(rows),
        "result_files": len(result_files),
        "output_dir": str(output_dir),
    }
    if not args.no_xlsx:
        workbook = output_dir / "results.xlsx"
        try:
            report["workbook"] = export_results(
                rows,
                output=workbook,
                quarantine_output=output_dir / "results_quarantine.xlsx",
                quarantine_jsonl=output_dir / "results_quarantine.jsonl",
                matrix_path=args.matrix,
            )
        except ExportValidationError as exc:
            report["workbook_error"] = str(exc)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "collection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if "workbook_error" not in report else 2


if __name__ == "__main__":
    raise SystemExit(main())
