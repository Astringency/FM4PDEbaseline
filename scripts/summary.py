#!/usr/bin/env python
"""Collect smooth, ID, and rough results into one compact XLSX workbook.

Every current run in ``matrices/main_results.jsonl`` contributes Smooth, ID,
and Rough rows.  The Smooth values come from ``runs/main_results``.  Matching
summaries under ``runs/evaluations/id`` and ``runs/evaluations/rough`` fill the
optional evaluation rows; missing evaluations remain blank.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo


SUMMARY_COLUMNS = [
    "PDE",
    "Method",
    "TASK",
    "DIST",
    "SENSOR",
    "rel L2(a)",
    "rel L2(u)",
    "pde L",
    "Remark",
]

_DISTRIBUTIONS = ("smooth", "id", "rough")
_METRIC_FIELDS = {
    "a": "relative_l2_input_or_coeff_mean",
    "u": "relative_l2_solution_mean",
}
_SUMMARY_IDENTITY_FIELDS = ("task_group", "pde", "baseline", "seed")
_TASK_LABELS = {
    "forward": "forward",
    "sparse_forward": "forward",
    "inverse": "inverse",
    "sparse_inverse": "inverse",
    "sparse_solution": "both",
    "sparse_solution_multicondition": "both",
}
_PDE_LABELS = {
    "poisson": "Poisson",
    "helmholtz": "Helmholtz",
    "darcy": "Darcy",
    "burger": "Burgers",
    "nsnonbounded": "NS",
}
_METHOD_LABELS = {
    "fno": "FNO",
    "deeponet": "DeepONet",
    "ifno": "IFNO",
    "recfno": "RecFNO",
    "senseiver": "Senseiver",
    "voronoicnn": "VoronoiCNN",
    "pinn_sparse": "PINN-Sparse",
    "pde_opt": "PDE-Opt",
    "pc_bnn": "PC-BNN",
    "var4d": "Var4D",
    "vivid": "VIVID",
}


def _resolve_out_root(out_root: str | Path | None) -> Path:
    if out_root is None:
        configured = os.environ.get("OUT_ROOT", "").strip()
        if not configured:
            configured = os.environ.get("FM_OUTPUT_ROOT", "").strip()
        out_root = configured or "outputs"
    return Path(out_root).expanduser().resolve()


def _read_summary(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid summary JSON: {path}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"cannot read summary: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"summary is not a JSON object: {path}")
    return value


def _load_matrix(root: Path) -> list[dict[str, Any]]:
    path = root / "matrices" / "main_results.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"cannot read main-results matrix: {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid matrix JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"matrix row is not a JSON object at {path}:{line_number}")
        if not row.get("skip_reason"):
            rows.append(row)
    if not rows:
        raise RuntimeError(f"main-results matrix has no runnable rows: {path}")
    return rows


def _main_summary_path(root: Path, row: Mapping[str, Any]) -> Path:
    configured = str(row.get("output_dir", ""))
    marker = "/runs/main_results/"
    if marker in configured:
        relative = configured.split(marker, 1)[1]
        return root / "runs" / "main_results" / relative / "summary.json"
    if configured:
        return Path(configured).expanduser() / "summary.json"
    return (
        root
        / "runs"
        / "main_results"
        / f"task_group={row['task_group']}"
        / f"pde={row['pde']}"
        / f"baseline={row['baseline']}"
        / f"seed={row['seed']}"
        / f"run={row['run_id']}"
        / "summary.json"
    )


def _evaluation_summary_path(
    root: Path,
    row: Mapping[str, Any],
    distribution: str,
) -> Path:
    return (
        root
        / "runs"
        / "evaluations"
        / distribution
        / f"task_group={row['task_group']}"
        / f"pde={row['pde']}"
        / f"baseline={row['baseline']}"
        / f"seed={row['seed']}"
        / f"run=eval_{distribution}_{row['run_id']}"
        / "summary.json"
    )


def _load_result(
    path: Path,
    row: Mapping[str, Any],
    *,
    expected_run_id: str,
    required: bool,
) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise RuntimeError(f"main-results summary is missing: {path}")
        return None
    item = _read_summary(path)
    if item.get("status") != "success":
        if required:
            raise RuntimeError(f"main-results summary is not successful: {path}")
        return None
    expected = {field: row.get(field) for field in _SUMMARY_IDENTITY_FIELDS}
    expected["run_id"] = expected_run_id
    for field, value in expected.items():
        if item.get(field) != value:
            raise RuntimeError(
                f"summary identity mismatch at {path}: "
                f"{field}={item.get(field)!r}, expected {value!r}"
            )
    return item


def _relative_l2(item: Mapping[str, Any] | None, metric: str) -> float | str:
    if item is None:
        return ""
    value = item.get(_METRIC_FIELDS[metric])
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if math.isfinite(number) else ""


def _finite_value(item: Mapping[str, Any] | None, field: str) -> float | str:
    if item is None:
        return ""
    try:
        value = float(item.get(field))
    except (TypeError, ValueError):
        return ""
    return value if math.isfinite(value) else ""


def _task_label(row: Mapping[str, Any]) -> str:
    task = str(row.get("task", ""))
    try:
        return _TASK_LABELS[task]
    except KeyError as exc:
        raise RuntimeError(f"unsupported task in main-results matrix: {task!r}") from exc


def _sensor_label(item: Mapping[str, Any]) -> str:
    mode = str(item.get("sensor_mode", item.get("requested_sensor_mode", ""))).lower()
    if mode in {"", "none"}:
        return ""
    if "time_slice" in mode or "sensor_col" in mode or "sersor_col" in mode:
        return "sersor_col"
    return "random"


def _display_label(value: Any, labels: Mapping[str, str]) -> str:
    text = str(value)
    return labels.get(text.lower(), text)


def _relative_folder(root: Path, summary_path: Path) -> str:
    folder = summary_path.parent
    try:
        return str(folder.relative_to(root))
    except ValueError:
        return str(folder)


def collect_summary_rows(out_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Return current matrix rows with smooth and optional ID/rough metrics."""

    root = _resolve_out_root(out_root)
    rows: list[dict[str, Any]] = []
    for row in _load_matrix(root):
        run_id = str(row["run_id"])
        main_path = _main_summary_path(root, row)
        smooth = _load_result(
            main_path,
            row,
            expected_run_id=run_id,
            required=True,
        )
        assert smooth is not None
        result_paths = {
            "smooth": main_path,
            **{
                distribution: _evaluation_summary_path(root, row, distribution)
                for distribution in _DISTRIBUTIONS
                if distribution != "smooth"
            },
        }
        results: dict[str, dict[str, Any] | None] = {
            "smooth": smooth,
            **{
                distribution: _load_result(
                    result_paths[distribution],
                    row,
                    expected_run_id=f"eval_{distribution}_{run_id}",
                    required=False,
                )
                for distribution in _DISTRIBUTIONS
                if distribution != "smooth"
            },
        }
        task = _task_label(row)
        for distribution in _DISTRIBUTIONS:
            result = results[distribution]
            source = _relative_folder(root, result_paths[distribution])
            rows.append(
                {
                    "PDE": _display_label(row.get("pde", ""), _PDE_LABELS),
                    "Method": _display_label(row.get("baseline", ""), _METHOD_LABELS),
                    "TASK": task,
                    "DIST": distribution.title() if distribution != "id" else "ID",
                    "SENSOR": _sensor_label(result or smooth),
                    "rel L2(a)": _relative_l2(result, "a"),
                    "rel L2(u)": _relative_l2(result, "u"),
                    "pde L": (
                        _finite_value(result, "pde_residual_mean") if task == "both" else ""
                    ),
                    "Remark": source if result is not None else f"missing: {source}",
                }
            )
    return rows


def summary(out_root: str | Path | None = None) -> Path:
    """Write ``OUT_ROOT/summary/results.xlsx`` and return its path."""

    root = _resolve_out_root(out_root)
    rows = collect_summary_rows(root)
    output_dir = root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "results.xlsx"
    temporary = output_dir / ".results.tmp.xlsx"
    try:
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "Results"
        worksheet.append(SUMMARY_COLUMNS)
        for row in rows:
            worksheet.append([row[column] for column in SUMMARY_COLUMNS])

        header_fill = PatternFill("solid", fgColor="2F75B5")
        for cell in worksheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = f"A1:I{len(rows) + 1}"
        widths = (14, 16, 10, 10, 12, 14, 14, 14, 72)
        for index, width in enumerate(widths, start=1):
            worksheet.column_dimensions[chr(64 + index)].width = width
        for row_number in range(2, len(rows) + 2):
            for column in (6, 7):
                worksheet.cell(row_number, column).number_format = "0.000000"
            worksheet.cell(row_number, 8).number_format = "0.000000E+00"
        table = Table(displayName="ResultsTable", ref=f"A1:I{len(rows) + 1}")
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        worksheet.add_table(table)
        workbook.save(temporary)
        workbook.close()

        validation = load_workbook(temporary, read_only=True, data_only=True)
        try:
            if validation.sheetnames != ["Results"]:
                raise RuntimeError(f"unexpected workbook sheets: {validation.sheetnames}")
            saved = validation["Results"]
            headers = [saved.cell(1, column).value for column in range(1, 10)]
            if headers != SUMMARY_COLUMNS or saved.max_row != len(rows) + 1:
                raise RuntimeError("saved workbook structure does not match the summary rows")
        finally:
            validation.close()
        temporary.replace(output)
        legacy_csv = output_dir / "results.csv"
        if legacy_csv.is_file():
            legacy_csv.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        "--output-root",
        dest="out_root",
        default=None,
        help="experiment output root (default: OUT_ROOT, FM_OUTPUT_ROOT, or outputs)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = summary(args.out_root)
    workbook = load_workbook(output, read_only=True)
    try:
        rows = workbook["Results"].max_row - 1
    finally:
        workbook.close()
    print(json.dumps({"output": str(output), "rows": rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
