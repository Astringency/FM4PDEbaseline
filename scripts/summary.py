#!/usr/bin/env python
"""Collect smooth, ID, and rough results into one compact CSV file.

Every successful run under ``runs/main_results`` contributes one output row.
Those results are the smooth-distribution results.  Matching summaries under
``runs/evaluations/id`` and ``runs/evaluations/rough`` fill the optional ID and
rough columns; missing evaluations remain blank.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_COLUMNS = [
    "task_group",
    "pde",
    "task",
    "baseline",
    "seed",
    "relative_l2_a_smooth_pct",
    "relative_l2_a_id_pct",
    "relative_l2_a_rough_pct",
    "relative_l2_u_smooth_pct",
    "relative_l2_u_id_pct",
    "relative_l2_u_rough_pct",
]

_DISTRIBUTIONS = ("id", "rough")
_METRIC_FIELDS = {
    "a": "relative_l2_input_or_coeff_mean",
    "u": "relative_l2_solution_mean",
}
_SUMMARY_IDENTITY_FIELDS = ("task_group", "pde", "baseline", "seed")


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


def _percent(item: Mapping[str, Any] | None, metric: str) -> float | str:
    if item is None:
        return ""
    value = item.get(_METRIC_FIELDS[metric])
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number * 100.0 if math.isfinite(number) else ""


def collect_summary_rows(out_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Return current matrix rows with smooth and optional ID/rough metrics."""

    root = _resolve_out_root(out_root)
    rows: list[dict[str, Any]] = []
    for row in _load_matrix(root):
        run_id = str(row["run_id"])
        smooth = _load_result(
            _main_summary_path(root, row),
            row,
            expected_run_id=run_id,
            required=True,
        )
        assert smooth is not None
        evaluations = {
            distribution: _load_result(
                _evaluation_summary_path(root, row, distribution),
                row,
                expected_run_id=f"eval_{distribution}_{run_id}",
                required=False,
            )
            for distribution in _DISTRIBUTIONS
        }
        item: dict[str, Any] = {
            "task_group": smooth.get("task_group", ""),
            "pde": smooth.get("pde", ""),
            "task": smooth.get("task", ""),
            "baseline": smooth.get("baseline", ""),
            "seed": smooth.get("seed", ""),
        }
        for metric in _METRIC_FIELDS:
            item[f"relative_l2_{metric}_smooth_pct"] = _percent(smooth, metric)
            for distribution in _DISTRIBUTIONS:
                item[f"relative_l2_{metric}_{distribution}_pct"] = _percent(
                    evaluations[distribution], metric
                )
        rows.append({column: item.get(column, "") for column in SUMMARY_COLUMNS})
    return rows


def summary(out_root: str | Path | None = None) -> Path:
    """Write ``OUT_ROOT/summary/results.csv`` and return its path."""

    root = _resolve_out_root(out_root)
    rows = collect_summary_rows(root)
    output_dir = root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "results.csv"
    temporary = output_dir / ".results.csv.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(
                {
                    key: f"{value:.6g}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
                for row in rows
            )
        temporary.replace(output)
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
    with output.open(encoding="utf-8") as handle:
        rows = sum(1 for _ in handle) - 1
    print(json.dumps({"output": str(output), "rows": rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
