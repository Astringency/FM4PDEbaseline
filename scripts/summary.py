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


Identity = tuple[str, str, str, str]

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
_IDENTITY_FIELDS = ("task_group", "pde", "baseline", "seed")


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


def _successful_summaries(root: Path, *, required: bool) -> list[dict[str, Any]]:
    if not root.is_dir():
        if required:
            raise RuntimeError(f"results directory is missing: {root}")
        return []

    summaries = [_read_summary(path) for path in sorted(root.glob("**/summary.json"))]
    successful = [item for item in summaries if item.get("status") == "success"]
    if required and not successful:
        raise RuntimeError(f"no successful result summaries found under: {root}")
    return successful


def _identity(item: Mapping[str, Any]) -> Identity:
    values = [str(item.get(field, "")) for field in _IDENTITY_FIELDS]
    return values[0], values[1], values[2], values[3]


def _index_main_results(
    summaries: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[Identity, str]]:
    by_run_id: dict[str, dict[str, Any]] = {}
    by_identity: dict[Identity, str] = {}
    for item in summaries:
        run_id = str(item.get("run_id", ""))
        if not run_id:
            raise RuntimeError("a main-results summary has no run_id")
        if run_id in by_run_id:
            raise RuntimeError(f"duplicate main-results run_id: {run_id}")
        identity = _identity(item)
        if identity in by_identity:
            raise RuntimeError(f"duplicate main-results identity: {identity}")
        by_run_id[run_id] = item
        by_identity[identity] = run_id
    return by_run_id, by_identity


def _index_evaluations(
    summaries: Sequence[dict[str, Any]],
    *,
    distribution: str,
    main_by_run_id: Mapping[str, dict[str, Any]],
    main_by_identity: Mapping[Identity, str],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    prefix = f"eval_{distribution}_"
    for item in summaries:
        eval_run_id = str(item.get("run_id", ""))
        source_run_id = eval_run_id[len(prefix) :] if eval_run_id.startswith(prefix) else ""
        if source_run_id not in main_by_run_id:
            source_run_id = main_by_identity.get(_identity(item), "")
        if not source_run_id:
            continue
        if source_run_id in indexed:
            raise RuntimeError(
                f"duplicate {distribution} evaluation for main run: {source_run_id}"
            )
        indexed[source_run_id] = item
    return indexed


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
    """Return one row per successful main run, with optional ID/rough metrics."""

    root = _resolve_out_root(out_root)
    main_summaries = _successful_summaries(root / "runs" / "main_results", required=True)
    main_by_run_id, main_by_identity = _index_main_results(main_summaries)
    evaluations = {
        distribution: _index_evaluations(
            _successful_summaries(
                root / "runs" / "evaluations" / distribution,
                required=False,
            ),
            distribution=distribution,
            main_by_run_id=main_by_run_id,
            main_by_identity=main_by_identity,
        )
        for distribution in _DISTRIBUTIONS
    }

    rows: list[dict[str, Any]] = []
    for run_id, smooth in main_by_run_id.items():
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
                    evaluations[distribution].get(run_id), metric
                )
        rows.append({column: item.get(column, "") for column in SUMMARY_COLUMNS})

    return sorted(
        rows,
        key=lambda item: (
            str(item["task_group"]),
            str(item["pde"]),
            str(item["baseline"]),
            str(item["seed"]),
        ),
    )


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
