#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Summarize FM4PDE baseline matrix run status.")
    parser.add_argument("--output-root", default=os.environ.get("OUT_ROOT", "outputs/baselines_large"))
    parser.add_argument("--matrix", default=os.environ.get("MATRIX", ""))
    parser.add_argument("--scope", choices=["main", "all", "matrix"], default="main")
    parser.add_argument("--all-matrices", action="store_true", help="Scan every matrix JSONL under matrices/, excluding *_skipped.jsonl.")
    raw_argv = sys.argv[1:] if argv is None else argv
    scope_explicit = any(arg == "--scope" or arg.startswith("--scope=") for arg in raw_argv)
    args = parser.parse_args(argv)
    if args.all_matrices:
        args.scope = "all"
    elif args.matrix and not scope_explicit:
        args.scope = "matrix"
    if args.matrix and scope_explicit and args.scope != "matrix":
        parser.error("--matrix requires --scope matrix unless --scope is omitted")
    if args.scope == "matrix" and not args.matrix:
        parser.error("--scope matrix requires --matrix")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_root = Path(args.output_root)
    matrix_paths = _matrix_paths(out_root, args.scope, args.matrix)
    rows = []
    for path in matrix_paths:
        rows.extend(_read_jsonl(path))
    matrix_names = _selected_matrix_names(args.scope, matrix_paths, args.matrix)
    skipped_rows = [
        row
        for row in _read_jsonl(out_root / "skipped_combinations.jsonl")
        if not matrix_names or str(row.get("matrix_name", "")) in matrix_names
    ]

    details = []
    counts = Counter()
    grouped = Counter()
    for row in rows:
        status = _row_status(row)
        counts[status] += 1
        key = (row.get("task_group", ""), row.get("pde", ""), row.get("baseline", ""), status)
        grouped[key] += 1
        details.append(
            {
                "task_group": row.get("task_group", ""),
                "pde": row.get("pde", ""),
                "baseline": row.get("baseline", ""),
                "run_id": row.get("run_id", ""),
                "status": status,
                "output_dir": row.get("output_dir", ""),
            }
        )

    skipped_expanded = sum(int(row.get("would_have_expanded", 1) or 1) for row in skipped_rows)
    counts["skipped"] += skipped_expanded
    total = len(rows) + skipped_expanded
    counts["total"] = total
    counts["pending"] = len(rows) - counts["done"] - counts["running"] - counts["failed"]

    status_csv, status_md = _status_paths(out_root, args.scope, matrix_paths)
    latest_status_csv = out_root / "status.csv"
    latest_status_md = out_root / "status.md"
    _write_grouped_csv(status_csv, grouped, skipped_rows)
    if latest_status_csv != status_csv:
        _write_grouped_csv(latest_status_csv, grouped, skipped_rows)
    markdown = _status_markdown(args.scope, matrix_paths, counts, grouped, skipped_rows)
    status_md.write_text(markdown, encoding="utf-8")
    if latest_status_md != status_md:
        latest_status_md.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "scope": args.scope,
                "matrices": [str(p) for p in matrix_paths],
                "status_csv": str(status_csv),
                "status_md": str(status_md),
                "latest_status_csv": str(latest_status_csv),
                "latest_status_md": str(latest_status_md),
                **dict(counts),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _matrix_paths(out_root: Path, scope: str, explicit: str = "") -> list[Path]:
    matrix_dir = out_root / "matrices"
    if scope == "matrix":
        return [Path(explicit)]
    if scope == "main":
        return [matrix_dir / "main_results.jsonl"]
    if scope == "all":
        if not matrix_dir.exists():
            return []
        return [p for p in sorted(matrix_dir.glob("*.jsonl")) if not p.name.endswith("_skipped.jsonl")]
    raise ValueError(f"unknown status scope: {scope}")


def _selected_matrix_names(scope: str, matrix_paths: list[Path], explicit: str = "") -> set[str]:
    if scope == "main":
        return {"main_results"}
    if scope == "matrix":
        return {Path(explicit).stem}
    return {path.stem for path in matrix_paths}


def _status_paths(out_root: Path, scope: str, matrix_paths: list[Path]) -> tuple[Path, Path]:
    if scope == "matrix":
        stem = matrix_paths[0].stem if matrix_paths else "matrix"
        suffix = _safe_status_suffix(stem)
    else:
        suffix = scope
    return out_root / f"status_{suffix}.csv", out_root / f"status_{suffix}.md"


def _safe_status_suffix(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return safe or "matrix"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_status(row: dict[str, Any]) -> str:
    output_dir = Path(str(row.get("output_dir", "")))
    if (output_dir / "run.done").exists():
        return "done"
    if (output_dir / "run.running").exists():
        return "running"
    if (output_dir / "run.failed").exists():
        return "failed"
    return "pending"


def _write_grouped_csv(path: Path, grouped: Counter, skipped_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"task_group": group, "pde": pde, "baseline": baseline, "status": status, "count": count}
        for (group, pde, baseline, status), count in sorted(grouped.items())
    ]
    skipped_grouped = Counter()
    for row in skipped_rows:
        skipped_grouped[(row.get("task_group", ""), row.get("pde", ""), row.get("baseline", ""), "skipped")] += int(
            row.get("would_have_expanded", 1) or 1
        )
    rows.extend(
        {"task_group": group, "pde": pde, "baseline": baseline, "status": status, "count": count}
        for (group, pde, baseline, status), count in sorted(skipped_grouped.items())
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_group", "pde", "baseline", "status", "count"])
        writer.writeheader()
        writer.writerows(rows)


def _status_markdown(scope: str, matrix_paths: list[Path], counts: Counter, grouped: Counter, skipped_rows: list[dict[str, Any]]) -> str:
    lines = ["# Large Baseline Status", f"Scope: {scope}", ""]
    lines.append("Matrices:")
    for path in matrix_paths:
        lines.append(f"- {path}")
    lines.extend(["", "## Totals", ""])
    for key in ["total", "done", "running", "failed", "skipped", "pending"]:
        lines.append(f"- {key}: {int(counts.get(key, 0))}")
    lines.extend(["", "## By Task Group", "", "| task_group | status | count |", "|---|---:|---:|"])
    by_group = Counter()
    for (group, _pde, _baseline, status), count in grouped.items():
        by_group[(group, status)] += count
    for row in skipped_rows:
        by_group[(row.get("task_group", ""), "skipped")] += int(row.get("would_have_expanded", 1) or 1)
    for (group, status), count in sorted(by_group.items()):
        lines.append(f"| {group} | {status} | {count} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
