#!/usr/bin/env python
"""Run, resume, or inspect an experiment matrix on local CPU/GPU slots."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.provenance import (
    quarantine_output_artifacts,
    reject_historical_experiment_path,
    reject_historical_experiment_row,
    summary_validation_reasons,
)
from scripts.experiments.run_one import load_matrix_rows, process_start_ticks


RUN_ONE = ROOT / "scripts" / "experiments" / "run_one.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("DATA_ROOT", ""))
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "0"), help="Comma-separated GPU ids; use cpu for CPU runs")
    parser.add_argument("--jobs-per-gpu", type=int, default=int(os.environ.get("JOBS_PER_GPU", "1")))
    parser.add_argument("--indices", default="", help="Subset such as 0,2,5-9")
    parser.add_argument("--first", type=int, default=0, help="Run at most the first N selected rows")
    parser.add_argument("--failed-only", action="store_true")
    parser.add_argument(
        "--rerun-running",
        action="store_true",
        help="Treat run.running markers as stale and relaunch them after quarantine",
    )
    parser.add_argument("--status", action="store_true", help="Only print matrix status")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_rows(matrix: Path) -> list[dict[str, Any]]:
    reject_historical_experiment_path(matrix, field="matrix")
    rows = load_matrix_rows(matrix)
    for row in rows:
        reject_historical_experiment_row(row)
    return rows


def completion_reasons(row: dict[str, Any]) -> list[str]:
    summary_path = Path(str(row["output_dir"])) / "summary.json"
    if not summary_path.is_file():
        return ["summary_missing"]
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ["summary_unreadable"]
    if not isinstance(summary, dict):
        return ["summary_not_object"]
    return summary_validation_reasons(row, summary)


def is_complete(row: dict[str, Any]) -> bool:
    return not completion_reasons(row)


def quarantine_invalid_output(row: dict[str, Any]) -> Path | None:
    reject_historical_experiment_row(row)
    output = Path(str(row["output_dir"]))
    if not output.exists() or is_complete(row):
        return None
    return quarantine_output_artifacts(row, completion_reasons(row))


def row_status(row: dict[str, Any]) -> str:
    output = Path(str(row.get("output_dir", "")))
    if row.get("skip_reason"):
        return "skipped"
    for status in ("done", "running", "failed"):
        if (output / f"run.{status}").exists():
            return status
    return "pending"


def running_process_alive(row: dict[str, Any]) -> bool:
    marker = Path(str(row.get("output_dir", ""))) / "run.running"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    for field, command_fragment in (("child_pid", "baselines.run"), ("pid", "run_one.py")):
        try:
            pid = int(payload.get(field, 0) or 0)
        except (TypeError, ValueError):
            continue
        if pid <= 0:
            continue
        actual_start_ticks = process_start_ticks(pid)
        if not actual_start_ticks:
            continue
        try:
            expected_start_ticks = int(payload.get(f"{field}_start_ticks", 0) or 0)
        except (TypeError, ValueError):
            expected_start_ticks = 0
        if expected_start_ticks:
            if actual_start_ticks == expected_start_ticks:
                return True
            continue
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if command_fragment in command:
            return True
    return False


def parse_indices(spec: str, total: int) -> list[int]:
    if not spec:
        return list(range(total))
    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid descending index range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    invalid = sorted(index for index in selected if index < 0 or index >= total)
    if invalid:
        raise IndexError(f"matrix indices out of range 0..{total - 1}: {invalid}")
    return sorted(selected)


def status_report(matrix: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [row_status(row) for row in rows]
    return {"matrix": str(matrix), "total": len(rows), **dict(Counter(statuses))}


def run_index(matrix: Path, index: int, gpu: str, data_root: Path) -> int:
    env = dict(os.environ)
    env["DATA_ROOT"] = str(data_root)
    env["PYTHON"] = sys.executable
    env.pop("ALLOW_ROW_OVERRIDE", None)
    env.pop("DEVICE", None)
    if gpu.lower() == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    completed = subprocess.run(
        [sys.executable, str(RUN_ONE), str(matrix), str(index)],
        cwd=ROOT,
        env=env,
        check=False,
    )
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.jobs_per_gpu < 1:
        raise ValueError("--jobs-per-gpu must be >= 1")
    if args.first < 0:
        raise ValueError("--first must be >= 0")
    rows = load_rows(args.matrix)
    if args.status:
        print(json.dumps(status_report(args.matrix, rows), indent=2, sort_keys=True))
        return 0
    if not args.data_root or not args.data_root.is_dir():
        raise FileNotFoundError(f"data root not found: {args.data_root or '<unset>'}")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id or 'cpu'")
    indices = parse_indices(args.indices, len(rows))
    selected: list[int] = []
    for index in indices:
        status = row_status(rows[index])
        if status == "skipped":
            continue
        if args.failed_only and status != "failed":
            continue
        if status == "running":
            if not args.rerun_running or running_process_alive(rows[index]):
                continue
        if status == "done" and is_complete(rows[index]):
            continue
        selected.append(index)
    if args.first:
        selected = selected[: args.first]
    slots = [gpu for gpu in gpus for _ in range(args.jobs_per_gpu)]
    plan = [
        {"index": index, "run_id": rows[index].get("run_id", ""), "gpu": slots[pos % len(slots)]}
        for pos, index in enumerate(selected)
    ]
    if args.dry_run:
        print(json.dumps({**status_report(args.matrix, rows), "selected": plan}, indent=2, sort_keys=True))
        return 0
    for item in plan:
        quarantine_invalid_output(rows[item["index"]])
    failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = {
            pool.submit(run_index, args.matrix.resolve(), item["index"], item["gpu"], args.data_root.resolve()): item
            for item in plan
        }
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            returncode = future.result()
            if returncode:
                failures.append({**item, "returncode": returncode})
    print(
        json.dumps(
            {**status_report(args.matrix, rows), "selected_count": len(plan), "failed_runs": failures},
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
