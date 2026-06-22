#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${1:-${OUT_ROOT:-outputs/baselines_large}}"
MATRIX="${MATRIX:-$OUT_ROOT/matrices/main_results.jsonl}"

python - "$OUT_ROOT" "$MATRIX" <<'PY'
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

out_root = Path(sys.argv[1])
matrix = Path(sys.argv[2])

def count(pattern: str) -> int:
    return sum(1 for _ in out_root.rglob(pattern)) if out_root.exists() else 0

def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def has_marker(row: dict, marker: str) -> bool:
    output_dir = Path(str(row.get("output_dir", "")))
    return (output_dir / marker).exists()

rows = read_rows(matrix)
done_total = count("run.done")
failed_total = count("run.failed")
summary_total = count("summary.json")
results_summary_total = count("results_summary.jsonl")
running_total = count("run.running")

print(f"OUT_ROOT: {out_root}")
print(f"run.done: {done_total}")
print(f"run.failed: {failed_total}")
print(f"summary.json: {summary_total}")
print(f"results_summary.jsonl: {results_summary_total}")
print(f"run.running: {running_total}")

if matrix.exists():
    matrix_total = len(rows)
    matrix_done = sum(1 for row in rows if has_marker(row, "run.done"))
    pct = (100.0 * matrix_done / matrix_total) if matrix_total else 0.0
    print(f"matrix: {matrix}")
    print(f"matrix rows: {matrix_total}")
    print(f"matrix done: {matrix_done}/{matrix_total} ({pct:.1f}%)")
else:
    print(f"matrix: {matrix} (not found)")

if rows:
    table: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "done": 0, "failed": 0, "summary": 0})
    for row in rows:
        pde = str(row.get("pde", ""))
        output_dir = Path(str(row.get("output_dir", "")))
        table[pde]["total"] += 1
        table[pde]["done"] += int((output_dir / "run.done").exists())
        table[pde]["failed"] += int((output_dir / "run.failed").exists())
        table[pde]["summary"] += int((output_dir / "summary.json").exists())
    print("")
    print("pde,total_rows,done,failed,summary")
    for pde in sorted(table):
        item = table[pde]
        print(f"{pde},{item['total']},{item['done']},{item['failed']},{item['summary']}")

def recent_lines(pattern: str, needle: str, limit: int = 20) -> list[str]:
    matches: list[tuple[float, str]] = []
    if not out_root.exists():
        return []
    for path in out_root.rglob(pattern):
        try:
            text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in text:
            if needle in line:
                matches.append((path.stat().st_mtime, f"{path}: {line}"))
    return [line for _, line in sorted(matches, key=lambda item: item[0])[-limit:]]

print("")
print("recent [fit epoch]")
fit_lines = recent_lines("stderr.log", "[fit epoch]")
print("\n".join(fit_lines) if fit_lines else "<none>")

print("")
print("recent [dataset memory]")
memory_lines = recent_lines("stderr.log", "[dataset memory]")
print("\n".join(memory_lines) if memory_lines else "<none>")

failed_dirs = sorted({path.parent for path in out_root.rglob("run.failed")})[:10] if out_root.exists() else []
if failed_dirs:
    print("")
    print("failed run dirs (first 10)")
    for path in failed_dirs:
        print(path)
    print("Inspect a failure with: tail -200 <failed-run-dir>/stderr.log")
PY

