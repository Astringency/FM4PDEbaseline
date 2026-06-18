#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Create a retry matrix from failed FM4PDE baseline runs.")
    parser.add_argument("--matrix", default=os.environ.get("MATRIX", ""))
    parser.add_argument("--output-root", default=os.environ.get("OUT_ROOT", "outputs/baselines_large"))
    parser.add_argument("--retry-limit", type=int, default=int(os.environ.get("RETRY_LIMIT", "3")))
    parser.add_argument("--output", default=os.environ.get("RETRY_MATRIX", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    matrix = Path(args.matrix) if args.matrix else out_root / "matrices" / "core.jsonl"
    rows = _read_jsonl(matrix)
    retry_rows = []
    for row in rows:
        failed = Path(row["output_dir"]) / "run.failed"
        if not failed.exists():
            continue
        attempt = _attempt(failed)
        if attempt < args.retry_limit:
            retry_rows.append(row)
    output = Path(args.output) if args.output else out_root / "matrices" / f"{matrix.stem}_retry.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in retry_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"matrix": str(matrix), "retry_matrix": str(output), "retry_rows": len(retry_rows)}, indent=2, sort_keys=True))


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


def _attempt(path: Path) -> int:
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("attempt", 1))
    except Exception:
        return 1


if __name__ == "__main__":
    main()
