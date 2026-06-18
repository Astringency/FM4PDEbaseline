#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-/home/tat512/C01Python/PDEdata}"
OUT_ROOT="${OUT_ROOT:-outputs/baselines_large}"

if [ ! -d "$DATA_ROOT" ]; then
  echo "DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT/matrices"
rm -f "$OUT_ROOT/skipped_combinations.jsonl"

python scripts/experiments/build_matrix.py --config configs/experiments/sanity.yaml --output-root "$OUT_ROOT" --matrix-name sanity
python scripts/experiments/build_matrix.py --config configs/experiments/core.yaml --output-root "$OUT_ROOT" --matrix-name core
python scripts/experiments/build_matrix.py --config configs/experiments/full_all.yaml --output-root "$OUT_ROOT" --matrix-name full_all
python scripts/experiments/build_matrix.py --config configs/experiments/time_varying.yaml --output-root "$OUT_ROOT" --matrix-name time_varying

python - "$OUT_ROOT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for summary_path in sorted((root / "matrices").glob("*_summary.json")):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"{summary['matrix_name']}: {summary['run_count']} runs, {summary['skipped_combo_count']} skipped combo keys")
    for group, count in summary.get("by_task_group", {}).items():
        print(f"  {group}: {count}")
PY
