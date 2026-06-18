#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/baselines_large}"

mkdir -p "$OUT_ROOT/matrices"
rm -f "$OUT_ROOT/skipped_combinations.jsonl"

matrices=(
  sanity_main
  main_results
  sensor_count_ablation
  noise_ablation
  sensor_mode_ablation
  time_varying_sensor_ablation
  runtime_budget_ablation
  train_size_ablation
)

for matrix in "${matrices[@]}"; do
  python scripts/experiments/build_matrix.py \
    --config "configs/experiments/${matrix}.yaml" \
    --output-root "$OUT_ROOT" \
    --matrix-name "$matrix"
done

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
