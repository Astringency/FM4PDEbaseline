#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

OUT_ROOT="${OUT_ROOT:-outputs/baselines_large}"
DATA_ROOT="${DATA_ROOT:-}"
DEVICE="${DEVICE:-}"
MATRIX="${MATRIX:-}"
N_JOBS="${N_JOBS:-}"
TASK_GROUPS="${TASK_GROUPS:-}"
ABLATION="${ABLATION:-}"

log "script=$(basename "$0") start_time=$(date '+%Y-%m-%d %H:%M:%S')"
log "ROOT=$ROOT"
log "DATA_ROOT=${DATA_ROOT:-<unset>}"
log "OUT_ROOT=$OUT_ROOT"
log "DEVICE=${DEVICE:-<unset>}"
log "MATRIX=${MATRIX:-<unset>}"
log "N_JOBS=${N_JOBS:-<unset>} TASK_GROUPS=${TASK_GROUPS:-<unset>} ABLATION=${ABLATION:-<unset>}"

mkdir -p "$OUT_ROOT/matrices"
log "removing stale skipped summary $OUT_ROOT/skipped_combinations.jsonl"
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

start_epoch="$(date +%s)"
for matrix in "${matrices[@]}"; do
  matrix_start="$(date +%s)"
  log "building matrix=$matrix config=configs/experiments/${matrix}.yaml output_root=$OUT_ROOT"
  python scripts/experiments/build_matrix.py \
    --config "configs/experiments/${matrix}.yaml" \
    --output-root "$OUT_ROOT" \
    --matrix-name "$matrix"
  matrix_end="$(date +%s)"
  log "finished matrix=$matrix elapsed_seconds=$((matrix_end - matrix_start))"
done

python - "$OUT_ROOT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for summary_path in sorted((root / "matrices").glob("*_summary.json")):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(
        f"{summary['matrix_name']}: {summary['run_count']} runs, {summary['skipped_combo_count']} skipped combo keys",
        file=sys.stderr,
        flush=True,
    )
    for group, count in summary.get("by_task_group", {}).items():
        print(f"  {group}: {count}", file=sys.stderr, flush=True)
PY
end_epoch="$(date +%s)"
log "finished building matrices elapsed_seconds=$((end_epoch - start_epoch)) output_root=$OUT_ROOT"
