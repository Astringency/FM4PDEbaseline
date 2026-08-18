#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 2
}

OUT_ROOT="${OUT_ROOT:-outputs/baselines_large}"
DATA_ROOT="${DATA_ROOT:-}"
DEVICE="${DEVICE:-}"
MATRIX="${MATRIX:-}"
N_JOBS="${N_JOBS:-}"
TASK_GROUPS="${TASK_GROUPS:-}"
ABLATION="${ABLATION:-}"
DATA_MANIFEST_DIR="${DATA_MANIFEST_DIR:-}"

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

log "script=$(basename "$0") start_time=$(date '+%Y-%m-%d %H:%M:%S')"
log "ROOT=$ROOT"
log "DATA_ROOT=${DATA_ROOT:-<unset>}"
log "OUT_ROOT=$OUT_ROOT"
log "DEVICE=${DEVICE:-<unset>}"
log "MATRIX=${MATRIX:-<unset>}"
log "N_JOBS=${N_JOBS:-<unset>} TASK_GROUPS=${TASK_GROUPS:-<unset>} ABLATION=${ABLATION:-<unset>}"
log "DATA_MANIFEST_DIR=${DATA_MANIFEST_DIR:-<unset>}"

[ -n "$DATA_MANIFEST_DIR" ] || die \
  "DATA_MANIFEST_DIR is required; run a full verification for every config before building formal matrices"
[ -d "$DATA_MANIFEST_DIR" ] || die "DATA_MANIFEST_DIR does not exist: $DATA_MANIFEST_DIR"

# Refuse the immutable historical namespace before mkdir/rm below.
python - "$OUT_ROOT" <<'PY'
import sys

from scripts.experiments.build_matrix import _reject_historical_matrix_target

_reject_historical_matrix_target(sys.argv[1], "sanity_main")
PY

# Each full report binds the exact experiment YAML hash, so a single manifest
# cannot be reused across these distinct configs.
for matrix in "${matrices[@]}"; do
  config="configs/experiments/${matrix}.yaml"
  DATA_MANIFEST="${DATA_MANIFEST_DIR}/${matrix}/full/data_protocol_report.json"
  [ -f "$config" ] || die "config not found: $config"
  [ -f "$DATA_MANIFEST" ] || die \
    "full data manifest not found for $config: $DATA_MANIFEST"
  python - "$config" "$DATA_MANIFEST" <<'PY'
from pathlib import Path
import sys

from scripts.experiments.build_matrix import load_config, load_data_manifest_binding

config, manifest = map(Path, sys.argv[1:])
try:
    experiment = load_config(config)
    load_data_manifest_binding(
        manifest,
        experiment_config_path=config,
        expected_pdes=[str(value) for value in experiment.get("pdes", [])],
    )
except (FileNotFoundError, ValueError) as exc:
    raise SystemExit(f"invalid full data manifest for {config}: {exc}") from exc
PY
done

mkdir -p "$OUT_ROOT/matrices"
log "removing stale skipped summary $OUT_ROOT/skipped_combinations.jsonl"
rm -f "$OUT_ROOT/skipped_combinations.jsonl"

start_epoch="$(date +%s)"
for matrix in "${matrices[@]}"; do
  matrix_start="$(date +%s)"
  DATA_MANIFEST="${DATA_MANIFEST_DIR}/${matrix}/full/data_protocol_report.json"
  log "building matrix=$matrix config=configs/experiments/${matrix}.yaml data_manifest=$DATA_MANIFEST output_root=$OUT_ROOT"
  python scripts/experiments/build_matrix.py \
    --config "configs/experiments/${matrix}.yaml" \
    --output-root "$OUT_ROOT" \
    --matrix-name "$matrix" \
    --data-manifest "$DATA_MANIFEST"
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
