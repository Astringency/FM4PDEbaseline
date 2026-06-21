#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

ABLATION="${ABLATION:-}"
log "script=$(basename "$0") start_time=$(date '+%Y-%m-%d %H:%M:%S')"
log "ROOT=$ROOT"
log "ABLATION=${ABLATION:-<unset>}"
case "$ABLATION" in
  sensor_count_ablation|noise_ablation|sensor_mode_ablation|time_varying_sensor_ablation|runtime_budget_ablation|train_size_ablation) ;;
  "")
    log "Set ABLATION to one of: sensor_count_ablation noise_ablation sensor_mode_ablation time_varying_sensor_ablation runtime_budget_ablation train_size_ablation"
    exit 2
    ;;
  *)
    log "Unknown ABLATION=$ABLATION"
    exit 2
    ;;
esac

DATA_ROOT="${DATA_ROOT:-$ROOT/../PDEdata}"
OUT_ROOT="${OUT_ROOT:-outputs/baselines_large}"
MATRIX="${MATRIX:-$OUT_ROOT/matrices/${ABLATION}.jsonl}"
N_JOBS="${N_JOBS:-1}"
DEVICE="${DEVICE:-cuda}"
TASK_GROUPS="${TASK_GROUPS:-}"
export DATA_ROOT OUT_ROOT DEVICE

log "DATA_ROOT=$DATA_ROOT"
log "OUT_ROOT=$OUT_ROOT"
log "DEVICE=$DEVICE"
log "MATRIX=$MATRIX"
log "N_JOBS=$N_JOBS TASK_GROUPS=${TASK_GROUPS:-<unset>} ABLATION=$ABLATION"

if [ ! -d "$DATA_ROOT" ]; then
  log "DATA_ROOT does not exist: $DATA_ROOT"
  exit 2
fi

if [ ! -f "$MATRIX" ]; then
  log "matrix not found; building $ABLATION matrix at $MATRIX"
  python scripts/experiments/build_matrix.py --config "configs/experiments/${ABLATION}.yaml" --output-root "$OUT_ROOT" --matrix-name "$ABLATION"
fi

matrix_total="$(python - "$MATRIX" <<'PY'
import sys
print(sum(1 for line in open(sys.argv[1], encoding="utf-8") if line.strip()), flush=True)
PY
)"
total="$matrix_total"

log "matrix path=$MATRIX"
log "matrix total rows=$matrix_total"
log "selected rows after filters=$total"
log "N_JOBS=$N_JOBS"

if [ "$total" -eq 0 ]; then
  log "No runs selected"
  exit 0
fi

start_epoch="$(date +%s)"
log "starting local execution matrix=$MATRIX total=$total N_JOBS=$N_JOBS start_time=$(date '+%Y-%m-%d %H:%M:%S')"
log "starting xargs execution start_time=$(date '+%Y-%m-%d %H:%M:%S') total=$total N_JOBS=$N_JOBS"
seq 0 $((total - 1)) | xargs -I{} -P "$N_JOBS" bash scripts/experiments/05_run_one.sh "$MATRIX" {}
end_epoch="$(date +%s)"
log "finished local execution end_time=$(date '+%Y-%m-%d %H:%M:%S') elapsed_seconds=$((end_epoch - start_epoch)) total=$total"
