#!/usr/bin/env bash
set -euo pipefail

# Example:
# DATA_ROOT=/home/zhangxf/share/zhangxfA100/large_storage/PDEdata/ \
# OUT_ROOT=outputs/main_results_20260622_1500 \
# GPUS=0,1 \
# JOBS_PER_GPU=2 \
# bash scripts/experiments/09_run_main_results_2gpu.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

DATA_ROOT="${DATA_ROOT:-/home/zhangxf/share/zhangxfA100/large_storage/PDEdata/}"
OUT_ROOT="${OUT_ROOT:-outputs/main_results_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
GPUS="${GPUS:-0,1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
NUM_WORKERS_PER_RUN="${NUM_WORKERS_PER_RUN:-0}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-amortized}"
MATRIX="$OUT_ROOT/matrices/main_results.jsonl"
export DATA_ROOT OUT_ROOT DEVICE GPUS JOBS_PER_GPU NUM_WORKERS_PER_RUN SAVE_CHECKPOINT

log "script=$(basename "$0") start_time=$(date '+%Y-%m-%d %H:%M:%S')"
log "DATA_ROOT=$DATA_ROOT"
log "OUT_ROOT=$OUT_ROOT"
log "MATRIX=$MATRIX"
log "DEVICE=$DEVICE GPUS=$GPUS JOBS_PER_GPU=$JOBS_PER_GPU NUM_WORKERS_PER_RUN=$NUM_WORKERS_PER_RUN"
log "SAVE_CHECKPOINT=$SAVE_CHECKPOINT"

if [ ! -f "$MATRIX" ]; then
  log "matrix not found; building main_results matrix"
  bash scripts/experiments/07_build_main_results_matrix.sh
else
  log "matrix exists; reusing $MATRIX"
fi

bash scripts/experiments/08_run_matrix_2gpu_parallel.sh "$MATRIX"
