#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 2
}

MATRIX="${1:-${MATRIX:-}}"
DATA_ROOT="${DATA_ROOT:-/home/zhangxf/share/zhangxfA100/large_storage/PDEdata/}"
GPUS="${GPUS:-0,1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
RUN_TAIL_LOGS="${RUN_TAIL_LOGS:-1}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-60}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-amortized}"

[ -n "$MATRIX" ] || die "matrix path is required: bash scripts/experiments/08_run_matrix_2gpu_parallel.sh <matrix.jsonl>"
[ -f "$MATRIX" ] || die "matrix not found: $MATRIX"
[ -d "$DATA_ROOT" ] || die "DATA_ROOT does not exist: $DATA_ROOT"
[ -n "$GPUS" ] || die "GPUS must not be empty"
[[ "$JOBS_PER_GPU" =~ ^[0-9]+$ ]] || die "JOBS_PER_GPU must be an integer: $JOBS_PER_GPU"
if [ "$JOBS_PER_GPU" -lt 1 ]; then
  die "JOBS_PER_GPU must be >= 1: $JOBS_PER_GPU"
fi
if [ -n "${N_JOBS:-}" ]; then
  log "WARNING: this script does not use N_JOBS=$N_JOBS; use JOBS_PER_GPU to control concurrency."
fi

if [ -z "${OUT_ROOT:-}" ]; then
  OUT_ROOT="$(cd "$(dirname "$MATRIX")/.." && pwd)"
else
  OUT_ROOT="$OUT_ROOT"
fi
mkdir -p "$OUT_ROOT/launcher_logs"
LAUNCHER_LOG="$OUT_ROOT/launcher_logs/run_matrix_2gpu_parallel_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LAUNCHER_LOG") 2> >(tee -a "$LAUNCHER_LOG" >&2)

total="$(python - "$MATRIX" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
print(sum(1 for line in path.open(encoding="utf-8") if line.strip()))
PY
)"
if [ "$total" -eq 0 ]; then
  die "matrix has zero rows: $MATRIX"
fi

IFS=',' read -r -a gpu_array <<< "$GPUS"
clean_gpus=()
for raw_gpu in "${gpu_array[@]}"; do
  gpu="$(printf '%s' "$raw_gpu" | tr -d '[:space:]')"
  if [ -n "$gpu" ]; then
    clean_gpus+=("$gpu")
  fi
done
if [ "${#clean_gpus[@]}" -eq 0 ]; then
  die "GPUS did not contain any usable GPU ids: $GPUS"
fi

log "script=$(basename "$0") start_time=$(date '+%Y-%m-%d %H:%M:%S')"
log "ROOT=$ROOT"
log "MATRIX=$MATRIX"
log "DATA_ROOT=$DATA_ROOT"
log "OUT_ROOT=$OUT_ROOT"
log "GPUS=${clean_gpus[*]}"
log "JOBS_PER_GPU=$JOBS_PER_GPU"
log "RUN_TAIL_LOGS=$RUN_TAIL_LOGS PROGRESS_INTERVAL_SECONDS=$PROGRESS_INTERVAL_SECONDS"
log "SAVE_CHECKPOINT=$SAVE_CHECKPOINT"
log "total=$total"
log "launcher_log=$LAUNCHER_LOG"

run_gpu_queue() {
  local gpu="$1"
  local offset="$2"
  local stride="$3"
  local last_index=$((total - 1))

  if [ "$offset" -gt "$last_index" ]; then
    log "gpu=$gpu queue empty offset=$offset total=$total"
    return 0
  fi

  log "gpu=$gpu queue start offset=$offset stride=$stride jobs_per_gpu=$JOBS_PER_GPU"
  seq "$offset" "$stride" "$last_index" | xargs -r -n 1 -P "$JOBS_PER_GPU" bash -c '
    set -uo pipefail
    matrix="$1"
    gpu="$2"
    export CUDA_VISIBLE_DEVICES="$gpu"
    # Scheduling controls choose a physical GPU only. Fingerprinted design and
    # loader settings must come from the matrix row unchanged.
    unset ALLOW_ROW_OVERRIDE DEVICE
    export DATA_ROOT="$3"
    export OUT_ROOT="$4"
    export RUN_TAIL_LOGS="$5"
    export PROGRESS_INTERVAL_SECONDS="$6"
    export SAVE_CHECKPOINT="$7"
    idx="$8"
    start_time="$(date "+%Y-%m-%d %H:%M:%S")"
    printf "[%s] child start gpu=%s matrix_index=%s\n" "$start_time" "$gpu" "$idx"
    bash scripts/experiments/05_run_one.sh "$matrix" "$idx"
    status="$?"
    finish_time="$(date "+%Y-%m-%d %H:%M:%S")"
    printf "[%s] child finish gpu=%s matrix_index=%s exit_code=%s\n" "$finish_time" "$gpu" "$idx" "$status"
    exit "$status"
  ' _ "$MATRIX" "$gpu" "$DATA_ROOT" "$OUT_ROOT" "$RUN_TAIL_LOGS" "$PROGRESS_INTERVAL_SECONDS" "$SAVE_CHECKPOINT"
  log "gpu=$gpu queue finish"
}

pids=()
gpu_count="${#clean_gpus[@]}"
for pos in "${!clean_gpus[@]}"; do
  run_gpu_queue "${clean_gpus[$pos]}" "$pos" "$gpu_count" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

log "script=$(basename "$0") finish_time=$(date '+%Y-%m-%d %H:%M:%S') exit_code=$status"
exit "$status"
