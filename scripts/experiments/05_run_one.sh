#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

MATRIX="${1:-${MATRIX:-}}"
TASK_INDEX="${2:-${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}}"

log "script=$(basename "$0") start_time=$(date '+%Y-%m-%d %H:%M:%S')"
log "ROOT=$ROOT"
log "DATA_ROOT=${DATA_ROOT:-<unset>}"
log "OUT_ROOT=${OUT_ROOT:-<unset>}"
log "DEVICE=${DEVICE:-<unset>}"
log "MATRIX=$MATRIX"
log "TASK_INDEX=$TASK_INDEX SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-<unset>}"
log "hostname=$(hostname) pid=$$ timestamp=$(date '+%Y-%m-%d %H:%M:%S')"

python scripts/experiments/run_one.py "$MATRIX" "$TASK_INDEX"
