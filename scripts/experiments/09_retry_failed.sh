#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

OUT_ROOT="${OUT_ROOT:-outputs/baselines_large}"
RETRY_LIMIT="${RETRY_LIMIT:-3}"
DATA_ROOT="${DATA_ROOT:-}"
DEVICE="${DEVICE:-}"
N_JOBS="${N_JOBS:-}"
TASK_GROUPS="${TASK_GROUPS:-}"
ABLATION="${ABLATION:-}"

log "script=$(basename "$0") start_time=$(date '+%Y-%m-%d %H:%M:%S')"
log "ROOT=$ROOT"
log "DATA_ROOT=${DATA_ROOT:-<unset>}"
log "OUT_ROOT=$OUT_ROOT"
log "DEVICE=${DEVICE:-<unset>}"
log "MATRIX=${MATRIX:-<unset>}"
log "N_JOBS=${N_JOBS:-<unset>} TASK_GROUPS=${TASK_GROUPS:-<unset>} ABLATION=${ABLATION:-<unset>} RETRY_LIMIT=$RETRY_LIMIT"

args=(--output-root "$OUT_ROOT" --retry-limit "$RETRY_LIMIT")
if [ -n "${MATRIX:-}" ]; then
  args+=(--matrix "$MATRIX")
fi
log "running retry_failed.py args=${args[*]}"
python scripts/experiments/retry_failed.py "${args[@]}"
