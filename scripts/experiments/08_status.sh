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
N_JOBS="${N_JOBS:-}"
TASK_GROUPS="${TASK_GROUPS:-}"
ABLATION="${ABLATION:-}"
if [ -n "${STATUS_SCOPE+x}" ]; then
  STATUS_SCOPE_EXPLICIT=1
else
  STATUS_SCOPE_EXPLICIT=0
fi
STATUS_SCOPE="${STATUS_SCOPE:-main}"
if [ "${ALL_MATRICES:-0}" = "1" ]; then
  STATUS_SCOPE=all
fi
if [ -n "${MATRIX:-}" ] && [ "$STATUS_SCOPE_EXPLICIT" = "0" ]; then
  STATUS_SCOPE=matrix
fi

log "script=$(basename "$0") start_time=$(date '+%Y-%m-%d %H:%M:%S')"
log "ROOT=$ROOT"
log "DATA_ROOT=${DATA_ROOT:-<unset>}"
log "OUT_ROOT=$OUT_ROOT"
log "DEVICE=${DEVICE:-<unset>}"
log "MATRIX=${MATRIX:-<unset>}"
log "N_JOBS=${N_JOBS:-<unset>} TASK_GROUPS=${TASK_GROUPS:-<unset>} ABLATION=${ABLATION:-<unset>} STATUS_SCOPE=$STATUS_SCOPE ALL_MATRICES=${ALL_MATRICES:-0}"

args=(--output-root "$OUT_ROOT" --scope "$STATUS_SCOPE")
if [ -n "${MATRIX:-}" ]; then
  args+=(--matrix "$MATRIX")
fi
log "running status.py args=${args[*]}"
python scripts/experiments/status.py "${args[@]}"
