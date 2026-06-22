#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

DATA_ROOT="${DATA_ROOT:-/home/zhangxf/share/zhangxfA100/large_storage/PDEdata/}"
OUT_ROOT="${OUT_ROOT:-outputs/main_results_$(date +%Y%m%d_%H%M)}"
MATRIX_NAME="${MATRIX_NAME:-main_results}"
CONFIG="${CONFIG:-configs/experiments/main_results.yaml}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
MATRIX="$OUT_ROOT/matrices/$MATRIX_NAME.jsonl"

log "script=$(basename "$0") start_time=$(date '+%Y-%m-%d %H:%M:%S')"
log "ROOT=$ROOT"
log "DATA_ROOT=$DATA_ROOT"
log "OUT_ROOT=$OUT_ROOT"
log "CONFIG=$CONFIG"
log "MATRIX_NAME=$MATRIX_NAME"
log "MATRIX=$MATRIX"
log "FORCE_REBUILD=$FORCE_REBUILD"

if [ ! -d "$DATA_ROOT" ]; then
  log "DATA_ROOT does not exist: $DATA_ROOT"
  exit 2
fi

if [ ! -f "$CONFIG" ]; then
  log "config not found: $CONFIG"
  exit 2
fi

if [ -f "$MATRIX" ] && [ "$FORCE_REBUILD" != "1" ]; then
  log "matrix already exists; keeping existing file. Set FORCE_REBUILD=1 to rebuild."
else
  mkdir -p "$OUT_ROOT"
  if [ -f "$MATRIX" ] && [ "$FORCE_REBUILD" = "1" ]; then
    log "FORCE_REBUILD=1 removing old matrix outputs for $MATRIX_NAME"
    rm -f \
      "$OUT_ROOT/matrices/$MATRIX_NAME.jsonl" \
      "$OUT_ROOT/matrices/$MATRIX_NAME.tsv" \
      "$OUT_ROOT/matrices/${MATRIX_NAME}_summary.json" \
      "$OUT_ROOT/matrices/${MATRIX_NAME}_skipped.jsonl"
  fi
  log "building matrix"
  python scripts/experiments/build_matrix.py \
    --config "$CONFIG" \
    --output-root "$OUT_ROOT" \
    --matrix-name "$MATRIX_NAME"
fi

if [ ! -f "$MATRIX" ]; then
  log "matrix build did not create expected file: $MATRIX"
  exit 1
fi

total="$(python - "$MATRIX" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
print(sum(1 for line in path.open(encoding="utf-8") if line.strip()))
PY
)"

log "matrix path=$MATRIX"
log "matrix total rows=$total"
log "OUT_ROOT=$OUT_ROOT"

