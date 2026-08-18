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

DATA_ROOT="${DATA_ROOT:-/home/zhangxf/share/zhangxfA100/large_storage/PDEdata/}"
OUT_ROOT="${OUT_ROOT:-outputs/main_results_$(date +%Y%m%d_%H%M%S)}"
MATRIX_NAME="${MATRIX_NAME:-main_results}"
CONFIG="${CONFIG:-configs/experiments/main_results.yaml}"
DATA_MANIFEST="${DATA_MANIFEST:-}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
MATRIX="$OUT_ROOT/matrices/$MATRIX_NAME.jsonl"

log "script=$(basename "$0") start_time=$(date '+%Y-%m-%d %H:%M:%S')"
log "ROOT=$ROOT"
log "DATA_ROOT=$DATA_ROOT"
log "OUT_ROOT=$OUT_ROOT"
log "CONFIG=$CONFIG"
log "DATA_MANIFEST=${DATA_MANIFEST:-<unset>}"
log "MATRIX_NAME=$MATRIX_NAME"
log "MATRIX=$MATRIX"
log "FORCE_REBUILD=$FORCE_REBUILD"

[ -d "$DATA_ROOT" ] || die "DATA_ROOT does not exist: $DATA_ROOT"
[ -f "$CONFIG" ] || die "config not found: $CONFIG"
[ -n "$DATA_MANIFEST" ] || die \
  "DATA_MANIFEST is required; verify this exact config with scripts/verify_data_protocol.py --full first"
[ -f "$DATA_MANIFEST" ] || die "DATA_MANIFEST not found: $DATA_MANIFEST"

# Refuse historical/path-traversal targets before FORCE_REBUILD can remove
# anything from the selected output root.
python - "$OUT_ROOT" "$MATRIX_NAME" <<'PY'
import sys

from scripts.experiments.build_matrix import _reject_historical_matrix_target

_reject_historical_matrix_target(sys.argv[1], sys.argv[2])
PY

# Validate the full report against the exact YAML before either reusing or
# rebuilding a formal matrix. If a matrix already exists, it must be bound to
# this same report rather than silently ignoring a newly supplied manifest.
python - "$CONFIG" "$DATA_MANIFEST" "$MATRIX" "$FORCE_REBUILD" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.experiments.build_matrix import load_config, load_data_manifest_binding

config, manifest, matrix = map(Path, sys.argv[1:4])
force_rebuild = sys.argv[4]
try:
    experiment = load_config(config)
    manifest_path, manifest_sha256 = load_data_manifest_binding(
        manifest,
        experiment_config_path=config,
        expected_pdes=[str(value) for value in experiment.get("pdes", [])],
    )
except (FileNotFoundError, ValueError) as exc:
    raise SystemExit(f"invalid DATA_MANIFEST for {config}: {exc}") from exc

if matrix.is_file() and force_rebuild != "1":
    validated_rows = 0
    for line_number, line in enumerate(matrix.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        validated_rows += 1
        if row.get("data_manifest_sha256") != manifest_sha256:
            raise SystemExit(
                f"existing matrix {matrix}:{line_number} is bound to a different DATA_MANIFEST; "
                "set FORCE_REBUILD=1"
            )
        bound_path = Path(str(row.get("data_manifest_path", ""))).expanduser().resolve()
        if bound_path != Path(manifest_path):
            raise SystemExit(
                f"existing matrix {matrix}:{line_number} uses DATA_MANIFEST path {bound_path}, "
                f"not {manifest_path}; set FORCE_REBUILD=1"
            )
    if validated_rows == 0:
        raise SystemExit(f"existing matrix {matrix} has zero rows; set FORCE_REBUILD=1")
PY

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
    --matrix-name "$MATRIX_NAME" \
    --data-manifest "$DATA_MANIFEST"
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

if [ "$total" -eq 0 ]; then
  die "matrix has zero rows: $MATRIX"
fi

log "matrix path=$MATRIX"
log "matrix total rows=$total"
log "OUT_ROOT=$OUT_ROOT"
