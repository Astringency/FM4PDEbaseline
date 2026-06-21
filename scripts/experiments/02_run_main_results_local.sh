#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

DATA_ROOT="${DATA_ROOT:-$ROOT/../PDEdata}"
OUT_ROOT="${OUT_ROOT:-outputs/baselines_large}"
MATRIX="${MATRIX:-$OUT_ROOT/matrices/main_results.jsonl}"
N_JOBS="${N_JOBS:-1}"
DEVICE="${DEVICE:-cuda}"
TASK_GROUPS="${TASK_GROUPS:-}"
export DATA_ROOT OUT_ROOT DEVICE

log "script=$(basename "$0") start_time=$(date '+%Y-%m-%d %H:%M:%S')"
log "ROOT=$ROOT"
log "DATA_ROOT=$DATA_ROOT"
log "OUT_ROOT=$OUT_ROOT"
log "DEVICE=$DEVICE"
log "MATRIX=$MATRIX"
log "N_JOBS=$N_JOBS TASK_GROUPS=${TASK_GROUPS:-<unset>}"

if [ ! -d "$DATA_ROOT" ]; then
  log "DATA_ROOT does not exist: $DATA_ROOT"
  exit 2
fi

if [ ! -f "$MATRIX" ]; then
  log "matrix not found; building main_results matrix at $MATRIX"
  python scripts/experiments/build_matrix.py --config configs/experiments/main_results.yaml --output-root "$OUT_ROOT" --matrix-name main_results
fi

matrix_total="$(python - "$MATRIX" <<'PY'
import sys
print(sum(1 for line in open(sys.argv[1], encoding="utf-8") if line.strip()), flush=True)
PY
)"
RUN_MATRIX="$MATRIX"
if [ -n "$TASK_GROUPS" ]; then
  RUN_MATRIX="$OUT_ROOT/matrices/$(basename "$MATRIX" .jsonl).filtered.jsonl"
  log "filtering matrix by TASK_GROUPS=$TASK_GROUPS into $RUN_MATRIX"
  python - "$MATRIX" "$RUN_MATRIX" "$TASK_GROUPS" <<'PY'
import json
import shlex
import sys

src, dst, groups_raw = sys.argv[1:4]
groups = set(shlex.split(groups_raw))
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task_group") in groups:
            out.write(json.dumps(row, sort_keys=True) + "\n")
PY
fi

total="$(python - "$RUN_MATRIX" <<'PY'
import sys
print(sum(1 for line in open(sys.argv[1], encoding="utf-8") if line.strip()), flush=True)
PY
)"

log "matrix path=$MATRIX"
log "run matrix path=$RUN_MATRIX"
log "matrix total rows=$matrix_total"
log "selected rows after filters=$total"
log "N_JOBS=$N_JOBS"

if [ "$total" -eq 0 ]; then
  log "No runs selected"
  exit 0
fi

start_epoch="$(date +%s)"
log "starting local execution matrix=$RUN_MATRIX total=$total N_JOBS=$N_JOBS start_time=$(date '+%Y-%m-%d %H:%M:%S')"
log "starting xargs execution start_time=$(date '+%Y-%m-%d %H:%M:%S') total=$total N_JOBS=$N_JOBS"
seq 0 $((total - 1)) | xargs -I{} -P "$N_JOBS" bash scripts/experiments/05_run_one.sh "$RUN_MATRIX" {}
end_epoch="$(date +%s)"
log "finished local execution end_time=$(date '+%Y-%m-%d %H:%M:%S') elapsed_seconds=$((end_epoch - start_epoch)) total=$total"
