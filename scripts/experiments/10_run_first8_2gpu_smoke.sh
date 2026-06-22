#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

DATA_ROOT="${DATA_ROOT:-/home/zhangxf/share/zhangxfA100/large_storage/PDEdata/}"
OUT_ROOT="${OUT_ROOT:-outputs/main_results_$(date +%Y%m%d_%H%M)}"
FIRST_N="${FIRST_N:-8}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
NUM_WORKERS_PER_RUN="${NUM_WORKERS_PER_RUN:-2}"
MATRIX="$OUT_ROOT/matrices/main_results.jsonl"
SMOKE_MATRIX="$OUT_ROOT/matrices/main_results_first${FIRST_N}.jsonl"
export DATA_ROOT OUT_ROOT JOBS_PER_GPU NUM_WORKERS_PER_RUN

if ! [[ "$FIRST_N" =~ ^[0-9]+$ ]] || [ "$FIRST_N" -lt 1 ]; then
  log "FIRST_N must be an integer >= 1: $FIRST_N"
  exit 2
fi

log "script=$(basename "$0") start_time=$(date '+%Y-%m-%d %H:%M:%S')"
log "DATA_ROOT=$DATA_ROOT"
log "OUT_ROOT=$OUT_ROOT"
log "FIRST_N=$FIRST_N"
log "MATRIX=$MATRIX"
log "SMOKE_MATRIX=$SMOKE_MATRIX"

if [ ! -f "$MATRIX" ]; then
  log "main matrix not found; building main_results matrix"
  bash scripts/experiments/07_build_main_results_matrix.sh
else
  log "main matrix exists; reusing $MATRIX"
fi

mkdir -p "$(dirname "$SMOKE_MATRIX")"
python - "$MATRIX" "$SMOKE_MATRIX" "$FIRST_N" <<'PY'
from pathlib import Path
import sys
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
n = int(sys.argv[3])
count = 0
with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in fin:
        if line.strip():
            fout.write(line)
            count += 1
            if count >= n:
                break
if count == 0:
    raise SystemExit(f"source matrix has no rows: {src}")
print(count)
PY

bash scripts/experiments/08_run_matrix_2gpu_parallel.sh "$SMOKE_MATRIX"

