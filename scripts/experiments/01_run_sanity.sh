#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-/home/tat512/C01Python/PDEdata}"
OUT_ROOT="${OUT_ROOT:-outputs/baselines_large}"
MATRIX="${MATRIX:-$OUT_ROOT/matrices/sanity.jsonl}"
N_JOBS="${N_JOBS:-1}"
FAIL_FAST="${FAIL_FAST:-1}"
DEVICE="${DEVICE:-cuda}"
export DATA_ROOT OUT_ROOT DEVICE

if [ ! -d "$DATA_ROOT" ]; then
  echo "DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 2
fi

if [ ! -f "$MATRIX" ]; then
  python scripts/experiments/build_matrix.py --config configs/experiments/sanity.yaml --output-root "$OUT_ROOT" --matrix-name sanity
fi

total="$(python - "$MATRIX" <<'PY'
import sys
print(sum(1 for line in open(sys.argv[1], encoding="utf-8") if line.strip()))
PY
)"

if [ "$total" -eq 0 ]; then
  echo "No sanity runs in $MATRIX"
  exit 0
fi

if [ "$N_JOBS" -le 1 ]; then
  for idx in $(seq 0 $((total - 1))); do
    if [ "$FAIL_FAST" = "1" ]; then
      bash scripts/experiments/05_run_one.sh "$MATRIX" "$idx"
    else
      bash scripts/experiments/05_run_one.sh "$MATRIX" "$idx" || true
    fi
  done
else
  seq 0 $((total - 1)) | xargs -I{} -P "$N_JOBS" bash scripts/experiments/05_run_one.sh "$MATRIX" {}
fi
