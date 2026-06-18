#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ABLATION="${ABLATION:-}"
case "$ABLATION" in
  sensor_count_ablation|noise_ablation|sensor_mode_ablation|time_varying_sensor_ablation|runtime_budget_ablation|train_size_ablation) ;;
  "")
    echo "Set ABLATION to one of: sensor_count_ablation noise_ablation sensor_mode_ablation time_varying_sensor_ablation runtime_budget_ablation train_size_ablation" >&2
    exit 2
    ;;
  *)
    echo "Unknown ABLATION=$ABLATION" >&2
    exit 2
    ;;
esac

DATA_ROOT="${DATA_ROOT:-/home/tat512/C01Python/PDEdata}"
OUT_ROOT="${OUT_ROOT:-outputs/baselines_large}"
MATRIX="${MATRIX:-$OUT_ROOT/matrices/${ABLATION}.jsonl}"
N_JOBS="${N_JOBS:-1}"
DEVICE="${DEVICE:-cuda}"
export DATA_ROOT OUT_ROOT DEVICE

if [ ! -d "$DATA_ROOT" ]; then
  echo "DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 2
fi

if [ ! -f "$MATRIX" ]; then
  python scripts/experiments/build_matrix.py --config "configs/experiments/${ABLATION}.yaml" --output-root "$OUT_ROOT" --matrix-name "$ABLATION"
fi

total="$(python - "$MATRIX" <<'PY'
import sys
print(sum(1 for line in open(sys.argv[1], encoding="utf-8") if line.strip()))
PY
)"

if [ "$total" -eq 0 ]; then
  echo "No $ABLATION runs in $MATRIX"
  exit 0
fi

seq 0 $((total - 1)) | xargs -I{} -P "$N_JOBS" bash scripts/experiments/05_run_one.sh "$MATRIX" {}
