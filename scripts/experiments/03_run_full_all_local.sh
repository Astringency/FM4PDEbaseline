#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [ "${CONFIRM_FULL_ALL:-0}" != "1" ]; then
  echo "Refusing to start full_all without CONFIRM_FULL_ALL=1" >&2
  exit 2
fi

DATA_ROOT="${DATA_ROOT:-/home/tat512/C01Python/PDEdata}"
OUT_ROOT="${OUT_ROOT:-outputs/baselines_large}"
MATRIX="${MATRIX:-$OUT_ROOT/matrices/full_all.jsonl}"
N_JOBS="${N_JOBS:-1}"
DEVICE="${DEVICE:-cuda}"
TASK_GROUPS="${TASK_GROUPS:-}"
export DATA_ROOT OUT_ROOT DEVICE

if [ ! -d "$DATA_ROOT" ]; then
  echo "DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 2
fi

if [ ! -f "$MATRIX" ]; then
  python scripts/experiments/build_matrix.py --config configs/experiments/full_all.yaml --output-root "$OUT_ROOT" --matrix-name full_all
fi

RUN_MATRIX="$MATRIX"
if [ -n "$TASK_GROUPS" ]; then
  RUN_MATRIX="$OUT_ROOT/matrices/$(basename "$MATRIX" .jsonl).filtered.jsonl"
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
print(sum(1 for line in open(sys.argv[1], encoding="utf-8") if line.strip()))
PY
)"

if [ "$total" -eq 0 ]; then
  echo "No full_all runs selected from $RUN_MATRIX"
  exit 0
fi

seq 0 $((total - 1)) | xargs -I{} -P "$N_JOBS" bash scripts/experiments/05_run_one.sh "$RUN_MATRIX" {}
