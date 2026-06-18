#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MATRIX="${1:-${MATRIX:-}}"
TASK_INDEX="${2:-${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}}"

python scripts/experiments/run_one.py "$MATRIX" "$TASK_INDEX"
