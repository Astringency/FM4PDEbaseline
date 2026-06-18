#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/baselines_large}"
args=(--output-root "$OUT_ROOT")
if [ -n "${MATRIX:-}" ]; then
  args+=(--matrix "$MATRIX")
fi
python scripts/experiments/status.py "${args[@]}"
