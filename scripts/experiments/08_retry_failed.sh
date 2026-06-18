#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/baselines_large}"
RETRY_LIMIT="${RETRY_LIMIT:-3}"
args=(--output-root "$OUT_ROOT" --retry-limit "$RETRY_LIMIT")
if [ -n "${MATRIX:-}" ]; then
  args+=(--matrix "$MATRIX")
fi
python scripts/experiments/retry_failed.py "${args[@]}"
