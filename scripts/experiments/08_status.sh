#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/baselines_large}"
if [ -n "${STATUS_SCOPE+x}" ]; then
  STATUS_SCOPE_EXPLICIT=1
else
  STATUS_SCOPE_EXPLICIT=0
fi
STATUS_SCOPE="${STATUS_SCOPE:-main}"
if [ "${ALL_MATRICES:-0}" = "1" ]; then
  STATUS_SCOPE=all
fi
if [ -n "${MATRIX:-}" ] && [ "$STATUS_SCOPE_EXPLICIT" = "0" ]; then
  STATUS_SCOPE=matrix
fi

args=(--output-root "$OUT_ROOT" --scope "$STATUS_SCOPE")
if [ -n "${MATRIX:-}" ]; then
  args+=(--matrix "$MATRIX")
fi
python scripts/experiments/status.py "${args[@]}"
