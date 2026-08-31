#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORCHESTRATOR_PYTHON="${ORCHESTRATOR_PYTHON:-${PYTHON:-python}}"

exec "$ORCHESTRATOR_PYTHON" "$ROOT/scripts/run_eval.py" "$@"
