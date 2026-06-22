#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${1:-${OUT_ROOT:-}}"
CONFIRM="${CONFIRM:-0}"
ONLY_RUNNING="${ONLY_RUNNING:-0}"
ONLY_FAILED="${ONLY_FAILED:-0}"
FORCE="${FORCE:-0}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

if [ -z "$OUT_ROOT" ]; then
  log "OUT_ROOT is required: bash scripts/experiments/12_clean_stale_run_locks.sh <OUT_ROOT>"
  exit 2
fi
if [ ! -d "$OUT_ROOT" ]; then
  log "OUT_ROOT does not exist: $OUT_ROOT"
  exit 2
fi
if [ "$ONLY_RUNNING" = "1" ] && [ "$ONLY_FAILED" = "1" ]; then
  log "ONLY_RUNNING=1 and ONLY_FAILED=1 cannot both be set"
  exit 2
fi

active_processes="$(pgrep -af 'python -m baselines\.run|scripts/experiments/run_one\.py' || true)"
if [ -n "$active_processes" ]; then
  log "WARNING: detected active training processes:"
  printf '%s\n' "$active_processes" >&2
  if [ "$CONFIRM" = "1" ] && [ "$FORCE" != "1" ]; then
    log "Refusing to delete locks while runs appear active. Set FORCE=1 as an extra confirmation."
    exit 2
  fi
fi

python - "$OUT_ROOT" "$CONFIRM" "$ONLY_RUNNING" "$ONLY_FAILED" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

out_root = Path(sys.argv[1])
confirm = sys.argv[2] == "1"
only_running = sys.argv[3] == "1"
only_failed = sys.argv[4] == "1"

patterns = []
if only_running:
    patterns = ["run.running"]
elif only_failed:
    patterns = ["run.failed"]
else:
    patterns = ["run.running", "run.failed"]

paths: list[Path] = []
for pattern in patterns:
    paths.extend(sorted(out_root.rglob(pattern)))
paths = sorted(set(paths))

print(f"OUT_ROOT: {out_root}")
print(f"mode: {'delete' if confirm else 'dry-run'}")
print(f"selected markers: {', '.join(patterns)}")
print(f"count: {len(paths)}")
for path in paths:
    print(path)

if confirm:
    for path in paths:
        path.unlink()
    print(f"deleted: {len(paths)}")
else:
    print("dry-run only; set CONFIRM=1 to delete these marker files")
PY

