#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${OUT:-outputs/baselines/paper}"
AGG_OUT="${AGG_OUT:-outputs/baselines/paper/aggregate}"

python -m baselines.aggregate_results "$OUT" --output-dir "$AGG_OUT" --latex-tex
