#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/baselines_large}"
input="$OUT_ROOT/runs/main_results"
output="$OUT_ROOT/aggregate/main_results"
python -m baselines.aggregate_results "$input" --output-dir "$output"
