#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/baselines_large}"
mkdir -p "$OUT_ROOT/aggregate"

groups=(full_forward full_inverse sparse_solution_amortized sparse_solution_physics sparse_inverse time_varying)
for group in "${groups[@]}"; do
  input="$OUT_ROOT/runs/task_group=$group"
  output="$OUT_ROOT/aggregate/$group"
  python -m baselines.aggregate_results "$input" --output-dir "$output"
done

python -m baselines.aggregate_results "$OUT_ROOT/runs" --output-dir "$OUT_ROOT/aggregate/all"
