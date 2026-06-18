#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/baselines_large}"

declare -A ablation_dirs=(
  [sensor_count_ablation]=sensor_count
  [noise_ablation]=noise
  [sensor_mode_ablation]=sensor_mode
  [time_varying_sensor_ablation]=time_varying
  [runtime_budget_ablation]=runtime_budget
  [train_size_ablation]=train_size
)

for matrix in sensor_count_ablation noise_ablation sensor_mode_ablation time_varying_sensor_ablation runtime_budget_ablation train_size_ablation; do
  key="${ablation_dirs[$matrix]}"
  input="$OUT_ROOT/runs/ablations/ablation=$key"
  output="$OUT_ROOT/aggregate/ablations/$key"
  python -m baselines.aggregate_results "$input" --output-dir "$output"
done
