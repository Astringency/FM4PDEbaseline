#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-/home/tat512/C01Python/PDEdata}"
OUT="${OUT:-outputs/baselines/paper/static_sparse_inverse}"
DEVICE="${DEVICE:-cpu}"
TRAIN_SIZE="${TRAIN_SIZE:-50000}"
VAL_SIZE="${VAL_SIZE:-1000}"
TEST_SIZE="${TEST_SIZE:-1000}"
TRAIN_SHARDS="${TRAIN_SHARDS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEEDS="${SEEDS:-1 2 3}"
SENSOR_COUNTS="${SENSOR_COUNTS:-50 100 250 500 1000}"
SENSOR_MODES="${SENSOR_MODES:-random grid fixed}"
NOISE_LEVELS="${NOISE_LEVELS:-0.0 0.01 0.05 0.10}"
PDES="${PDES:-poisson helmholtz darcy steady_heat_conduction}"
BASELINES="${BASELINES:-pinn_sparse pde_opt}"
SCALAR_PARAM_MODE="${SCALAR_PARAM_MODE:-metadata}"
DATA_LOADING_MODE="${DATA_LOADING_MODE:-lazy}"
SENSOR_BUDGET_MODE="${SENSOR_BUDGET_MODE:-per_time}"
CONFIG="${CONFIG:-baselines/configs/paper.yaml}"
SKIPPED="${SKIPPED:-$OUT/skipped_combinations.jsonl}"
mkdir -p "$OUT"

for seed in $SEEDS; do
  for pde in $PDES; do
    for baseline in $BASELINES; do
      for sensors in $SENSOR_COUNTS; do
        for sensor_mode in $SENSOR_MODES; do
          if ! python -m baselines.experiment_matrix --baseline "$baseline" --pde "$pde" --task sparse_inverse --sensor-mode "$sensor_mode" --skipped-path "$SKIPPED"; then
            continue
          fi
          for noise in $NOISE_LEVELS; do
            python -m baselines.run \
              --experiment-mode paper \
              --baseline "$baseline" --pde "$pde" --task sparse_inverse \
              --data-root "$DATA_ROOT" --config "$CONFIG" \
              --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" --test-size "$TEST_SIZE" \
              --train-shards "$TRAIN_SHARDS" --batch-size "$BATCH_SIZE" \
              --seed "$seed" --device "$DEVICE" \
              --num-sensors "$sensors" --sensor-mode "$sensor_mode" --noise-level "$noise" \
              --data-loading-mode "$DATA_LOADING_MODE" \
              --sensor-budget-mode "$SENSOR_BUDGET_MODE" \
              --scalar-param-mode "$SCALAR_PARAM_MODE" --output-dir "$OUT"
          done
        done
      done
    done
  done
done
