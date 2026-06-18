#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-/home/tat512/C01Python/PDEdata}"
OUT="${OUT:-outputs/baselines/paper/native_sparse_reconstruction}"
DEVICE="${DEVICE:-cpu}"
TRAIN_SIZE="${TRAIN_SIZE:-50000}"
VAL_SIZE="${VAL_SIZE:-0}"
TEST_SIZE="${TEST_SIZE:-1000}"
TRAIN_SHARDS="${TRAIN_SHARDS:-5}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEEDS="${SEEDS:-1 2 3}"
SENSOR_COUNTS="${SENSOR_COUNTS:-50 100 250 500 1000}"
SENSOR_MODES="${SENSOR_MODES:-random grid fixed}"
NOISE_LEVELS="${NOISE_LEVELS:-0.0 0.01 0.05 0.10}"
PDES="${PDES:-darcy poisson helmholtz nsnonbounded burger reaction_diffusion shallow_water heat wave advection_diffusion steady_heat_conduction}"
BASELINES="${BASELINES:-recfno senseiver voronoicnn pinn_sparse pc_bnn pde_opt}"
SCALAR_PARAM_MODE="${SCALAR_PARAM_MODE:-metadata}"
DATA_LOADING_MODE="${DATA_LOADING_MODE:-lazy}"
CONFIG="${CONFIG:-baselines/configs/paper.yaml}"
PCBNN_IMPLEMENTATION_MODE="${PCBNN_IMPLEMENTATION_MODE:-official_aligned}"
SKIPPED="${SKIPPED:-$OUT/skipped_combinations.jsonl}"
mkdir -p "$OUT"

for seed in $SEEDS; do
  for pde in $PDES; do
    for baseline in $BASELINES; do
      for sensors in $SENSOR_COUNTS; do
        for sensor_mode in $SENSOR_MODES; do
          method_args=()
          if [ "$baseline" = "pc_bnn" ]; then
            method_args+=(--implementation-mode "$PCBNN_IMPLEMENTATION_MODE" --official-backend pc_bnn)
          fi
          if ! python -m baselines.experiment_matrix --baseline "$baseline" --pde "$pde" --task sparse_solution --sensor-mode "$sensor_mode" --skipped-path "$SKIPPED"; then
            continue
          fi
          for noise in $NOISE_LEVELS; do
            python -m baselines.run \
              --experiment-mode paper \
              --baseline "$baseline" --pde "$pde" --task sparse_solution \
              --data-root "$DATA_ROOT" --config "$CONFIG" \
              --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" --test-size "$TEST_SIZE" \
              --train-shards "$TRAIN_SHARDS" --batch-size "$BATCH_SIZE" \
              --epochs "$EPOCHS" --seed "$seed" --device "$DEVICE" \
              --num-sensors "$sensors" --sensor-mode "$sensor_mode" --noise-level "$noise" \
              --data-loading-mode "$DATA_LOADING_MODE" \
              --scalar-param-mode "$SCALAR_PARAM_MODE" --output-dir "$OUT" \
              "${method_args[@]}"
          done
        done
      done
    done
  done
done
