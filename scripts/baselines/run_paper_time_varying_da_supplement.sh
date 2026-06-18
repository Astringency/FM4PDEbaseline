#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-/home/tat512/C01Python/PDEdata}"
OUT="${OUT:-outputs/baselines/paper/time_varying_da_supplement}"
DEVICE="${DEVICE:-cpu}"
TRAIN_SIZE="${TRAIN_SIZE:-50000}"
VAL_SIZE="${VAL_SIZE:-0}"
TEST_SIZE="${TEST_SIZE:-1000}"
TRAIN_SHARDS="${TRAIN_SHARDS:-5}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEEDS="${SEEDS:-1 2 3}"
SENSOR_COUNTS="${SENSOR_COUNTS:-50 100 250 500}"
NOISE_LEVELS="${NOISE_LEVELS:-0.0 0.01 0.05}"
PDES="${PDES:-nsnonbounded burger reaction_diffusion shallow_water}"
BASELINES="${BASELINES:-vivid}"
SCALAR_PARAM_MODE="${SCALAR_PARAM_MODE:-metadata}"
DATA_LOADING_MODE="${DATA_LOADING_MODE:-lazy}"
SKIPPED="${SKIPPED:-$OUT/skipped_combinations.jsonl}"
CONFIG="$OUT/vivid_style_adapted.yaml"
mkdir -p "$OUT"
cat > "$CONFIG" <<'YAML'
method:
  implementation_mode: adapted
  official_backend: local
  train_inverse_operator: true
  uses_official_inverse_observation_operator: false
  epochs: 200
  lr: 0.005
  refine_steps: 300
  lambda_obs: 1.0
  lambda_inverse_operator: 0.25
  lambda_background: 0.05
  lambda_int: 0.01
  lambda_bc: 0.01
  lambda_ic: 0.01
YAML

for seed in $SEEDS; do
  for pde in $PDES; do
    for baseline in $BASELINES; do
      for sensors in $SENSOR_COUNTS; do
        for noise in $NOISE_LEVELS; do
          python -m baselines.run \
            --experiment-mode paper \
            --baseline "$baseline" --pde "$pde" --task sparse_solution \
            --data-root "$DATA_ROOT" --config "$CONFIG" \
            --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" --test-size "$TEST_SIZE" \
            --train-shards "$TRAIN_SHARDS" --batch-size "$BATCH_SIZE" \
            --epochs "$EPOCHS" --seed "$seed" --device "$DEVICE" \
            --num-sensors "$sensors" --sensor-mode time_varying --noise-level "$noise" \
            --task-group time_varying --load-full-trajectory \
            --data-loading-mode "$DATA_LOADING_MODE" \
            --scalar-param-mode "$SCALAR_PARAM_MODE" --output-dir "$OUT"
        done
      done
    done
  done
done
