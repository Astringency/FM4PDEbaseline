#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-/home/tat512/C01Python/PDEdata}"
OUT="${OUT:-outputs/baselines/tuning_lightweight}"
DEVICE="${DEVICE:-cpu}"
CONFIG="${CONFIG:-baselines/configs/tuning.yaml}"
TRAIN_SIZE="${TRAIN_SIZE:-256}"
VAL_SIZE="${VAL_SIZE:-64}"
TEST_SIZE="${TEST_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-5}"
SEED="${SEED:-1}"
SENSOR_BUDGET_MODE="${SENSOR_BUDGET_MODE:-per_time}"
mkdir -p "$OUT"

run_one() {
  local baseline="$1"
  local pde="$2"
  local task="$3"
  shift 3
  python -m baselines.run \
    --experiment-mode debug \
    --baseline "$baseline" --pde "$pde" --task "$task" \
    --data-root "$DATA_ROOT" --config "$CONFIG" \
    --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" --test-size "$TEST_SIZE" \
    --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" --seed "$SEED" --device "$DEVICE" \
    --sensor-budget-mode "$SENSOR_BUDGET_MODE" \
    --data-loading-mode lazy \
    --output-dir "$OUT" \
    "$@"
}

for width in 32 64; do
  run_one fno darcy forward --method-override width="$width" --method-override modes1=12 --method-override modes2=12 --method-override layers=3
done

for hidden in 128 256; do
  run_one deeponet poisson forward --method-override hidden="$hidden" --method-override basis=64
done

for cycle in 0.05 0.1; do
  run_one ifno darcy inverse --implementation-mode official_aligned --official-backend ifno --method-override cycle_weight="$cycle" --method-override width=32 --method-override layers=3
done

for token_dim in 64 128; do
  run_one senseiver reaction_diffusion sparse_solution --num-sensors 50 --sensor-mode time_varying --load-full-trajectory --task-group time_varying --batch-size 1 --method-override token_dim="$token_dim"
done

for refine in 50 100; do
  run_one vivid reaction_diffusion sparse_solution --num-sensors 50 --sensor-mode time_varying --load-full-trajectory --task-group time_varying --batch-size 1 --implementation-mode official_aligned --official-backend vivid --method-override uses_official_inverse_observation_operator=true --method-override refine_steps="$refine"
done

python -m baselines.aggregate_results "$OUT" --output-dir "$OUT/aggregate"
