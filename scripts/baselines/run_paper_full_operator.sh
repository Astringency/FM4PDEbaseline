#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-/home/tat512/C01Python/PDEdata}"
OUT="${OUT:-outputs/baselines/paper/full_operator}"
DEVICE="${DEVICE:-cpu}"
TRAIN_SIZE="${TRAIN_SIZE:-50000}"
VAL_SIZE="${VAL_SIZE:-1000}"
TEST_SIZE="${TEST_SIZE:-1000}"
TRAIN_SHARDS="${TRAIN_SHARDS:-5}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEEDS="${SEEDS:-1 2 3}"
PDES="${PDES:-darcy poisson helmholtz nsnonbounded burger reaction_diffusion shallow_water heat wave advection_diffusion steady_heat_conduction}"
BASELINES="${BASELINES:-fno deeponet ifno}"
SCALAR_PARAM_MODE="${SCALAR_PARAM_MODE:-metadata}"
CONFIG="${CONFIG:-baselines/configs/paper.yaml}"

for seed in $SEEDS; do
  for pde in $PDES; do
    for baseline in $BASELINES; do
      python -m baselines.run \
        --experiment-mode paper \
        --baseline "$baseline" --pde "$pde" --task forward \
        --data-root "$DATA_ROOT" --config "$CONFIG" \
        --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" --test-size "$TEST_SIZE" \
        --train-shards "$TRAIN_SHARDS" --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" --seed "$seed" --device "$DEVICE" \
        --scalar-param-mode "$SCALAR_PARAM_MODE" --output-dir "$OUT"
    done
  done
done
