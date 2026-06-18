#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-/home/tat512/C01Python/PDEdata}"
OUT="${OUT:-outputs/baselines/paper/native_full_operator}"
DEVICE="${DEVICE:-cpu}"
TRAIN_SIZE="${TRAIN_SIZE:-50000}"
VAL_SIZE="${VAL_SIZE:-0}"
TEST_SIZE="${TEST_SIZE:-1000}"
TRAIN_SHARDS="${TRAIN_SHARDS:-5}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEEDS="${SEEDS:-1 2 3}"
PDES="${PDES:-darcy poisson helmholtz nsnonbounded burger reaction_diffusion shallow_water heat wave advection_diffusion steady_heat_conduction}"
FORWARD_BASELINES="${FORWARD_BASELINES:-fno deeponet ifno}"
INVERSE_BASELINES="${INVERSE_BASELINES:-ifno}"
SCALAR_PARAM_MODE_WAS_SET="${SCALAR_PARAM_MODE+x}"
SCALAR_PARAM_MODE="${SCALAR_PARAM_MODE:-metadata}"
SUPERVISED_SCALAR_PARAM_MODE="${SUPERVISED_SCALAR_PARAM_MODE:-materialize}"
DATA_LOADING_MODE="${DATA_LOADING_MODE:-lazy}"
CONFIG="${CONFIG:-baselines/configs/paper.yaml}"
IFNO_IMPLEMENTATION_MODE="${IFNO_IMPLEMENTATION_MODE:-official_aligned}"
SKIPPED="${SKIPPED:-$OUT/skipped_combinations.jsonl}"
mkdir -p "$OUT"

is_future_pde() {
  case "$1" in
    heat|wave|advection_diffusion|steady_heat_conduction) return 0 ;;
    *) return 1 ;;
  esac
}

run_one() {
  local task="$1"
  local baseline="$2"
  local pde="$3"
  local seed="$4"
  if ! python -m baselines.experiment_matrix --baseline "$baseline" --pde "$pde" --task "$task" --skipped-path "$SKIPPED"; then
    return 0
  fi
  local scalar_mode="$SCALAR_PARAM_MODE"
  if is_future_pde "$pde" && [ -z "$SCALAR_PARAM_MODE_WAS_SET" ]; then
    scalar_mode="$SUPERVISED_SCALAR_PARAM_MODE"
  fi
  local method_args=()
  if [ "$baseline" = "ifno" ]; then
    method_args+=(--implementation-mode "$IFNO_IMPLEMENTATION_MODE" --official-backend ifno)
  fi
  python -m baselines.run \
    --experiment-mode paper \
    --baseline "$baseline" --pde "$pde" --task "$task" \
    --data-root "$DATA_ROOT" --config "$CONFIG" \
    --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" --test-size "$TEST_SIZE" \
    --train-shards "$TRAIN_SHARDS" --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" --seed "$seed" --device "$DEVICE" \
    --data-loading-mode "$DATA_LOADING_MODE" \
    --scalar-param-mode "$scalar_mode" --output-dir "$OUT" \
    "${method_args[@]}"
}

for seed in $SEEDS; do
  for pde in $PDES; do
    for baseline in $FORWARD_BASELINES; do
      run_one forward "$baseline" "$pde" "$seed"
    done
    for baseline in $INVERSE_BASELINES; do
      run_one inverse "$baseline" "$pde" "$seed"
    done
  done
done
