#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-/home/tat512/C01Python/PDEdata}"
OUT="${OUT:-outputs/baselines/sparse_main}"
SEED="${SEED:-1}"

for pde in darcy poisson helmholtz nsnonbounded burger reaction_diffusion shallow_water; do
  for baseline in recfno senseiver voronoicnn; do
    python -m baselines.run \
      --baseline "$baseline" --pde "$pde" --task sparse_solution \
      --data-root "$DATA_ROOT" --prefer-test \
      --num-sensors "${NUM_SENSORS:-500}" --sensor-mode "${SENSOR_MODE:-random}" \
      --noise-level "${NOISE_LEVEL:-0.0}" --train-size "${TRAIN_SIZE:-64}" \
      --batch-size "${BATCH_SIZE:-8}" --epochs "${EPOCHS:-1}" --seed "$SEED" \
      --output-dir "$OUT"
  done
done

