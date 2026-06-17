#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-/home/tat512/C01Python/PDEdata}"
OUT="${OUT:-outputs/baselines/full_operator}"

for pde in darcy poisson helmholtz reaction_diffusion shallow_water nsnonbounded burger; do
  for baseline in fno deeponet ifno; do
    python -m baselines.run \
      --baseline "$baseline" --pde "$pde" --task forward \
      --data-root "$DATA_ROOT" --prefer-test \
      --train-size "${TRAIN_SIZE:-64}" --batch-size "${BATCH_SIZE:-8}" \
      --epochs "${EPOCHS:-1}" --seed "${SEED:-1}" --output-dir "$OUT"
  done
done

