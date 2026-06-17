#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-/home/tat512/C01Python/PDEdata}"
OUT="${OUT:-outputs/baselines/time_dependent_da}"

for pde in shallow_water reaction_diffusion nsnonbounded; do
  python -m baselines.run --baseline var4d --pde "$pde" --task sparse_solution --data-root "$DATA_ROOT" --prefer-test --num-sensors 500 --train-size 4 --batch-size 1 --epochs 1 --seed "${SEED:-1}" --output-dir "$OUT"
  python -m baselines.run --baseline vivid --pde "$pde" --task sparse_solution --data-root "$DATA_ROOT" --prefer-test --num-sensors 500 --train-size 4 --batch-size 1 --epochs 1 --seed "${SEED:-1}" --output-dir "$OUT"
done

