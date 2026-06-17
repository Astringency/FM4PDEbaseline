#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-/home/tat512/C01Python/PDEdata}"
OUT="${OUT:-outputs/baselines/physics_opt}"

python -m baselines.run --baseline pinn_sparse --pde burger --task sparse_solution --data-root "$DATA_ROOT" --prefer-test --num-sensors 100 --train-size 4 --batch-size 1 --epochs 1 --seed "${SEED:-1}" --output-dir "$OUT"
python -m baselines.run --baseline pde_opt --pde darcy --task inverse --data-root "$DATA_ROOT" --prefer-test --num-sensors 500 --train-size 4 --batch-size 1 --epochs 1 --seed "${SEED:-1}" --output-dir "$OUT"
python -m baselines.run --baseline pc_bnn --pde poisson --task sparse_solution --data-root "$DATA_ROOT" --prefer-test --num-sensors 100 --train-size 4 --batch-size 1 --epochs 1 --seed "${SEED:-1}" --output-dir "$OUT"

