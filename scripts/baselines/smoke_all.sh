#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

COMMON=(--experiment-mode smoke --dry-run --synthetic-data --synthetic-resolution 32 --train-size 8 --val-size 2 --test-size 4 --batch-size 2 --epochs 1 --num-sensors 50 --sensor-mode random --noise-level 0.0 --output-dir outputs/baselines/smoke)

python -m baselines.run --baseline fno --pde darcy --task forward "${COMMON[@]}"
python -m baselines.run --baseline deeponet --pde poisson --task forward "${COMMON[@]}"
python -m baselines.run --baseline voronoicnn --pde darcy --task sparse_solution "${COMMON[@]}"
python -m baselines.run --baseline recfno --pde poisson --task sparse_solution "${COMMON[@]}"
python -m baselines.run --baseline senseiver --pde shallow_water --task sparse_solution "${COMMON[@]}"
python -m baselines.run --baseline ifno --pde helmholtz --task inverse "${COMMON[@]}"
python -m baselines.run --baseline pinn_sparse --pde burger --task sparse_solution "${COMMON[@]}"
python -m baselines.run --baseline pc_bnn --pde poisson --task sparse_solution "${COMMON[@]}"
python -m baselines.run --baseline pde_opt --pde darcy --task inverse "${COMMON[@]}"
python -m baselines.run --baseline var4d --pde shallow_water --task sparse_solution "${COMMON[@]}"
python -m baselines.run --baseline vivid --pde shallow_water --task sparse_solution "${COMMON[@]}"
python -m baselines.run --baseline fno --pde heat --task forward --scalar-param-mode metadata "${COMMON[@]}"
