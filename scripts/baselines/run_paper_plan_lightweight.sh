#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export OUT="${OUT:-outputs/baselines/paper/lightweight_plan}"
export DEVICE="${DEVICE:-cpu}"
export TRAIN_SIZE="${TRAIN_SIZE:-50000}"
export VAL_SIZE="${VAL_SIZE:-0}"
export TEST_SIZE="${TEST_SIZE:-1000}"
export SEEDS="${SEEDS:-1}"
export SENSOR_COUNTS="${SENSOR_COUNTS:-100 500}"
export SENSOR_MODES="${SENSOR_MODES:-random grid}"
export NOISE_LEVELS="${NOISE_LEVELS:-0.0 0.05}"

PDES="${PDES:-darcy reaction_diffusion shallow_water heat steady_heat_conduction}" \
BASELINES="${BASELINES:-fno deeponet ifno}" \
bash scripts/baselines/run_paper_full_operator.sh

PDES="${SPARSE_PDES:-darcy reaction_diffusion shallow_water heat steady_heat_conduction}" \
BASELINES="${SPARSE_BASELINES:-recfno senseiver voronoicnn}" \
bash scripts/baselines/run_paper_sparse_reconstruction.sh

PDES="${PHYSICS_PDES:-reaction_diffusion shallow_water heat}" \
BASELINES="${PHYSICS_BASELINES:-pinn_sparse pc_bnn pde_opt var4d vivid}" \
BATCH_SIZE="${PHYSICS_BATCH_SIZE:-1}" \
bash scripts/baselines/run_paper_physics_da.sh
