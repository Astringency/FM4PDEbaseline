#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

bash scripts/baselines/run_paper_full_operator.sh
if [ "${RUN_INVERSE:-1}" != "0" ]; then
  bash scripts/baselines/run_paper_full_inverse.sh
fi
bash scripts/baselines/run_paper_sparse_reconstruction.sh
if [ "${RUN_INVERSE:-1}" != "0" ]; then
  bash scripts/baselines/run_paper_sparse_inverse.sh
fi
bash scripts/baselines/run_paper_physics_da.sh
