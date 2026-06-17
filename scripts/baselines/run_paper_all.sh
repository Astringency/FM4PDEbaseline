#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

bash scripts/baselines/run_paper_full_operator.sh
bash scripts/baselines/run_paper_sparse_reconstruction.sh
bash scripts/baselines/run_paper_physics_da.sh
