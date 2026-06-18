#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

bash scripts/baselines/run_paper_native_full_operator.sh
bash scripts/baselines/run_paper_native_sparse_reconstruction.sh
bash scripts/baselines/run_paper_static_sparse_inverse.sh
bash scripts/baselines/run_paper_time_varying_da.sh
