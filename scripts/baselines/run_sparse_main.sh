#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${OUT:-outputs/baselines/sparse_main}" bash scripts/baselines/run_paper_sparse_reconstruction.sh
