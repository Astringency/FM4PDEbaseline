#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${OUT:-outputs/baselines/physics_opt}" bash scripts/baselines/run_paper_physics_da.sh
