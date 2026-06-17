#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${OUT:-outputs/baselines/time_dependent_da}" PDES="${PDES:-shallow_water reaction_diffusion nsnonbounded}" BASELINES="${BASELINES:-var4d vivid}" bash scripts/baselines/run_paper_physics_da.sh
