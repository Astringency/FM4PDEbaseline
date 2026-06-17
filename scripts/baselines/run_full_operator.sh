#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${OUT:-outputs/baselines/full_operator}" bash scripts/baselines/run_paper_full_operator.sh
