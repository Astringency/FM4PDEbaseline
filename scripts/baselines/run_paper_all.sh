#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Compatibility entry point. The reviewer-facing native matrix is delegated to
# run_paper_all_native.sh; older per-task scripts remain available for adapted
# supplement/debug runs.
bash scripts/baselines/run_paper_all_native.sh
