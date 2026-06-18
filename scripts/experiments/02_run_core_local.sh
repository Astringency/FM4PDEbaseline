#!/usr/bin/env bash
set -euo pipefail

echo "02_run_core_local.sh is a compatibility wrapper; using 02_run_main_results_local.sh" >&2
bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/02_run_main_results_local.sh"
