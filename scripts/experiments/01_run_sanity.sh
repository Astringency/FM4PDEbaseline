#!/usr/bin/env bash
set -euo pipefail

echo "01_run_sanity.sh is a compatibility wrapper; using 01_run_sanity_main.sh" >&2
bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/01_run_sanity_main.sh"
