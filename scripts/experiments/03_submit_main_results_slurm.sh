#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/baselines_large}"
MATRIX="${MATRIX:-$OUT_ROOT/matrices/main_results.jsonl}"
MAX_ARRAY_CONCURRENT="${MAX_ARRAY_CONCURRENT:-16}"
PARTITION="${PARTITION:-gpu}"
TIME="${TIME:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"
GRES="${GRES:-gpu:1}"
JOB_NAME="${JOB_NAME:-fm4pde_main_results}"
DATA_ROOT="${DATA_ROOT:-/home/tat512/C01Python/PDEdata}"
DEVICE="${DEVICE:-cuda}"

if [ ! -f "$MATRIX" ]; then
  python scripts/experiments/build_matrix.py --config configs/experiments/main_results.yaml --output-root "$OUT_ROOT" --matrix-name main_results
fi
if [ ! -d "$DATA_ROOT" ]; then
  echo "DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 2
fi

total="$(python - "$MATRIX" <<'PY'
import sys
print(sum(1 for line in open(sys.argv[1], encoding="utf-8") if line.strip()))
PY
)"
if [ "$total" -eq 0 ]; then
  echo "No rows in $MATRIX"
  exit 0
fi

mkdir -p "$OUT_ROOT/logs/slurm"
SBATCH_FILE="$OUT_ROOT/logs/slurm/${JOB_NAME}.sbatch"
cat > "$SBATCH_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --array=0-$((total - 1))%$MAX_ARRAY_CONCURRENT
#SBATCH --time=$TIME
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEM
#SBATCH --gres=$GRES
#SBATCH --output=$OUT_ROOT/logs/slurm/%A_%a.out
#SBATCH --error=$OUT_ROOT/logs/slurm/%A_%a.err

set -euo pipefail
cd "$ROOT"
export DATA_ROOT="$DATA_ROOT"
export OUT_ROOT="$OUT_ROOT"
export MATRIX="$MATRIX"
export DEVICE="$DEVICE"
export TASK_INDEX="\${SLURM_ARRAY_TASK_ID}"
bash scripts/experiments/05_run_one.sh "\$MATRIX" "\$TASK_INDEX"
EOF

echo "Wrote $SBATCH_FILE"
if command -v sbatch >/dev/null 2>&1; then
  sbatch "$SBATCH_FILE"
else
  echo "sbatch not found; submit manually with: sbatch $SBATCH_FILE"
fi
