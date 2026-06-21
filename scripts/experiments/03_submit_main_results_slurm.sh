#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

OUT_ROOT="${OUT_ROOT:-outputs/baselines_large}"
MATRIX="${MATRIX:-$OUT_ROOT/matrices/main_results.jsonl}"
MAX_ARRAY_CONCURRENT="${MAX_ARRAY_CONCURRENT:-16}"
PARTITION="${PARTITION:-gpu}"
TIME="${TIME:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"
GRES="${GRES:-gpu:1}"
JOB_NAME="${JOB_NAME:-fm4pde_main_results}"
DATA_ROOT="${DATA_ROOT:-$ROOT/../PDEdata}"
DEVICE="${DEVICE:-cuda}"
N_JOBS="${N_JOBS:-}"
TASK_GROUPS="${TASK_GROUPS:-}"
ABLATION="${ABLATION:-}"

log "script=$(basename "$0") start_time=$(date '+%Y-%m-%d %H:%M:%S')"
log "ROOT=$ROOT"
log "DATA_ROOT=$DATA_ROOT"
log "OUT_ROOT=$OUT_ROOT"
log "DEVICE=$DEVICE"
log "MATRIX=$MATRIX"
log "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-<unset>} MAX_ARRAY_CONCURRENT=$MAX_ARRAY_CONCURRENT N_JOBS=${N_JOBS:-<unset>}"
log "TASK_GROUPS=${TASK_GROUPS:-<unset>} ABLATION=${ABLATION:-<unset>} PARTITION=$PARTITION TIME=$TIME CPUS_PER_TASK=$CPUS_PER_TASK MEM=$MEM GRES=$GRES JOB_NAME=$JOB_NAME"

if [ ! -f "$MATRIX" ]; then
  log "matrix not found; building main_results matrix at $MATRIX"
  python scripts/experiments/build_matrix.py --config configs/experiments/main_results.yaml --output-root "$OUT_ROOT" --matrix-name main_results
fi
if [ ! -d "$DATA_ROOT" ]; then
  log "DATA_ROOT does not exist: $DATA_ROOT"
  exit 2
fi

total="$(python - "$MATRIX" <<'PY'
import sys
print(sum(1 for line in open(sys.argv[1], encoding="utf-8") if line.strip()), flush=True)
PY
)"
log "matrix path=$MATRIX"
log "matrix total rows=$total"
log "selected rows after filters=$total"
log "Slurm array=0-$((total - 1))%$MAX_ARRAY_CONCURRENT"
if [ "$total" -eq 0 ]; then
  log "No runs selected"
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
log() {
  printf '[%s] %s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" "\$*" >&2
}
export DATA_ROOT="$DATA_ROOT"
export OUT_ROOT="$OUT_ROOT"
export MATRIX="$MATRIX"
export DEVICE="$DEVICE"
export TASK_INDEX="\${SLURM_ARRAY_TASK_ID}"
log "script=\$(basename "\$0") start_time=\$(date '+%Y-%m-%d %H:%M:%S')"
log "ROOT=$ROOT"
log "DATA_ROOT=\$DATA_ROOT"
log "OUT_ROOT=\$OUT_ROOT"
log "DEVICE=\$DEVICE"
log "MATRIX=\$MATRIX"
log "SLURM_ARRAY_TASK_ID=\${SLURM_ARRAY_TASK_ID:-<unset>} TASK_INDEX=\$TASK_INDEX"
log "hostname=\$(hostname) pid=\$\$ timestamp=\$(date '+%Y-%m-%d %H:%M:%S')"
bash scripts/experiments/05_run_one.sh "\$MATRIX" "\$TASK_INDEX"
EOF

log "Wrote $SBATCH_FILE"
if command -v sbatch >/dev/null 2>&1; then
  log "submitting $SBATCH_FILE"
  sbatch "$SBATCH_FILE"
else
  log "sbatch not found; submit manually with: sbatch $SBATCH_FILE"
fi
