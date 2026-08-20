#!/usr/bin/env bash
set -Eeuo pipefail

# ---------------------------------------------------------------------------
# User configuration. Every value can also be overridden as an environment
# variable, e.g. GPUS=0 JOBS_PER_GPU=1 bash scripts/run_baseline.sh.
# ---------------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-${HOME}/share/PDEdata}"
OUT_ROOT="${OUT_ROOT:-outputs/main_results}"
CONFIG="${CONFIG:-configs/experiments/main_results.yaml}"
MATRIX_NAME="${MATRIX_NAME:-main_results}"
GPUS="${GPUS:-0,1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
# Optional comma-separated subset used for recovery runs, for example
# BASELINES=ifno MATRIX_NAME=ifno_earlystop. The matrix builder applies the
# filter without changing the experiment protocol/configuration files.
BASELINES="${BASELINES:-}"
export BASELINES

# Set VERIFY_DATA=0 only when DATA_REPORT already exists and matches CONFIG.
VERIFY_DATA="${VERIFY_DATA:-1}"
RERUN_RUNNING="${RERUN_RUNNING:-1}"
WRITE_LATEX="${WRITE_LATEX:-1}"
PLOT_SAMPLES="${PLOT_SAMPLES:-${REDRAW_SAMPLES:-1}}"
# Maximum PDFs per experiment. Set to 0 to draw every evaluated sample.
PLOT_LIMIT="${PLOT_LIMIT:-100}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON:-python}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_REPORT="${DATA_REPORT:-$OUT_ROOT/data_protocol/$MATRIX_NAME/full/data_protocol_report.json}"
MATRIX="$OUT_ROOT/matrices/$MATRIX_NAME.jsonl"
AGGREGATE_DIR="${AGGREGATE_DIR:-$OUT_ROOT/aggregate/$MATRIX_NAME}"

log() {
  printf '[run_baseline] %s\n' "$*"
}

run_cmd() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

show_status_on_exit() {
  status=$?
  trap - EXIT
  if [ "$DRY_RUN" != "1" ] && [ -f "$MATRIX" ]; then
    log "final matrix status"
    "$PYTHON_BIN" scripts/run_experiments.py "$MATRIX" --status || true
  fi
  if [ "$status" -eq 0 ]; then
    log "workflow completed"
  else
    log "workflow stopped with exit code $status; rerun this script to resume"
  fi
  exit "$status"
}
trap show_status_on_exit EXIT

[ -d "$DATA_ROOT" ] || { log "DATA_ROOT does not exist: $DATA_ROOT"; exit 2; }
[ -f "$CONFIG" ] || { log "CONFIG does not exist: $CONFIG"; exit 2; }
[[ "$JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || {
  log "JOBS_PER_GPU must be a positive integer: $JOBS_PER_GPU"
  exit 2
}
[[ "$PLOT_LIMIT" =~ ^[0-9]+$ ]] || {
  log "PLOT_LIMIT must be a non-negative integer: $PLOT_LIMIT"
  exit 2
}

log "DATA_ROOT=$DATA_ROOT"
log "OUT_ROOT=$OUT_ROOT"
log "CONFIG=$CONFIG"
log "MATRIX=$MATRIX"
log "GPUS=$GPUS JOBS_PER_GPU=$JOBS_PER_GPU"
log "BASELINES=${BASELINES:-all configured baselines}"
log "PLOT_SAMPLES=$PLOT_SAMPLES PLOT_LIMIT=$PLOT_LIMIT"

if [ "$DRY_RUN" != "1" ]; then
  mkdir -p "$OUT_ROOT"
  command -v flock >/dev/null || { log "flock is required"; exit 2; }
  exec 9>"$OUT_ROOT/.run_baseline.lock"
  flock -n 9 || { log "another run_baseline.sh is already active for $OUT_ROOT"; exit 2; }
fi

if [ "$VERIFY_DATA" = "1" ]; then
  log "step 1/5: verify data protocol"
  run_cmd "$PYTHON_BIN" scripts/verify_data_protocol.py \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --output-dir "$(dirname "$DATA_REPORT")" \
    --full
elif [ "$DRY_RUN" != "1" ] && [ ! -f "$DATA_REPORT" ]; then
  log "VERIFY_DATA=0 but DATA_REPORT does not exist: $DATA_REPORT"
  exit 2
fi

log "step 2/5: build experiment matrix"
run_cmd "$PYTHON_BIN" scripts/build_experiment_matrix.py \
  --config "$CONFIG" \
  --matrix-name "$MATRIX_NAME" \
  --output-root "$OUT_ROOT" \
  --data-manifest "$DATA_REPORT"

if [ "$DRY_RUN" != "1" ] && [ ! -f "$MATRIX" ]; then
  log "matrix was not created: $MATRIX"
  exit 2
fi

log "step 3/5: run or resume experiments"
runner_args=(
  "$PYTHON_BIN" scripts/run_experiments.py "$MATRIX"
  --data-root "$DATA_ROOT"
  --gpus "$GPUS"
  --jobs-per-gpu "$JOBS_PER_GPU"
)
if [ "$RERUN_RUNNING" = "1" ]; then
  runner_args+=(--rerun-running)
fi
run_cmd "${runner_args[@]}"

log "step 4/5: collect complete results"
collector_args=(
  "$PYTHON_BIN" scripts/collect_results.py "$MATRIX"
  --output-dir "$AGGREGATE_DIR"
)
if [ "$WRITE_LATEX" = "1" ]; then
  collector_args+=(--latex)
fi
run_cmd "${collector_args[@]}"

if [ "$PLOT_SAMPLES" = "1" ]; then
  log "step 5/5: render stored sample manifests"
  run_cmd "$PYTHON_BIN" scripts/plot_results.py \
    --matrix "$MATRIX" \
    --max-samples "$PLOT_LIMIT"
else
  log "step 5/5: sample plotting disabled"
fi
