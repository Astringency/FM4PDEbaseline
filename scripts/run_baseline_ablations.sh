#!/usr/bin/env bash
set -Eeuo pipefail

# One-command workflow for sparse_solution_multicondition.
# Override values as environment variables, for example:
#   DATA_ROOT=/data/PDEdata GPUS=0,1 JOBS_PER_GPU=1 bash scripts/run_baseline_ablations.sh
# Run a PDE subset with, for example:
#   PDE_LIST=poisson bash scripts/run_baseline_ablations.sh
DATA_ROOT="${DATA_ROOT:-${HOME}/share/PDEdata}"
CONFIG="${CONFIG:-configs/experiments/sparse_solution_multicondition_ablation.yaml}"
MATRIX_NAME="${MATRIX_NAME:-sparse_solution_multicondition_ablation}"
GPUS="${GPUS:-0,1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
VERIFY_DATA="${VERIFY_DATA:-1}"
RERUN_RUNNING="${RERUN_RUNNING:-1}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON:-python}"

PDE_SELECTOR_ACTIVE=0
PDE_LIST_RAW="${PDE_LIST:-}"
SELECTED_PDES=(poisson helmholtz darcy nsnonbounded)
if [ -n "${PDE_LIST_RAW//[[:space:],]/}" ]; then
  PDE_SELECTOR_ACTIVE=1
  PDE_LIST_NORMALIZED="${PDE_LIST_RAW//,/ }"
  read -r -a REQUESTED_PDES <<< "$PDE_LIST_NORMALIZED"
  SELECTED_PDES=()
  for pde in "${REQUESTED_PDES[@]}"; do
    case "$pde" in
      poisson|helmholtz|darcy|nsnonbounded) ;;
      burger)
        printf '[run_baseline_ablations] sparse_solution_multicondition does not support Burgers trajectory semantics\n' >&2
        exit 2
        ;;
      *)
        printf '[run_baseline_ablations] unsupported PDE in PDE_LIST: %s\n' "$pde" >&2
        exit 2
        ;;
    esac
    for selected in "${SELECTED_PDES[@]}"; do
      if [ "$selected" = "$pde" ]; then
        printf '[run_baseline_ablations] duplicate PDE in PDE_LIST: %s\n' "$pde" >&2
        exit 2
      fi
    done
    SELECTED_PDES+=("$pde")
  done
fi

PDE_LIST=""
PDE_TAG=""
for pde in "${SELECTED_PDES[@]}"; do
  if [ -n "$PDE_LIST" ]; then
    PDE_LIST+=","
    PDE_TAG+="_"
  fi
  PDE_LIST+="$pde"
  PDE_TAG+="$pde"
done
export PDE_LIST

if [ -n "${OUT_ROOT:-}" ]; then
  OUT_ROOT="$OUT_ROOT"
elif [ "$PDE_SELECTOR_ACTIVE" = "1" ]; then
  OUT_ROOT="results/ablations/sparse_solution_multicondition/$PDE_TAG"
else
  OUT_ROOT="results/ablations/sparse_solution_multicondition"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_REPORT="${DATA_REPORT:-$OUT_ROOT/data_protocol/$MATRIX_NAME/full/data_protocol_report.json}"
MATRIX="$OUT_ROOT/matrices/$MATRIX_NAME.jsonl"
REPORT_DIR="${REPORT_DIR:-$OUT_ROOT/report}"
PDE_COUNT="${#SELECTED_PDES[@]}"
BASELINE_COUNT=3
CONDITION_COUNT=3
TRAIN_COUNT=$((PDE_COUNT * BASELINE_COUNT))
EVAL_COUNT=$((TRAIN_COUNT * CONDITION_COUNT))
TOTAL_COUNT=$((TRAIN_COUNT + EVAL_COUNT))
TRAIN_INDICES="0-$((TRAIN_COUNT - 1))"
EVAL_INDICES="$TRAIN_COUNT-$((TOTAL_COUNT - 1))"
VERIFY_PDE_ARGS=()
if [ "$PDE_SELECTOR_ACTIVE" = "1" ]; then
  for pde in "${SELECTED_PDES[@]}"; do
    VERIFY_PDE_ARGS+=(--pde "$pde")
  done
fi

log() {
  printf '[run_baseline_ablations] %s\n' "$*"
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
    log "workflow completed; report=$REPORT_DIR"
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

log "DATA_ROOT=$DATA_ROOT"
log "OUT_ROOT=$OUT_ROOT"
log "CONFIG=$CONFIG"
log "MATRIX=$MATRIX"
log "PDE_LIST=$PDE_LIST"
log "GPUS=$GPUS JOBS_PER_GPU=$JOBS_PER_GPU"
log "workflow=$TRAIN_COUNT training rows -> checkpoint binding -> $EVAL_COUNT eval-only rows -> report"

if [ "$DRY_RUN" != "1" ]; then
  mkdir -p "$OUT_ROOT"
  command -v flock >/dev/null || { log "flock is required"; exit 2; }
  exec 9>"$OUT_ROOT/.run_baseline_ablations.lock"
  flock -n 9 || {
    log "another run_baseline_ablations.sh is already active for $OUT_ROOT"
    exit 2
  }
fi

if [ "$VERIFY_DATA" = "1" ]; then
  log "step 1/6: verify the complete data protocol"
  run_cmd "$PYTHON_BIN" scripts/verify_data_protocol.py \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --output-dir "$(dirname "$DATA_REPORT")" \
    "${VERIFY_PDE_ARGS[@]}" \
    --full
elif [ "$DRY_RUN" != "1" ] && [ ! -f "$DATA_REPORT" ]; then
  log "VERIFY_DATA=0 but DATA_REPORT does not exist: $DATA_REPORT"
  exit 2
fi

build_matrix() {
  run_cmd "$PYTHON_BIN" scripts/build_experiment_matrix.py \
    --config "$CONFIG" \
    --matrix-name "$MATRIX_NAME" \
    --output-root "$OUT_ROOT" \
    --data-manifest "$DATA_REPORT"
}

log "step 2/6: build the $TRAIN_COUNT-train + $EVAL_COUNT-evaluation matrix"
build_matrix
if [ "$DRY_RUN" != "1" ] && [ ! -f "$MATRIX" ]; then
  log "matrix was not created: $MATRIX"
  exit 2
fi

runner_args=(
  "$PYTHON_BIN" scripts/run_experiments.py "$MATRIX"
  --data-root "$DATA_ROOT"
  --gpus "$GPUS"
  --jobs-per-gpu "$JOBS_PER_GPU"
)
if [ "$RERUN_RUNNING" = "1" ]; then
  runner_args+=(--rerun-running)
fi

runner_failed=0
log "step 3/6: train or resume exactly $TRAIN_COUNT models"
run_cmd "${runner_args[@]}" --indices "$TRAIN_INDICES" || runner_failed=1

log "step 4/6: rebuild the matrix and bind checkpoint paths/SHA-256 values"
build_matrix

log "step 5/6: evaluate a_only, u_only, and both without retraining"
run_cmd "${runner_args[@]}" --indices "$EVAL_INDICES" || runner_failed=1

if [ "$runner_failed" = "1" ]; then
  log "one or more rows failed; completed checkpoints and evaluations were preserved"
  exit 1
fi

log "step 6/6: validate shared provenance and write CSV/Markdown/JSON reports"
run_cmd "$PYTHON_BIN" scripts/build_sparse_solution_multicondition_report.py \
  --input-root "$OUT_ROOT" \
  --output-dir "$REPORT_DIR"
