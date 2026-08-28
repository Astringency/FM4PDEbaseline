#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate existing checkpoints on a replacement test file without retraining.
#
# Defaults target the RecFNO/Senseiver/VoronoiCNN Poisson experiments in the
# main_results matrix. All paths and selections can be overridden with CLI
# flags or their corresponding environment variables.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${HOME}/share/PDEdata}"
CONFIG="${CONFIG:-baselines/configs/paper.yaml}"
PDE="${PDE:-poisson}"
TEST_FILE="${TEST_FILE:-poisson/poisson_test_10000-128-128-2.mat}"
TEST_SIZE="${TEST_SIZE:-10000}"
BASELINES="${BASELINES:-recfno,senseiver,voronoicnn}"
TASKS="${TASKS:-sparse_forward,sparse_inverse,sparse_solution}"
DEVICE="${DEVICE:-cuda}"
SAVE_SAMPLES="${SAVE_SAMPLES:-1}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_TAG="${EVAL_TAG:-}"

log() {
  printf '[run_eval] %s\n' "$*"
}

die() {
  log "ERROR: $*"
  exit 2
}

usage() {
  cat <<'EOF'
Usage: bash scripts/run_eval.sh [options]

Evaluate existing trained checkpoints on a new test dataset. Training is
skipped by passing --eval-only to baselines.run, and outputs are written below
a separate evaluation root.

Options:
  --test-file PATH       Test file relative to DATA_ROOT, or an absolute path
                         located below DATA_ROOT.
  --test-size N          Number of test samples to evaluate (default: 10000).
  --pde NAME             PDE name (default: poisson).
  --baselines CSV        Baselines to evaluate
                         (default: recfno,senseiver,voronoicnn).
  --tasks CSV            Tasks to evaluate
                         (default: sparse_forward,sparse_inverse,sparse_solution).
  --data-root DIR        Dataset root (default: ~/share/PDEdata).
  --train-root DIR       Existing main_results run root.
  --eval-root DIR        Destination root for the new evaluation.
  --eval-tag NAME        Destination tag when --eval-root is not supplied.
  --config FILE          Baseline method config (default: baselines/configs/paper.yaml).
  --device DEVICE        Torch device (default: cuda).
  --python PATH          Python executable (default: $PYTHON or python).
  --save-samples         Save one reloadable .pt artifact per test sample (default).
  --no-save-samples      Save metrics only, without per-sample .pt artifacts.
  --dry-run              Print commands without launching evaluation.
  -h, --help             Show this help.

Environment equivalents:
  DATA_ROOT, TRAIN_ROOT, EVAL_ROOT, FM_OUTPUT_ROOT, TEST_FILE, TEST_SIZE,
  PDE, BASELINES, TASKS, CONFIG, DEVICE, PYTHON, SAVE_SAMPLES, DRY_RUN,
  EVAL_TAG.

Examples:
  bash scripts/run_eval.sh --dry-run
  bash scripts/run_eval.sh --no-save-samples
  bash scripts/run_eval.sh --baselines recfno --tasks sparse_inverse
  CUDA_VISIBLE_DEVICES=1 bash scripts/run_eval.sh --device cuda
EOF
}

TRAIN_ROOT="${TRAIN_ROOT:-}"
EVAL_ROOT="${EVAL_ROOT:-}"
FM_OUTPUT_ROOT="${FM_OUTPUT_ROOT:-}"

while (($#)); do
  case "$1" in
    --test-file)
      (($# >= 2)) || die "--test-file requires a value"
      TEST_FILE="$2"
      shift 2
      ;;
    --test-size)
      (($# >= 2)) || die "--test-size requires a value"
      TEST_SIZE="$2"
      shift 2
      ;;
    --pde)
      (($# >= 2)) || die "--pde requires a value"
      PDE="$2"
      shift 2
      ;;
    --baselines)
      (($# >= 2)) || die "--baselines requires a value"
      BASELINES="$2"
      shift 2
      ;;
    --tasks)
      (($# >= 2)) || die "--tasks requires a value"
      TASKS="$2"
      shift 2
      ;;
    --data-root)
      (($# >= 2)) || die "--data-root requires a value"
      DATA_ROOT="$2"
      shift 2
      ;;
    --train-root)
      (($# >= 2)) || die "--train-root requires a value"
      TRAIN_ROOT="$2"
      shift 2
      ;;
    --eval-root)
      (($# >= 2)) || die "--eval-root requires a value"
      EVAL_ROOT="$2"
      shift 2
      ;;
    --eval-tag)
      (($# >= 2)) || die "--eval-tag requires a value"
      EVAL_TAG="$2"
      shift 2
      ;;
    --config)
      (($# >= 2)) || die "--config requires a value"
      CONFIG="$2"
      shift 2
      ;;
    --device)
      (($# >= 2)) || die "--device requires a value"
      DEVICE="$2"
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --save-samples)
      SAVE_SAMPLES=1
      shift
      ;;
    --no-save-samples)
      SAVE_SAMPLES=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ "$TEST_SIZE" =~ ^[1-9][0-9]*$ ]] || die "TEST_SIZE must be a positive integer: $TEST_SIZE"
[[ "$SAVE_SAMPLES" =~ ^[01]$ ]] || die "SAVE_SAMPLES must be 0 or 1: $SAVE_SAMPLES"
[[ "$DRY_RUN" =~ ^[01]$ ]] || die "DRY_RUN must be 0 or 1: $DRY_RUN"
[ -d "$DATA_ROOT" ] || die "DATA_ROOT does not exist: $DATA_ROOT"
[ -f "$CONFIG" ] || die "CONFIG does not exist: $CONFIG"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"

DATA_ROOT="$(realpath "$DATA_ROOT")"
if [[ "$TEST_FILE" = /* ]]; then
  TEST_PATH="$(realpath "$TEST_FILE")"
  case "$TEST_PATH" in
    "$DATA_ROOT"/*) TEST_FILE="${TEST_PATH#"$DATA_ROOT"/}" ;;
    *) die "absolute --test-file must be located below DATA_ROOT: $TEST_PATH" ;;
  esac
else
  TEST_PATH="$(realpath "$DATA_ROOT/$TEST_FILE")"
fi
[ -f "$TEST_PATH" ] || die "test file does not exist: $TEST_PATH"

detect_output_root() {
  local candidate
  for candidate in \
    "${HOME}/share/outputs/FM4PDEbaseline" \
    "${HOME}/share/zhangxfA100/large_storage/outputs/FM4PDEbaseline"; do
    if [ -d "$candidate" ]; then
      realpath "$candidate"
      return 0
    fi
  done
  printf '%s\n' "${HOME}/share/outputs/FM4PDEbaseline"
}

if [ -z "$FM_OUTPUT_ROOT" ]; then
  FM_OUTPUT_ROOT="$(detect_output_root)"
fi
TRAIN_ROOT="${TRAIN_ROOT:-$FM_OUTPUT_ROOT/runs/main_results}"
[ -d "$TRAIN_ROOT" ] || die "training run root does not exist: $TRAIN_ROOT"
TRAIN_ROOT="$(realpath "$TRAIN_ROOT")"

if [ -z "$EVAL_TAG" ]; then
  EVAL_TAG="$(basename "$TEST_FILE")"
  EVAL_TAG="${EVAL_TAG%.mat}"
fi
EVAL_TAG="${EVAL_TAG//[^a-zA-Z0-9_.-]/_}"
[ -n "$EVAL_TAG" ] || die "EVAL_TAG resolves to an empty value"
EVAL_ROOT="${EVAL_ROOT:-$FM_OUTPUT_ROOT/runs/evaluations/$EVAL_TAG}"

IFS=',' read -r -a BASELINE_LIST <<< "$BASELINES"
IFS=',' read -r -a TASK_LIST <<< "$TASKS"
((${#BASELINE_LIST[@]} > 0)) || die "no baselines selected"
((${#TASK_LIST[@]} > 0)) || die "no tasks selected"

task_group_for() {
  case "$1" in
    sparse_forward) printf '%s\n' sparse_forward_main_amortized ;;
    sparse_inverse) printf '%s\n' sparse_inverse_main_amortized ;;
    sparse_solution) printf '%s\n' sparse_solution_main_amortized ;;
    *) return 1 ;;
  esac
}

for task in "${TASK_LIST[@]}"; do
  task_group_for "$task" >/dev/null || die "unsupported task selection: $task"
done

run_cmd() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

# Print checkpoint and adjacent summary metadata as one value per line. The
# checkpoint is authoritative for source-training identity; summary.json is
# used for the original data-file list and runtime protocol fields.
checkpoint_metadata() {
  "$PYTHON_BIN" - "$1" "$TEST_FILE" <<'PY'
import json
import sys
from pathlib import Path

import torch

checkpoint = Path(sys.argv[1])
replacement_test = sys.argv[2]
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
provenance = payload.get("provenance")
if not isinstance(provenance, dict):
    raise SystemExit(f"checkpoint has no provenance dictionary: {checkpoint}")

run_id = str(provenance.get("run_id") or checkpoint.stem)
summary_path = checkpoint.parent / "summary.json"
if not summary_path.is_file():
    candidate = checkpoint.parent / f"{run_id}_summary.json"
    summary_path = candidate if candidate.is_file() else summary_path
if not summary_path.is_file():
    raise SystemExit(f"summary.json not found beside checkpoint: {checkpoint}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))

try:
    data_files = json.loads(summary["data_files_json"])
except (KeyError, TypeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"summary has no valid data_files_json: {summary_path}") from exc
if not isinstance(data_files, dict) or not data_files.get("train"):
    raise SystemExit(f"summary data_files_json has no train files: {summary_path}")
data_files["test"] = [replacement_test]

def pick(name, default=""):
    value = summary.get(name, provenance.get(name, default))
    return default if value is None else value

values = [
    run_id,
    provenance.get("run_fingerprint", ""),
    payload.get("baseline", summary.get("baseline", "")),
    provenance.get("pde", summary.get("pde", "")),
    provenance.get("task", summary.get("task", "")),
    provenance.get("seed", summary.get("seed", 1)),
    provenance.get("sensor_seed", summary.get("sensor_seed", provenance.get("seed", 1))),
    provenance.get("requested_train_size", summary.get("train_requested_size", 50000)),
    provenance.get("val_size", summary.get("val_requested_size", 5000)),
    provenance.get("train_shards", summary.get("train_shards", 5)),
    provenance.get("batch_size", summary.get("batch_size", 16)),
    provenance.get("epochs", summary.get("epochs", 200)),
    provenance.get("sensor_mode", summary.get("sensor_mode", "random_per_sample")),
    pick("sensor_budget_mode", "per_time"),
    provenance.get("num_sensors", summary.get("num_sensors", 500)),
    pick("noise_level", 0.0),
    provenance.get("comparison_track", summary.get("comparison_track", "unified_adapted")),
    provenance.get("task_protocol_version", summary.get("task_protocol_version", "fm4pde-task-contract-v3")),
    provenance.get("sensor_protocol_version", summary.get("sensor_protocol_version", "fm4pde-sensor-contract-v3")),
    pick("scalar_param_mode", "metadata"),
    pick("physics_metric_mode", "per_sample"),
    "1" if bool(pick("load_full_trajectory", False)) else "0",
    json.dumps(data_files, sort_keys=True, separators=(",", ":")),
    pick("num_workers", summary.get("dataloader_num_workers", 4)),
    "1" if bool(pick("pin_memory_requested", summary.get("dataloader_pin_memory", True))) else "0",
    "1" if bool(pick("persistent_workers_requested", summary.get("dataloader_persistent_workers", True))) else "0",
    pick("prefetch_factor_requested", 2),
    pick("data_loading_mode_requested", summary.get("data_loading_mode", "eager")),
]
for value in values:
    print(value)
PY
}

log "DATA_ROOT=$DATA_ROOT"
log "TEST_FILE=$TEST_FILE TEST_SIZE=$TEST_SIZE"
log "TRAIN_ROOT=$TRAIN_ROOT"
log "EVAL_ROOT=$EVAL_ROOT"
log "PDE=$PDE BASELINES=$BASELINES TASKS=$TASKS DEVICE=$DEVICE"
log "SAVE_SAMPLES=$SAVE_SAMPLES DRY_RUN=$DRY_RUN"

if [ "$DRY_RUN" != "1" ]; then
  mkdir -p "$EVAL_ROOT"
  command -v flock >/dev/null 2>&1 || die "flock is required"
  exec 9>"$EVAL_ROOT/.run_eval.lock"
  flock -n 9 || die "another run_eval.sh is already active for $EVAL_ROOT"
fi

found_count=0
launched_count=0
completed_skip_count=0
missing_count=0

for baseline in "${BASELINE_LIST[@]}"; do
  [ -n "$baseline" ] || die "BASELINES contains an empty item"
  for selected_task in "${TASK_LIST[@]}"; do
    task_group="$(task_group_for "$selected_task")"
    checkpoint_root="$TRAIN_ROOT/task_group=$task_group/pde=$PDE/baseline=$baseline"
    checkpoints=()
    if [ -d "$checkpoint_root" ]; then
      mapfile -t checkpoints < <(
        find "$checkpoint_root" -mindepth 3 -maxdepth 3 -type f -name '*.pt' -print | sort
      )
    fi
    if ((${#checkpoints[@]} == 0)); then
      log "PENDING/MISSING: baseline=$baseline task=$selected_task (no checkpoint)"
      ((missing_count += 1))
      continue
    fi

    for checkpoint in "${checkpoints[@]}"; do
      ((found_count += 1))
      metadata="$(checkpoint_metadata "$checkpoint")" || die "cannot read checkpoint metadata: $checkpoint"
      mapfile -t meta <<< "$metadata"
      ((${#meta[@]} == 28)) || die "unexpected checkpoint metadata field count for $checkpoint: ${#meta[@]}"

      source_run_id="${meta[0]}"
      source_fingerprint="${meta[1]}"
      checkpoint_baseline="${meta[2]}"
      checkpoint_pde="${meta[3]}"
      checkpoint_task="${meta[4]}"
      seed="${meta[5]}"
      sensor_seed="${meta[6]}"
      train_size="${meta[7]}"
      val_size="${meta[8]}"
      train_shards="${meta[9]}"
      batch_size="${meta[10]}"
      epochs="${meta[11]}"
      sensor_mode="${meta[12]}"
      sensor_budget_mode="${meta[13]}"
      num_sensors="${meta[14]}"
      noise_level="${meta[15]}"
      comparison_track="${meta[16]}"
      task_protocol_version="${meta[17]}"
      sensor_protocol_version="${meta[18]}"
      scalar_param_mode="${meta[19]}"
      physics_metric_mode="${meta[20]}"
      load_full_trajectory="${meta[21]}"
      data_files_json="${meta[22]}"
      num_workers="${meta[23]}"
      pin_memory="${meta[24]}"
      persistent_workers="${meta[25]}"
      prefetch_factor="${meta[26]}"
      data_loading_mode="${meta[27]}"

      [ "$checkpoint_baseline" = "$baseline" ] || die "checkpoint baseline mismatch: $checkpoint"
      [ "$checkpoint_pde" = "$PDE" ] || die "checkpoint PDE mismatch: $checkpoint"
      [ "$checkpoint_task" = "$selected_task" ] || die "checkpoint task mismatch: $checkpoint"
      [ -n "$source_fingerprint" ] || die "checkpoint has no source run fingerprint: $checkpoint"

      eval_run="eval_${EVAL_TAG}_${source_run_id}"
      output_dir="$EVAL_ROOT/task_group=$task_group/pde=$PDE/baseline=$baseline/seed=$seed/run=$eval_run"
      if [ -f "$output_dir/summary.json" ]; then
        log "SKIP completed: baseline=$baseline task=$selected_task seed=$seed output=$output_dir"
        ((completed_skip_count += 1))
        continue
      fi

      command=(
        "$PYTHON_BIN" -m baselines.run
        --baseline "$baseline"
        --pde "$PDE"
        --task "$selected_task"
        --task-group "$task_group"
        --config "$CONFIG"
        --experiment-mode debug
        --execution-mode eval_only
        --comparison-track "$comparison_track"
        --task-protocol-version "$task_protocol_version"
        --sensor-protocol-version "$sensor_protocol_version"
        --data-root "$DATA_ROOT"
        --data-files-json "$data_files_json"
        --train-size "$train_size"
        --val-size "$val_size"
        --test-size "$TEST_SIZE"
        --train-shards "$train_shards"
        --batch-size "$batch_size"
        --epochs "$epochs"
        --seed "$seed"
        --sensor-seed "$sensor_seed"
        --device "$DEVICE"
        --data-loading-mode "$data_loading_mode"
        --num-workers "$num_workers"
        --prefetch-factor "$prefetch_factor"
        --scalar-param-mode "$scalar_param_mode"
        --physics-metric-mode "$physics_metric_mode"
        --strict-size
        --eval-only
        --checkpoint "$checkpoint"
        --source-train-run-id "$source_run_id"
        --source-train-run-fingerprint "$source_fingerprint"
        --source-train-seed "$seed"
        --output-dir "$output_dir"
        --run-id "$eval_run"
        --run-name "$EVAL_TAG/$baseline/$PDE/$selected_task/seed=$seed"
        --experiment-kind evaluation
        --ablation-factor replacement_test
        --no-save-checkpoint
      )

      if [[ "$selected_task" == sparse_* ]]; then
        command+=(
          --num-sensors "$num_sensors"
          --sensor-mode "$sensor_mode"
          --sensor-budget-mode "$sensor_budget_mode"
          --noise-level "$noise_level"
        )
      fi
      if [ "$load_full_trajectory" = "1" ]; then
        command+=(--load-full-trajectory)
      fi
      if [ "$pin_memory" = "1" ]; then
        command+=(--pin-memory)
      else
        command+=(--no-pin-memory)
      fi
      if [ "$persistent_workers" = "1" ]; then
        command+=(--persistent-workers)
      else
        command+=(--no-persistent-workers)
      fi
      if [ "$SAVE_SAMPLES" = "1" ]; then
        command+=(--save-sample-artifacts)
      else
        command+=(--no-save-sample-artifacts)
      fi

      log "RUN baseline=$baseline task=$selected_task seed=$seed checkpoint=$checkpoint"
      run_cmd "${command[@]}"
      ((launched_count += 1))
    done
  done
done

log "finished: checkpoints_found=$found_count launched=$launched_count already_complete=$completed_skip_count missing=$missing_count"
if ((found_count == 0)); then
  die "no matching checkpoints were found"
fi
