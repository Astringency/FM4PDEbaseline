# FM4PDE Baseline

External baseline experiments for the protocol in
[`docs/baseline_exp.md`](docs/baseline_exp.md). The formal `main_results`
design contains 76 runnable experiments and 6 explicitly skipped combinations.

## Entry points

| Script | Purpose |
|---|---|
| `scripts/build_experiment_matrix.py` | Generate a matrix from an experiment YAML |
| `scripts/run_experiments.py` | Run, resume, inspect, or retry a matrix |
| `scripts/collect_results.py` | Produce CSV, JSON, XLSX, LaTeX, and PDF artifacts |

Reusable implementation modules remain under `scripts/experiments/`. The
historical `experiment_plan_v2` namespace is immutable audit evidence and is
rejected by every mutating entry point.

## Quick start

Set the data and output paths:

```bash
export DATA_ROOT=/path/to/PDEdata
export OUT_ROOT=outputs/main_results
export DATA_REPORT="$OUT_ROOT/data_protocol/full/data_protocol_report.json"
```

### 1. Verify the data

Formal matrices require a complete report bound to the exact YAML. Regenerate
the report whenever the experiment config or data changes.

```bash
python scripts/verify_data_protocol.py \
  --config configs/experiments/main_results.yaml \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT_ROOT/data_protocol/full" \
  --full
```

### 2. Build and run the matrix

```bash
python scripts/build_experiment_matrix.py \
  --output-root "$OUT_ROOT" \
  --data-manifest "$DATA_REPORT"

python scripts/run_experiments.py \
  "$OUT_ROOT/matrices/main_results.jsonl" \
  --data-root "$DATA_ROOT" \
  --gpus 0,1 \
  --jobs-per-gpu 2
```

Run the same command again to resume. Valid completed rows are skipped, and
active `run.running` rows are left untouched.

Inspect the plan or current status without launching work:

```bash
python scripts/run_experiments.py \
  "$OUT_ROOT/matrices/main_results.jsonl" \
  --data-root "$DATA_ROOT" --gpus 0,1 --dry-run

python scripts/run_experiments.py \
  "$OUT_ROOT/matrices/main_results.jsonl" --status
```

Useful selections:

```bash
# First 8 pending rows
python scripts/run_experiments.py "$OUT_ROOT/matrices/main_results.jsonl" \
  --data-root "$DATA_ROOT" --gpus 0,1 --first 8

# Zero-based rows 0, 3, and 10 through 15
python scripts/run_experiments.py "$OUT_ROOT/matrices/main_results.jsonl" \
  --data-root "$DATA_ROOT" --gpus 0,1 --indices 0,3,10-15

# Failed rows only
python scripts/run_experiments.py "$OUT_ROOT/matrices/main_results.jsonl" \
  --data-root "$DATA_ROOT" --gpus 0,1 --failed-only
```

Use `--rerun-running` only after confirming that all `run.running` markers are
stale. Existing invalid artifacts are moved into the run's `quarantine/`
directory before relaunch.

### 3. Collect the results

```bash
python scripts/collect_results.py \
  "$OUT_ROOT/matrices/main_results.jsonl" \
  --output-dir "$OUT_ROOT/aggregate/main_results" \
  --latex \
  --redraw-samples
```

Collection fails before writing publication tables if any run is missing,
invalid, from another provenance cohort, or lacks a requested sample manifest.

## Ablations

Available designs:

- `sensor_count_ablation`
- `noise_ablation`
- `sensor_mode_ablation`
- `time_varying_sensor_ablation`
- `runtime_budget_ablation`
- `train_size_ablation`

Use the same workflow with another config:

```bash
NAME=noise_ablation
REPORT="$OUT_ROOT/data_protocol/$NAME/full/data_protocol_report.json"

python scripts/verify_data_protocol.py \
  --config "configs/experiments/$NAME.yaml" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT_ROOT/data_protocol/$NAME/full" --full

python scripts/build_experiment_matrix.py \
  --config "configs/experiments/$NAME.yaml" \
  --matrix-name "$NAME" --output-root "$OUT_ROOT" \
  --data-manifest "$REPORT"

python scripts/run_experiments.py "$OUT_ROOT/matrices/$NAME.jsonl" \
  --data-root "$DATA_ROOT" --gpus 0,1 --jobs-per-gpu 2

python scripts/collect_results.py "$OUT_ROOT/matrices/$NAME.jsonl" \
  --output-dir "$OUT_ROOT/aggregate/$NAME"
```

## Outputs

Each run stores its checkpoint, status markers, raw metrics, `summary.json`,
per-sample `.pt` artifacts, a checksum manifest, and a multipage PDF. Exact
artifact paths are recorded in `summary.json`.

Load the first saved evaluation sample:

```python
import json
from pathlib import Path

from baselines.common.sample_artifacts import load_evaluation_sample

summary = json.loads(Path("outputs/.../summary.json").read_text())
manifest = Path(summary["sample_manifest_path"])
record = json.loads(manifest.read_text().splitlines()[0])
sample = load_evaluation_sample(record["artifact_path"])
```

## Tests

```bash
python -m py_compile \
  scripts/build_experiment_matrix.py scripts/run_experiments.py \
  scripts/collect_results.py scripts/experiments/*.py
python -m pytest -q
```

Do not edit `offical/` unless intentionally updating the vendored upstream
source snapshots.
