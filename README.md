# FM4PDE Baseline

External baseline experiments for the protocol in
[`docs/baseline_exp.md`](docs/baseline_exp.md). The formal `main_results`
design is the single source of truth and contains 76 runnable experiments.
`baselines/configs/paper.yaml` contains the shared runtime and canonical
per-method recipes; ablation-only methods are labeled in place.

## Entry points

| Script | Purpose |
|---|---|
| `scripts/build_experiment_matrix.py` | Generate a matrix from an experiment YAML |
| `scripts/run_experiments.py` | Run, resume, inspect, or retry a matrix |
| `scripts/collect_results.py` | Produce CSV, JSON, XLSX, and LaTeX artifacts |
| `scripts/plot_results.py` | Render one PDF per stored evaluation sample |
| `scripts/run_baseline.sh` | Run the complete resumable workflow |

Reusable implementation modules remain under `scripts/experiments/`. There is
no runnable v2 design; its historical output namespace remains protected from
accidental overwrite by mutating entry points.

## Quick start

For the simplest setup, edit the configuration block at the top of
`scripts/run_baseline.sh`, then run:

```bash
bash scripts/run_baseline.sh
```

The script verifies data, builds the matrix, resumes unfinished experiments,
collects results, renders PDFs in a separate process, and prints status when it
finishes or is interrupted. It uses an output lock and will not restart a
`run.running` row whose process is alive.

The same values can be overridden without editing the file:

```bash
DATA_ROOT=/path/to/PDEdata OUT_ROOT=outputs/main_results \
GPUS=0,1 JOBS_PER_GPU=2 PLOT_LIMIT=100 bash scripts/run_baseline.sh
```

`PLOT_LIMIT` is applied per experiment. It defaults to `100`; set it to `0`
to render every evaluated sample, or set `PLOT_SAMPLES=0` to skip plotting.

The commands below show the equivalent manual workflow.

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

Use `--rerun-running` to recover stale markers. The runner checks the recorded
PID and process start time, leaves live work untouched, and moves invalid stale
artifacts into the run's `quarantine/` directory before relaunch.

### 3. Collect the results

```bash
python scripts/collect_results.py \
  "$OUT_ROOT/matrices/main_results.jsonl" \
  --output-dir "$OUT_ROOT/aggregate/main_results" \
  --latex

python scripts/plot_results.py \
  --matrix "$OUT_ROOT/matrices/main_results.jsonl" \
  --max-samples 100
```

Collection fails before writing publication tables if any run is missing,
invalid, or from another provenance cohort. Plotting reads only stored sample
manifests, writes one PDF per sample, skips complete PDFs, and can be rerun
without retraining.

## Ablations

These are ablation designs, not alternate definitions of the formal
experiment. They inherit the formal one-seed, 50,000-total-training-sample and
1,000-test-sample defaults, then vary only their named factor.

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

python scripts/plot_results.py --matrix "$OUT_ROOT/matrices/$NAME.jsonl"
```

## Outputs

Each run stores its checkpoint, status markers, raw metrics, `summary.json`,
per-sample `.pt` artifacts, and a checksum manifest. PDF rendering is a separate
step, so plotting failures never invalidate a completed experiment.

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
