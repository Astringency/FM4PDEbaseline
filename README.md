# FM4PDE Baseline

This repository implements the external baseline experiments defined in
`docs/baseline_exp.md`. The formal `main_results` design currently expands to
76 runnable experiments and 6 explicitly skipped unsupported combinations.

## Layout

```text
baselines/                         training, evaluation, metrics, artifacts
configs/experiments/               declarative main/ablation designs
scripts/build_experiment_matrix.py public matrix generator
scripts/run_experiments.py         public local runner, resume, status, retry
scripts/collect_results.py         public CSV/JSON/XLSX/PDF result collector
scripts/experiments/               internal matrix/provenance/single-row modules
scripts/verify_data_protocol.py    data audit required before formal runs
tests/                             pytest coverage
outputs/                           generated outputs and audit fixtures
```

The numbered shell launchers were removed. Use only the three public Python
entry points above for experiments. `offical/` contains vendored source
snapshots and must not be edited unless those snapshots are intentionally
updated.

## Run the formal 76-experiment matrix

Set paths once:

```bash
export DATA_ROOT=/path/to/PDEdata
export OUT_ROOT=outputs/main_results
export DATA_REPORT="$OUT_ROOT/data_protocol/full/data_protocol_report.json"
```

1. Verify the exact data/config contract. A changed YAML invalidates the old
report, so rerun this command after editing the experiment design.

```bash
python scripts/verify_data_protocol.py \
  --config configs/experiments/main_results.yaml \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT_ROOT/data_protocol/full" \
  --full
```

2. Generate the matrix. `main_results.yaml` and matrix name `main_results` are
the defaults.

```bash
python scripts/build_experiment_matrix.py \
  --output-root "$OUT_ROOT" \
  --data-manifest "$DATA_REPORT"
```

The primary file is `$OUT_ROOT/matrices/main_results.jsonl`; TSV, summary JSON,
and skipped-combination JSONL files are created beside it.

3. Inspect the plan and status without launching work:

```bash
python scripts/run_experiments.py \
  "$OUT_ROOT/matrices/main_results.jsonl" \
  --data-root "$DATA_ROOT" \
  --gpus 0,1 --jobs-per-gpu 2 --dry-run

python scripts/run_experiments.py \
  "$OUT_ROOT/matrices/main_results.jsonl" --status
```

4. Run on two GPUs. Completed rows are skipped automatically; rerunning the
same command resumes the matrix. Invalid partial artifacts are moved into each
run's `quarantine/` directory before retry. Rows carrying `run.running` are
left untouched so a second launcher cannot duplicate active work; use
`--rerun-running` only after confirming the marker is stale.

```bash
python scripts/run_experiments.py \
  "$OUT_ROOT/matrices/main_results.jsonl" \
  --data-root "$DATA_ROOT" \
  --gpus 0,1 --jobs-per-gpu 2
```

Useful subsets:

```bash
# First 8 pending rows
python scripts/run_experiments.py "$OUT_ROOT/matrices/main_results.jsonl" \
  --data-root "$DATA_ROOT" --gpus 0,1 --first 8

# Exact zero-based rows
python scripts/run_experiments.py "$OUT_ROOT/matrices/main_results.jsonl" \
  --data-root "$DATA_ROOT" --gpus 0,1 --indices 0,3,10-15

# Retry only failed rows
python scripts/run_experiments.py "$OUT_ROOT/matrices/main_results.jsonl" \
  --data-root "$DATA_ROOT" --gpus 0,1 --failed-only

```

5. Aggregate completed results. This writes publication/supplement CSV and
JSON tables, capability/tuning tables, `results.xlsx`, and a collection report.
Evaluation already saves every sample as a reloadable `.pt` dictionary and
creates `samples.pdf`; `--redraw-samples` regenerates those PDFs from manifests.

```bash
python scripts/collect_results.py \
  "$OUT_ROOT/matrices/main_results.jsonl" \
  --output-dir "$OUT_ROOT/aggregate/main_results" \
  --latex --redraw-samples
```

## Run an ablation

Use the same three commands with another YAML. Supported designs are
`sensor_count_ablation`, `noise_ablation`, `sensor_mode_ablation`,
`time_varying_sensor_ablation`, `runtime_budget_ablation`, and
`train_size_ablation`.

```bash
NAME=noise_ablation
REPORT="$OUT_ROOT/data_protocol/$NAME/full/data_protocol_report.json"

python scripts/verify_data_protocol.py \
  --config "configs/experiments/$NAME.yaml" --data-root "$DATA_ROOT" \
  --output-dir "$OUT_ROOT/data_protocol/$NAME/full" --full

python scripts/build_experiment_matrix.py \
  --config "configs/experiments/$NAME.yaml" --matrix-name "$NAME" \
  --output-root "$OUT_ROOT" --data-manifest "$REPORT"

python scripts/run_experiments.py "$OUT_ROOT/matrices/$NAME.jsonl" \
  --data-root "$DATA_ROOT" --gpus 0,1 --jobs-per-gpu 2

python scripts/collect_results.py "$OUT_ROOT/matrices/$NAME.jsonl" \
  --output-dir "$OUT_ROOT/aggregate/$NAME"
```

## Evaluation artifacts

Each run stores a model checkpoint, `summary.json`, raw/summary result tables,
and one tensor dictionary per test sample. Formal runs use a run-prefixed sample
directory; its exact path is recorded in `summary.json`. The sample artifact
contains inputs, targets, predictions, observations, masks, per-sample metrics,
predictive standard deviations when available, and PC-BNN posterior particles
when available.

```python
import json
from pathlib import Path

from baselines.common.sample_artifacts import load_evaluation_sample

summary = json.loads(Path("outputs/.../summary.json").read_text())
manifest = Path(summary["sample_manifest_path"])
first_record = json.loads(manifest.read_text().splitlines()[0])
sample = load_evaluation_sample(first_record["artifact_path"])
```

## Tests

```bash
python -m py_compile \
  scripts/build_experiment_matrix.py scripts/run_experiments.py \
  scripts/collect_results.py scripts/experiments/*.py \
  baselines/run.py baselines/aggregate_results.py
python -m pytest -q
```

The historical `experiment_plan_v2` matrix and outputs are immutable audit
evidence. All mutating entry points reject that namespace.
