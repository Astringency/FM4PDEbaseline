# FM4PDE Baseline

This repository runs and audits external baseline experiments for FM4PDE. The
active entry points are the matrix-based scripts under `scripts/experiments/`;
older per-task shell interfaces have been removed.

## Scope

The framework covers:

- full-grid supervised operator learning
- sparse sensor field reconstruction
- static sparse inverse problems
- time-varying data assimilation

FM4PDE internal ablations and DiffusionPDE comparisons are outside this external
baseline matrix.

## Layout

```text
baselines/                  baseline framework code
  capabilities.py           baseline/task/PDE capability registry
  experiment_matrix.py      capability checks and skip rows
  run.py                    single-run baseline runner
  aggregate_results.py      result aggregation
  common/                   data, sensors, physics residuals, metrics
  methods/                  baseline wrappers
  configs/                  runner method configs

configs/experiments/        matrix configs for main and ablation experiments
scripts/experiments/        matrix build, run, status, retry, aggregation scripts
tests/                      pytest coverage
offical/                    vendored official source snapshots
outputs/                    generated outputs and checked-in fixtures
```

Do not edit `offical/` unless intentionally updating the vendored snapshots.

## Active Experiment Flow

Set the data root and output root first:

```bash
export DATA_ROOT=/path/to/PDEdata
export OUT_ROOT=outputs/baselines_large
export DEVICE=cuda
export DATA_MANIFEST_DIR=outputs/data_protocol
```

Formal matrices are built only from a complete, passing data report that is
bound to the exact experiment YAML. Verify every design first; editing a YAML
after verification invalidates its report and requires another full pass:

```bash
for matrix in \
  sanity_main main_results sensor_count_ablation noise_ablation \
  sensor_mode_ablation time_varying_sensor_ablation \
  runtime_budget_ablation train_size_ablation
do
  python scripts/verify_data_protocol.py \
    --config "configs/experiments/${matrix}.yaml" \
    --data-root "$DATA_ROOT" \
    --output-dir "${DATA_MANIFEST_DIR}/${matrix}/full" \
    --full
done
```

The reports land at
`$DATA_MANIFEST_DIR/<matrix>/full/data_protocol_report.json`. Build the matrices
only after every report above passes:

```bash
bash scripts/experiments/00_build_matrices.sh
```

The historical `experiment_plan_v2` matrix and its outputs are audit evidence;
do not rebuild, rerun, move, or overwrite them. Corrected protocol-v2 work uses
the separate `experiment_plan_v2_corrected` namespace.

Run the small sanity matrix:

```bash
N_JOBS=1 bash scripts/experiments/01_run_sanity_main.sh
```

Run the main paper matrix:

```bash
N_JOBS=1 bash scripts/experiments/02_run_main_results_local.sh
```

Run a selected ablation:

```bash
ABLATION=sensor_count_ablation N_JOBS=1 bash scripts/experiments/04_run_ablation_local.sh
```

Supported ablation names are:

```text
sensor_count_ablation
noise_ablation
sensor_mode_ablation
time_varying_sensor_ablation
runtime_budget_ablation
train_size_ablation
```

Check status:

```bash
bash scripts/experiments/08_status.sh
```

Baseline runs save reusable model checkpoints by default. Each run writes
`<output_dir>/<run_prefix>.pt`, and records the path in `summary.json` and the
result tables as `checkpoint_path`. Formal amortized runs require this artifact
for provenance validation. Only non-formal smoke/debug runs may disable it:

```bash
python -m baselines.run ... --no-save-checkpoint
```

Reload a saved model in Python:

```python
from baselines.run import load_baseline_checkpoint

model = load_baseline_checkpoint("outputs/.../run_id.pt", map_location="cpu")
model.eval()
```

Retry failed runs:

```bash
RETRY_LIMIT=3 bash scripts/experiments/09_retry_failed.sh
```

Aggregate results:

```bash
bash scripts/experiments/06_aggregate_main_results.sh
bash scripts/experiments/07_aggregate_ablations.sh
```

## Slurm

Submit the main matrix:

```bash
DATA_ROOT=/path/to/PDEdata \
OUT_ROOT=outputs/baselines_large \
PARTITION=gpu \
GRES=gpu:1 \
MAX_ARRAY_CONCURRENT=16 \
bash scripts/experiments/03_submit_main_results_slurm.sh
```

Submit an ablation matrix:

```bash
ABLATION=noise_ablation \
DATA_ROOT=/path/to/PDEdata \
OUT_ROOT=outputs/baselines_large \
PARTITION=gpu \
GRES=gpu:1 \
MAX_ARRAY_CONCURRENT=16 \
bash scripts/experiments/05_submit_ablation_slurm.sh
```

## Testing

```bash
python -m py_compile \
  baselines/run.py baselines/aggregate_results.py baselines/experiment_matrix.py \
  baselines/capabilities.py baselines/common/*.py baselines/methods/*.py \
  scripts/experiments/*.py tests/test_*.py

bash -n scripts/experiments/*.sh
python -m pytest -q
```

## Principles

- Do not present local compact implementations as official baselines.
- Do not mix adapted/surrogate results into the main paper table.
- Unsupported combinations must skip with an explicit reason.
- `pde_opt` is a canonical mathematical baseline, not an official-code claim.
