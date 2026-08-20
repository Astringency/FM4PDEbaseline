# FM4PDE Baseline

External baseline experiments for the protocol in
[`docs/baseline_exp.md`](docs/baseline_exp.md). The formal `main_results`
design is the single source of truth and contains 80 runnable experiments.
`baselines/configs/paper.yaml` contains the shared runtime and per-method
recipes. Capability and result metadata distinguish direct official code,
official-architecture adapters, canonical mathematical baselines, and local
adaptations; these labels must not be treated as interchangeable.

## Formal experiment protocol

The formal split uses one seed, a 50,000-sample train/validation pool split as
45,000 train + 5,000 validation, and 1,000 test samples. With no explicit
validation files, validation is the final non-overlapping portion of the
training pool.

| Task | PDE scope | Baselines |
|---|---|---|
| Full Forward | Poisson, Helmholtz, Darcy, Navier-Stokes | FNO, DeepONet, iFNO |
| Full Inverse | Poisson, Helmholtz, Darcy, Navier-Stokes | iFNO |
| Sparse Solution Reconstruction | Five formal PDEs | RecFNO, Senseiver, VoronoiCNN |
| Burgers Sparse Solution Reconstruction | Burgers only | Var4D, VIVID, plus the three amortized reconstruction methods |
| Sparse Forward | Poisson, Helmholtz, Darcy, Navier-Stokes for amortized methods; no Navier-Stokes physics baseline | RecFNO, Senseiver, VoronoiCNN, PINN-sparse, PDE-opt, PC-BNN |
| Sparse Inverse | Poisson, Helmholtz, Darcy, Navier-Stokes for amortized methods; no Navier-Stokes physics baseline | RecFNO, Senseiver, VoronoiCNN, PINN-sparse, PDE-opt, PC-BNN |

Burgers reconstructs the complete `128 × 128` time-space trajectory under two
protocols: 500 total scattered observations sampled independently per example,
or 5 complete time slices (`5 × 128` observations). Var4D and VIVID are not
configured for `nsnonbounded`, Reaction-Diffusion, or Shallow-Water.

Important method budgets and adaptations:

- Senseiver and VoronoiCNN use early-stopping patience 20 with minimum
  improvement `1e-4`.
- iFNO records requested and completed budgets independently for operator
  pretraining, VAE pretraining, and joint training. Each stage monitors the
  5,000-sample validation split and restores its own best checkpoint. Operator
  and VAE pretraining use patience 20 with minimum improvement `1e-4` and
  minimum budgets of 50 and 30 epochs, respectively; joint training keeps its
  configured patience 12. This validation stopping is an FM4PDE compute-budget
  adaptation on top of the official fixed-epoch three-stage recipe.
- PINN-sparse runs at most 1,000 Adam iterations followed by 500 L-BFGS steps;
  both phases and their combined budget are recorded separately.
- Var4D performs at most 500 per-sample L-BFGS-B iterations over a
  covariance-decorrelated Burgers initial-state increment; a differentiable
  pseudo-spectral solver propagates the complete trajectory under a strong
  dynamical constraint.
- VIVID trains the official VCNN architecture with Adam/MSE for 20 epochs at
  learning rate `1e-4` and effective batch size 64, then performs at most 1,000
  per-sample L-BFGS-B iterations of the original three-term VIVID objective.

Var4D is now labeled `canonical_math`: it is strong-constraint 4D-Var with a
Burgers-specific propagator, not an end-to-end official-code claim. VIVID is an
`official_architecture` Burgers task adapter: it preserves the vendored 6×48
hidden `8×8` VCNN plus linear output layer, Glorot initialization, official
training budget, $J_b+J_p+J_o$ objective, covariance scales, and L-BFGS-B
recipe. The original VIVID is 3D-Var, so the adapter treats the complete
time-space solution as one two-dimensional state; it does not silently turn
VIVID into 4D-Var. Both are unified-table eligible but remain
`official_native_eligible=false` because their Burgers data/operator adapters
are not the official shallow-water experiment. See the [Var4D/VIVID
official-source audit](docs/var4d_vivid_official_audit.md) for retained core
details and disclosed adaptations.

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

## Environment

Create the pinned environment once from the repository root:

```bash
conda env create -f environment.yml
conda activate fm4pdebaseline
```

## Quick start

The recommended formal entry point runs all 80 rows in `main_results`, including
data verification, matrix construction, resumable execution, aggregation, and
plotting. Replace `DATA_ROOT` with the directory containing the files declared
in `configs/data_files/formal_128.yaml`:

```bash
DATA_ROOT=/absolute/path/to/PDEdata \
OUT_ROOT=outputs/main_results \
GPUS=0,1 \
JOBS_PER_GPU=1 \
PLOT_LIMIT=100 \
bash scripts/run_baseline.sh
```

The script verifies data, builds the matrix, resumes unfinished experiments,
collects results, renders PDFs in a separate process, and prints status when it
finishes or is interrupted. It uses an output lock and will not restart a
`run.running` row whose process is alive.

After changing iFNO training code, do not rebuild and launch the complete
`main_results` matrix: the formal code fingerprint changes. Stop the old jobs
and create an isolated recovery matrix instead. The following command launches
only the eight full-field iFNO rows under a new matrix/output namespace, so all
existing completed artifacts remain untouched. `JOBS_PER_GPU=1` is recommended
for 24 GB GPUs because concurrent iFNO jobs can exhaust device memory.

```bash
DATA_ROOT=/absolute/path/to/PDEdata \
OUT_ROOT=/absolute/path/to/outputs/FM4PDEbaseline \
MATRIX_NAME=ifno_earlystop \
BASELINES=ifno \
GPUS=0,1 \
JOBS_PER_GPU=1 \
PLOT_LIMIT=100 \
bash scripts/run_baseline.sh
```

Preview the complete workflow without launching training:

```bash
DATA_ROOT=/absolute/path/to/PDEdata \
OUT_ROOT=outputs/main_results \
GPUS=0,1 \
JOBS_PER_GPU=1 \
DRY_RUN=1 \
bash scripts/run_baseline.sh
```

`PLOT_LIMIT` is applied per experiment. It defaults to `100`; set it to `0`
to render every evaluated sample, or set `PLOT_SAMPLES=0` to skip plotting. Use
`GPUS=cpu` for CPU execution. Formal Var4D and VIVID runs are computationally
expensive because their L-BFGS-B optimization is performed separately for every
test sample; `JOBS_PER_GPU=1` is the conservative starting point.

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

### Run only Burgers Var4D/VIVID

After data verification and matrix construction, the following code derives the
matching row indices from the matrix instead of relying on fixed row numbers.
With the current formal design it selects four rows: two methods times the 500
scattered-observation and 5-time-slice protocols.

```bash
export DATA_ROOT=/absolute/path/to/PDEdata
export OUT_ROOT=outputs/main_results
export MATRIX="$OUT_ROOT/matrices/main_results.jsonl"

VARIATIONAL_INDICES="$(
python - "$MATRIX" <<'PY'
import json
import sys
from pathlib import Path

matrix = Path(sys.argv[1])
rows = [json.loads(line) for line in matrix.read_text().splitlines() if line.strip()]
selected = [
    str(index)
    for index, row in enumerate(rows)
    if row.get("pde") == "burger"
    and row.get("task") == "sparse_solution"
    and row.get("baseline") in {"var4d", "vivid"}
    and not row.get("skip_reason")
]
if len(selected) != 4:
    raise SystemExit(f"expected 4 Burgers Var4D/VIVID rows, found {len(selected)}")
print(",".join(selected))
PY
)"

python scripts/run_experiments.py "$MATRIX" \
  --data-root "$DATA_ROOT" \
  --gpus 0,1 \
  --jobs-per-gpu 1 \
  --indices "$VARIATIONAL_INDICES"

python scripts/run_experiments.py "$MATRIX" --status
```

The publication collector intentionally requires every runnable row in the
matrix. Therefore, run the collection command below only after the entire
formal matrix has completed, not immediately after this four-row selection.

### Var4D/VIVID smoke checks

These synthetic CPU commands exercise the complete method paths without the PDE
dataset. They deliberately reduce every budget to one and are diagnostics only;
their outputs are not eligible for the formal comparison.

```bash
python -m baselines.run \
  --baseline var4d --pde burger --task sparse_solution \
  --task-group time_varying_da_main \
  --config baselines/configs/paper.yaml \
  --experiment-mode smoke --synthetic-data --synthetic-resolution 128 \
  --train-size 2 --val-size 1 --test-size 1 --batch-size 1 \
  --num-sensors 500 --sensor-mode random_per_sample \
  --sensor-budget-mode total --load-full-trajectory \
  --steps 1 --device cpu --output-dir outputs/smoke/var4d \
  --no-save-checkpoint --no-save-sample-artifacts

python -m baselines.run \
  --baseline vivid --pde burger --task sparse_solution \
  --task-group time_varying_da_main \
  --config baselines/configs/paper.yaml \
  --experiment-mode smoke --synthetic-data --synthetic-resolution 128 \
  --train-size 2 --val-size 1 --test-size 1 --batch-size 1 --epochs 1 \
  --num-sensors 500 --sensor-mode random_per_sample \
  --sensor-budget-mode total --load-full-trajectory \
  --refine-steps 1 --device cpu --output-dir outputs/smoke/vivid \
  --no-save-checkpoint --no-save-sample-artifacts
```

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

`time_varying_sensor_ablation` varies the total scattered-observation count on
Burgers for Var4D, VIVID, and Senseiver. The Var4D/VIVID portion of
`runtime_budget_ablation` also uses Burgers Sparse Solution Reconstruction;
neither ablation generates legacy `nsnonbounded` Var4D/VIVID rows.

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
source snapshots. The snapshots currently lack verified upstream commit/tag
records; see [`baselines/OFFICIAL_SOURCE_METADATA.md`](baselines/OFFICIAL_SOURCE_METADATA.md)
before making official-source claims.
