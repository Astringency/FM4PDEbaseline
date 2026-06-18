# Large-Scale External Baseline Experiments

## Goal

This system builds and runs a large pool of external-baseline experiments across the available FM4PDE PDE datasets. The default goal is to run every compatible external baseline on every compatible PDE/task/sensor/noise/seed combination, then choose paper main tables and appendix tables after aggregation.

Formal runs use real data only. Paper-mode commands generated here do not include `--dry-run`, `--synthetic-data`, `--allow-synthetic-fallback`, or `--prefer-test`.

## PDEs And Baselines

Default PDEs:

`darcy`, `poisson`, `helmholtz`, `nsnonbounded`, `burger`, `reaction_diffusion`, `shallow_water`, `heat`, `wave`, `advection_diffusion`, `steady_heat_conduction`.

Default external baselines:

`fno`, `deeponet`, `ifno`, `recfno`, `senseiver`, `voronoicnn`, `pinn_sparse`, `pc_bnn`, `pde_opt`, `var4d`, `vivid`.

`DiffusionPDE` and FM4PDE internal ablations are intentionally not part of this matrix.

## Task Groups

- `full_forward`: `forward`, all PDEs, `fno/deeponet/ifno`.
- `full_inverse`: `inverse`, all PDEs, `fno/deeponet/ifno`.
- `sparse_solution_amortized`: `sparse_solution`, all PDEs, `recfno/senseiver/voronoicnn/fno/deeponet`; `ifno` is recorded as skipped.
- `sparse_solution_physics`: `sparse_solution`, compatible PDEs, `pinn_sparse/pc_bnn/pde_opt/var4d/vivid`.
- `sparse_inverse`: `sparse_inverse`, all PDEs, `recfno/senseiver/voronoicnn/fno/deeponet`; per-instance baselines are skipped.
- `time_varying`: `sparse_solution` with `time_varying` sensors on `nsnonbounded/burger/reaction_diffusion/shallow_water`, using `var4d/vivid/senseiver`.

## Scalar Params

For supervised `forward`, `inverse`, and `sparse_inverse`, future PDEs (`heat`, `wave`, `advection_diffusion`, `steady_heat_conduction`) default to `scalar_param_mode=materialize`. Legacy PDEs default to `metadata`.

For `sparse_solution`, the default is `metadata`. Set `SCALAR_PARAM_MODE=materialize` when you want the sensitivity run.

Future PDE inverse targets may include initial/source fields plus scalar PDE parameters after materialization; this is expected and is recorded in result metadata.

## Loading Rules

Default data loading is `lazy`. `time_varying` and trajectory DA rows force `--load-full-trajectory`. Endpoint/two-level surrogate runs are not labeled as full-trajectory DA.

## Compatibility And Skips

Compatibility is centralized in [baselines/experiment_matrix.py](../baselines/experiment_matrix.py). Unsupported combinations are not run. They are written to `OUT_ROOT/skipped_combinations.jsonl` and matrix-specific `*_skipped.jsonl` files with a `reason` and `would_have_expanded` count.

Key skip rules:

- `ifno` is full `forward/inverse` only.
- `var4d/vivid` run only on time-dependent PDEs.
- `steady_heat_conduction` is steady and is skipped for `time_varying` and 4D-Var/VIVID.
- `sparse_inverse` skips per-instance baselines because they would need a PDE forward solver from predicted coefficients/initial state to observed solution sensors.
- `time_varying` sensors require trajectory-compatible PDEs and baselines adapted to 5D targets.

## Build Matrices

```bash
bash scripts/experiments/00_build_matrices.sh
```

This writes:

- `OUT_ROOT/matrices/sanity.jsonl`
- `OUT_ROOT/matrices/core.jsonl`
- `OUT_ROOT/matrices/full_all.jsonl`
- `OUT_ROOT/matrices/time_varying.jsonl`
- matching `.tsv` and `_summary.json` files

Defaults can be overridden with environment variables such as `OUT_ROOT`, `TRAIN_SIZE`, `TEST_SIZE`, `SEEDS`, `SENSOR_COUNTS`, `NOISE_LEVELS`, `DEVICE`, and `DATA_LOADING_MODE`.

## Sanity Run

```bash
DATA_ROOT=/path/to/PDEdata OUT_ROOT=outputs/baselines_large \
bash scripts/experiments/01_run_sanity.sh
```

Sanity defaults to `TRAIN_SIZE=128`, `TEST_SIZE=16`, `SEEDS=1`, `SENSOR_COUNTS=50`, and `NOISE_LEVELS=0.0`.

## Local Core Runs

Run a small slice first:

```bash
DATA_ROOT=/path/to/PDEdata OUT_ROOT=outputs/baselines_large N_JOBS=1 \
TASK_GROUPS="full_forward" \
bash scripts/experiments/02_run_core_local.sh
```

`N_JOBS` controls local parallelism. `CUDA_VISIBLE_DEVICES` can be set by the user.

## Full All Runs

`full_all` is large. The default matrix contains 34,542 actual run rows plus 16,560 skipped expanded combinations. It will not start unless explicitly confirmed:

```bash
DATA_ROOT=/path/to/PDEdata OUT_ROOT=outputs/baselines_large \
CONFIRM_FULL_ALL=1 TASK_GROUPS="full_forward sparse_solution_amortized" \
bash scripts/experiments/03_run_full_all_local.sh
```

## SLURM Array

```bash
MATRIX=outputs/baselines_large/matrices/core.jsonl \
OUT_ROOT=outputs/baselines_large \
DATA_ROOT=/path/to/PDEdata \
MAX_ARRAY_CONCURRENT=16 \
bash scripts/experiments/04_submit_slurm_array.sh
```

Useful SLURM env vars: `PARTITION`, `TIME`, `CPUS_PER_TASK`, `MEM`, `GRES`, `JOB_NAME`.

## Resume And Retry

Each run writes:

`run.started`, `run.running`, `run.done`, `run.failed`, `command.txt`, `env.txt`, `stdout.log`, `stderr.log`, `metadata.json`, and `run.status.json`.

`05_run_one.sh` skips completed runs unless `FORCE=1`. It skips failed runs unless `RETRY_FAILED=1`. Stale running locks are controlled by `LOCK_TIMEOUT_SECONDS`.

Generate a retry matrix:

```bash
MATRIX=outputs/baselines_large/matrices/core.jsonl \
OUT_ROOT=outputs/baselines_large RETRY_LIMIT=3 \
bash scripts/experiments/08_retry_failed.sh
```

## Status

```bash
OUT_ROOT=outputs/baselines_large bash scripts/experiments/07_status.sh
```

This writes `OUT_ROOT/status.csv` and `OUT_ROOT/status.md`.

## Aggregate

```bash
OUT_ROOT=outputs/baselines_large bash scripts/experiments/06_aggregate_all.sh
```

Outputs are grouped under:

- `aggregate/full_forward`
- `aggregate/full_inverse`
- `aggregate/sparse_solution_amortized`
- `aggregate/sparse_solution_physics`
- `aggregate/sparse_inverse`
- `aggregate/time_varying`
- `aggregate/all`

Each directory contains `summary.csv`, `summary.json`, and `latex_table.csv`.

## Paper Table Selection

Use `aggregate/all/summary.csv` as the full result pool. Main tables should usually filter by `task_group`, `pde`, `baseline`, `num_sensors`, `sensor_mode`, `noise_level`, and `scalar_param_mode`. Appendix tables can include the wider sensor/noise sweeps and the skipped-combination rationale.

## Notes

- Do not launch `full_all` blindly.
- Per-instance baselines are slow and default to `batch_size=1`.
- `time_varying` is only for trajectory-compatible PDE/baseline combinations.
- `sparse_inverse` per-instance baselines are skipped by design.
- Existing `scripts/baselines/run_paper_*.sh` remain available as legacy/simple wrappers.
