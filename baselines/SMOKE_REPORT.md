# Smoke Test Report

Date: 2026-06-17
Environment: `conda run -n FM4PDEbaseline`

## 2026-06-18 Targeted Fix Verification

Current shell environment: `/opt/miniconda3/bin/python` 3.13.5. This environment does not have `torch` or `pytest` installed, and `/opt/miniconda3/envs/research/bin/python` also lacks `torch`, `pytest`, `yaml`, `h5py`, and `scipy`. Runner and pytest execution are therefore blocked in this shell until the intended `FM4PDEbaseline` environment is activated or installed.

Commands that completed in the current shell:

```bash
python -m py_compile baselines/run.py baselines/common/data_adapter.py baselines/common/physics.py baselines/aggregate_results.py baselines/experiment_matrix.py baselines/methods/*.py tests/test_*.py
bash -n scripts/baselines/run_paper_full_operator.sh scripts/baselines/run_paper_sparse_reconstruction.sh scripts/baselines/run_paper_physics_da.sh scripts/baselines/run_paper_plan_lightweight.sh scripts/baselines/run_paper_all.sh scripts/baselines/smoke_all.sh
python -m baselines.aggregate_results outputs/baselines --output-dir outputs/baselines/aggregate_test
```

Commands blocked by missing dependencies in the current shell:

```bash
python -m baselines.run --baseline fno --pde darcy --task forward --dry-run --synthetic-data --experiment-mode debug --test-size 4 --train-size 8 --val-size 0 --batch-size 2
python -m pytest
bash scripts/baselines/smoke_all.sh
```

Targeted fixes added after the original report: paper `VAL_SIZE=0`, strict `nsnonbounded` test-file filtering, per-instance spec-only train loading, sparse-task background preservation for residuals, per-sample physics metric values, pooled summary-only aggregation, explicit backend/fallback flags, and paper matrix compatibility skipping.

## Commands Run

```bash
python -m compileall -q baselines
conda run -n FM4PDEbaseline pytest -q
conda run -n FM4PDEbaseline bash scripts/baselines/smoke_all.sh
python -m baselines.run --baseline fno --pde darcy --task forward --dry-run --synthetic-data --synthetic-resolution 16 --train-size 2 --batch-size 1
python -m baselines.run --baseline fno --pde darcy --task forward --data-root /home/tat512/C01Python/PDEdata --prefer-test --dry-run --train-size 1 --batch-size 1
python -m baselines.run --baseline fno --pde nsnonbounded --task forward --dry-run --synthetic-data --synthetic-resolution 16 --train-size 2 --batch-size 1
python -m baselines.run --baseline fno --pde reaction_diffusion --task forward --dry-run --synthetic-data --synthetic-resolution 16 --train-size 2 --batch-size 1
```

## Results

- `compileall`: passed.
- `pytest`: 51 passed.
- `scripts/baselines/smoke_all.sh`: passed for all 11 baseline wrappers on deterministic synthetic data.
- Real Darcy test-file dry-run: passed, with input/target/prediction shape `[1,1,128,128]`.
- Structured physics metrics produce finite `pde_residual`, `bc_residual`, `ic_residual`, and `physics_loss` values for the current PDE set.
- Per-instance methods (`pinn_sparse`, `pc_bnn`, `pde_opt`, `var4d`, `vivid`) write nonzero `inference_optimization_time`; amortized methods write `0.0`.
- Outputs were written under `outputs/baselines/debug` and `outputs/baselines/smoke`.

## PDE Adapter Status

| PDE | Native tiny fixture load | Synthetic smoke | Channel/trajectory status |
| --- | ---: | ---: | --- |
| `darcy` | passed | passed | `[a,p]`, static `[N,2,H,W]`; real HDF5 uses `[H,W,N]` raw layout |
| `poisson` | passed | passed | `[f,phi]`, static `[N,2,H,W]` |
| `helmholtz` | passed | passed | `[f,psi]`, static `[N,2,H,W]`, `k=1` default |
| `nsnonbounded` | passed | passed | vorticity trajectory `[N,1,11,H,W]` preserved |
| `burger` | passed | passed | `output` as `[N,1,T,X]`, `input` kept as metadata |
| `reaction_diffusion` | passed | passed | `[u,v]` trajectory `[N,2,T,H,W]` preserved |
| `shallow_water` | passed | passed | conservative `[h,hu,hv]` trajectory `[N,3,T,H,W]` preserved |
| `heat` | fallback only | passed | reserved key |
| `wave` | fallback only | passed | reserved key |
| `advection_diffusion` | fallback only | passed | reserved key |

## Baseline Status

| Baseline | Implemented | Wrapped/source referenced | Dry-run passed | Notes |
| --- | ---: | ---: | ---: | --- |
| FNO | yes | neuraloperator/FNO sources referenced | yes | compact PyTorch FNO2d |
| DeepONet | yes | DeepONet/DeepXDE referenced | yes | branch/trunk operator model |
| iFNO | yes | official local `iFNO/` referenced | yes | bidirectional Fourier coupling core; optional VAE/posterior not included |
| RecFNO | yes | local `RecFNO/` referenced | yes | mask and Voronoi embeddings |
| Senseiver | yes | local `Senseiver/` referenced | yes | lightweight Perceiver-IO variant |
| VoronoiCNN | yes | local `Voronoi-CNN/` referenced | yes | PyTorch adaptation |
| PINN-Sparse | yes | DeepXDE/PINN referenced | yes | neural-field per-instance optimizer with shared residuals |
| PC-BNN | yes | PC-BNN upstream referenced | yes | SVGD particles with predictive mean/std |
| PDE-Opt | yes | direct implementation | yes | per-instance PDE-constrained grid optimization |
| 4D-Var | yes | direct implementation | yes | full-space weak 4D-Var; uses available trajectory segment |
| VIVID | yes | VIVID/invobs referenced | yes | Voronoi inverse op + variational refinement |

## Limitations

- Smoke tests use synthetic data to avoid multi-GB real data reads; final paper numbers should be produced with the real `PDEdata` root and fixed experiment configs.
- Tiny native fixtures validate adapter file-format logic for all current PDEs, but not full-scale I/O throughput.
- NS/RD/SWE residuals are implemented from the data-generation summary and smoke-tested for finite values. They are finite-difference diagnostics and should be reported with the exact metadata/mesh assumptions from `baselines/README.md`.
- Final-state tasks for RD/SWE build `mode="two_level"` residuals from input/background and final prediction. Full trajectory states use `mode="full_trajectory"` when available, such as in full-space 4D-Var/VIVID state optimization.
- PINN-Sparse, PC-BNN, PDE-Opt, 4D-Var, and VIVID now optimize structured physics total loss by default; old `lambda_pde` configs scale interior, BC, and IC terms together unless `lambda_int`, `lambda_bc`, or `lambda_ic` are explicitly set.
- Optional advanced variants not included in the unified runner are iFNO VAE posterior inference and POD-reduced VIVID.
- The original `PDEdata/data_summary.md` reports a Burgers train/test distribution shift; experiments should verify whether this is intentional before comparing final test metrics.
