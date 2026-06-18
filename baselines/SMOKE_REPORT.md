# Smoke Test Report

Date: 2026-06-17
Environment: `conda run -n FM4PDEbaseline`

## 2026-06-19 Official-Aligned Baseline Reimplementation Verification

This update adds main-table eligible official-aligned reimplementations where the vendored official repositories are script/checkpoint/data-pipeline oriented rather than safely importable as libraries.

Focused commands completed during implementation:

```bash
python -m py_compile baselines/methods/base.py baselines/methods/official.py baselines/capabilities.py baselines/run.py baselines/aggregate_results.py baselines/methods/ifno.py baselines/methods/ifno_official_aligned.py baselines/methods/vivid.py baselines/methods/vivid_official_aligned.py baselines/methods/pc_bnn.py
python -m baselines.run --baseline ifno --pde darcy --task forward --experiment-mode smoke --dry-run --synthetic-data --synthetic-resolution 8 --train-size 4 --test-size 2 --batch-size 1 --epochs 1 --implementation-mode official_aligned --output-dir outputs/baselines/dev_smoke_ifno_forward
python -m baselines.run --baseline ifno --pde darcy --task inverse --experiment-mode smoke --dry-run --synthetic-data --synthetic-resolution 8 --train-size 4 --test-size 2 --batch-size 1 --epochs 1 --implementation-mode official_aligned --output-dir outputs/baselines/dev_smoke_ifno_inverse
python -m baselines.run --baseline pc_bnn --pde shallow_water --task sparse_solution --experiment-mode smoke --dry-run --synthetic-data --synthetic-resolution 8 --train-size 2 --test-size 1 --batch-size 1 --num-sensors 4 --steps 1 --particles 2 --implementation-mode official_aligned --output-dir outputs/baselines/dev_smoke_pcbnn_swe
python -m baselines.run --baseline vivid --pde reaction_diffusion --task sparse_solution --task-group time_varying --sensor-mode time_varying --load-full-trajectory --experiment-mode smoke --dry-run --synthetic-data --synthetic-resolution 8 --train-size 4 --test-size 2 --batch-size 1 --num-sensors 4 --output-dir outputs/baselines/dev_smoke_vivid_auto
python -m pytest -q tests/test_ifno_official_aligned.py tests/test_vivid_official_aligned.py tests/test_pcbnn_official_aligned.py tests/test_preflight_backend_availability.py tests/test_paper_table_eligibility.py tests/test_metadata_backend_fields.py tests/test_baseline_forward_shapes.py
python -m py_compile baselines/run.py baselines/aggregate_results.py baselines/experiment_matrix.py baselines/capabilities.py baselines/common/*.py baselines/methods/*.py tests/test_*.py
bash -n scripts/baselines/*.sh
python -m pytest -q
python -m baselines.experiment_matrix --dump-matrix --output outputs/baselines/capability_matrix_test
python -m baselines.run --baseline ifno --pde darcy --task forward --experiment-mode smoke --dry-run --synthetic-data --synthetic-resolution 8 --train-size 4 --test-size 2 --batch-size 1 --output-dir outputs/baselines/smoke_ifno_forward
python -m baselines.run --baseline ifno --pde darcy --task inverse --experiment-mode smoke --dry-run --synthetic-data --synthetic-resolution 8 --train-size 4 --test-size 2 --batch-size 1 --output-dir outputs/baselines/smoke_ifno_inverse
python -m baselines.run --baseline vivid --pde reaction_diffusion --task sparse_solution --task-group time_varying --sensor-mode time_varying --load-full-trajectory --experiment-mode smoke --dry-run --synthetic-data --synthetic-resolution 8 --train-size 4 --test-size 2 --batch-size 1 --num-sensors 8 --output-dir outputs/baselines/smoke_vivid_da
```

Verified behavior:

- iFNO full forward/full inverse run as `implementation_mode_effective=official_aligned`, set `official_reimplementation_success=true`, and are `paper_table_eligible=true`; sparse iFNO remains unsupported.
- VIVID time-varying DA defaults to `official_aligned_vivid_invobs_reimplementation`, trains/uses an inverse observation operator, records `assimilation_mode=full_trajectory`, and is main-table eligible. Explicit `implementation_mode=adapted` VIVID-style remains supplement-only.
- PC-BNN shallow-water sparse reconstruction runs as `official_aligned_pcbnn_reimplementation` and is main-table eligible. Scalar generic PC-BNN remains supplement-only.
- These rows do not claim direct official import. `official_import_success=false` is paired with `official_reimplementation_success=true` and alignment notes.

## 2026-06-18 Paper-Mode Gating Verification

The current baseline layer is an official/native/canonical gating framework. Paper main rows require both a supported capability and an eligible implementation mode; adapted, style, surrogate, fallback, or local debug rows are routed to supplement or `skipped_combinations.jsonl`.

Commands completed in the active workspace during this update:

```bash
python -m py_compile baselines/capabilities.py baselines/run.py baselines/aggregate_results.py baselines/experiment_matrix.py baselines/methods/vivid.py baselines/methods/voronoicnn.py baselines/methods/var4d.py baselines/methods/pinn_sparse.py
python -m pytest -q tests/test_official_mode_no_fallback.py tests/test_paper_table_eligibility.py tests/test_time_varying_da_matrix.py tests/test_assimilation_mode.py tests/test_time_varying_sensor_guard.py tests/test_sparse_inverse_physics_baselines.py tests/test_aggregate_main_supplement_guard.py
python -m py_compile baselines/run.py baselines/aggregate_results.py baselines/experiment_matrix.py baselines/capabilities.py baselines/common/*.py baselines/methods/*.py tests/test_*.py
bash -n scripts/baselines/*.sh
python -m pytest -q
python -m baselines.experiment_matrix --dump-matrix --output outputs/baselines/capability_matrix_test
```

Verified behavior:

- `implementation_mode: official` hard-fails when official FNO imports are unavailable.
- `implementation_mode: official_or_skip` writes a complete skip row and does not write `results_summary.jsonl` when official FNO imports are unavailable.
- Main-table eligibility rejects `fallback_used=true`, VIVID-style adapter statuses, RecFNO-UNet-as-VoronoiCNN, and official requirements satisfied by `canonical_math`.
- PDE-Opt and PINN-Sparse static sparse inverse run on tiny synthetic `poisson`, `helmholtz`, `darcy`, and `steady_heat_conduction` fixtures.
- Burgers time-varying 4D-Var now uses `full_trajectory` mode for `[B,1,T,X]` trajectories.

Not paper main baselines: compact/local FNO, simplified local iFNO coupling blocks, VIVID-style refinement, generic PC-BNN particles outside official channel assumptions, and RecFNO UNet used as a VoronoiCNN surrogate.

## 2026-06-18 Targeted Fix Verification

Earlier shell environment observed during the 2026-06-18 targeted fix pass: `/opt/miniconda3/bin/python` 3.13.5 did not have `torch` or `pytest` installed, and `/opt/miniconda3/envs/research/bin/python` also lacked `torch`, `pytest`, `yaml`, `h5py`, and `scipy`. That note is retained as historical environment context; the paper-mode gating checks above were run with the active workspace Python.

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

## Legacy Smoke Wrapper Status

This table records smoke/debug wrapper coverage only. It does not imply paper main-table eligibility.

| Baseline | Implemented | Wrapped/source referenced | Dry-run passed | Notes |
| --- | ---: | ---: | ---: | --- |
| FNO | yes | neuraloperator/FNO sources referenced | yes | compact PyTorch FNO2d is smoke/adapted only |
| DeepONet | yes | DeepONet/DeepXDE referenced | yes | branch/trunk operator model |
| iFNO | yes | official local `iFNO/` referenced | yes | official-aligned invertible FNO reimplementation is main-eligible for full forward/inverse; simplified local coupling is supplement/debug only |
| RecFNO | yes | local `RecFNO/` referenced | yes | mask and Voronoi embeddings |
| Senseiver | yes | local `Senseiver/` referenced | yes | lightweight Perceiver-IO variant |
| VoronoiCNN | yes | local `Voronoi-CNN/` referenced | yes | official-architecture Conv2D reimplementation can be main; RecFNO UNet surrogate cannot |
| PINN-Sparse | yes | DeepXDE/PINN referenced | yes | neural-field per-instance optimizer with shared residuals |
| PC-BNN | yes | PC-BNN upstream referenced | yes | official-aligned for matched shallow-water three-channel sparse reconstruction; generic SVGD is supplement only |
| PDE-Opt | yes | direct implementation | yes | per-instance PDE-constrained grid optimization |
| 4D-Var | yes | direct implementation | yes | full-space weak 4D-Var; uses available trajectory segment |
| VIVID | yes | VIVID/invobs referenced | yes | official-aligned inverse-observation plus variational refinement is main-eligible for time-varying DA; VIVID-style adapted is supplement only |

## Limitations

- Smoke tests use synthetic data to avoid multi-GB real data reads; final paper numbers should be produced with the real `PDEdata` root and fixed experiment configs.
- Tiny native fixtures validate adapter file-format logic for all current PDEs, but not full-scale I/O throughput.
- NS/RD/SWE residuals are implemented from the data-generation summary and smoke-tested for finite values. They are finite-difference diagnostics and should be reported with the exact metadata/mesh assumptions from `baselines/README.md`.
- Final-state tasks for RD/SWE build `mode="two_level"` residuals from input/background and final prediction. Full trajectory states use `mode="full_trajectory"` when available, such as in full-space 4D-Var/VIVID state optimization.
- PINN-Sparse, PC-BNN, PDE-Opt, 4D-Var, and VIVID now optimize structured physics total loss by default; old `lambda_pde` configs scale interior, BC, and IC terms together unless `lambda_int`, `lambda_bc`, or `lambda_ic` are explicitly set.
- Optional advanced variants not included in the unified runner are iFNO VAE posterior inference and POD-reduced VIVID.
- The original `PDEdata/data_summary.md` reports a Burgers train/test distribution shift; experiments should verify whether this is intentional before comparing final test metrics.
