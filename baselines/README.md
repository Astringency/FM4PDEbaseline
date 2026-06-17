# FM4PDE Baseline Framework

This module adds a reproducible JMLR-revision baseline layer without modifying the original FM4PDE training or sampling code. It provides shared data adapters, sensor/noise generation, metrics, logging, and method wrappers under a single command-line interface.

## Step 0 Findings

Files inspected in the original FM4PDE repository:

- `data/load.py`: loads each PDE and returns tensors for FM4PDE generative training.
- `data/DataGen/`: documents and scripts for static and time-dependent data generation.
- `train.py`, `train_arg_parser.py`, `training/train_loop.py`: FM4PDE trains a flow-matching generative model on full paired fields normalized by `data/transform.py`.
- `sample.py`, `sampling/generate_pde.py`, `sampling/sampler.py`: sampling applies observation/PDE guidance after loading one test instance.
- `configs/*.yaml`: sampling configs define PDE-specific data keys, pretrained model paths, observation counts, and guide weights.

Important behavior:

- FM4PDE training input is generally normalized to `[N,C,128,128]`, but this is not the raw data layout for all PDEs.
- `reaction_diffusion` raw HDF5 contains trajectories; the FM4PDE loader uses one intermediate frame and the final frame.
- `shallow_water` raw HDF5 contains time series for `h`, `hu`, `hv`; the FM4PDE loader uses first and final states.
- `nsnonbounded` raw data stores `w0` and a 10-step vorticity trajectory; the FM4PDE loader concatenates them as channel-like snapshots.
- `burger` stores an initial 1D field and a 2D space-time output. FM4PDE currently uses only the output as a single channel.
- Darcy MATLAB v7.3/HDF5 files store arrays as `[H,W,N]` (`thresh_a_data`, `thresh_p_data`), matching the original FM4PDE loader's `transpose(3,0,1,2)` path. The adapter supports both `[H,W,N]` and `[N,H,W]` layouts.
- Existing PDE guidance for Navier-Stokes in `sampling/generate_pde.py` is not a reliable vorticity transport residual. This framework does not reuse that placeholder; it defines Navier-Stokes vorticity, Reaction-Diffusion, and Shallow-Water residuals in `baselines/common/physics.py` using the equation forms and parameters from `data/DataGen/pde_data_generation_summary.md`.
- Existing repository includes FNO reference code in PDEBench data generation, but no unified FNO/DeepONet/DiffusionPDE baseline runner.

## Canonical Data Contract

`baselines/common/data_adapter.py` defines:

```python
@dataclass
class PDEBatch:
    pde_name: str
    task: str
    full_tensor: torch.Tensor
    input_fields: torch.Tensor
    target_fields: torch.Tensor
    coords: torch.Tensor | None
    mask: torch.Tensor | None
    obs_values: torch.Tensor | None
    obs_coords: torch.Tensor | None
    channel_names: list[str]
    metadata: dict
```

Canonical layouts:

- Static 2D PDEs: `full_tensor = [N,C,H,W]`.
- Time-dependent 2D trajectories: `full_tensor = [N,C,T,H,W]`.
- Burgers: `full_tensor = [N,1,T,X]`, with initial 1D condition preserved in `metadata["initial_1d"]` when present.
- Task tensors are channel-first 2D grids when possible. For NS forward, trajectory targets are flattened to `[N,T,H,W]` for 2D amortized baselines while full trajectory remains in `full_tensor`.

## PDE Channel Mapping

| PDE | Raw/canonical channels | Forward | Inverse |
| --- | --- | --- | --- |
| `darcy` | `[a,p]` | `a -> p` | `p -> a` |
| `poisson` | `[f,phi]` | `f -> phi` | `phi -> f` |
| `helmholtz` | `[f,psi]` | `f -> psi` | `psi -> f` |
| `nsnonbounded` | `[w0,w_t1...w_t10]` as `[N,1,11,H,W]` | `w0 -> trajectory` | trajectory/final-view -> initial vorticity |
| `burger` | `input` 1D plus `output` `[T,X]` | repeated `u0 -> u(t,x)` | `u(t,x) -> u0` |
| `reaction_diffusion` | `[u,v]` trajectory | `[u_t0,v_t0] -> [u_T,v_T]` | final -> initial |
| `shallow_water` | conservative `[h,hu,hv]` trajectory | initial -> final | final -> initial |
| `heat` | `[u0,alpha,uT,alpha]` synthetic/future layout | `[u0,alpha] -> [uT,alpha]` | add concrete file adapter when files exist |
| `wave` | `[u0,v0,uT,vT]` synthetic/future layout | `[u0,v0] -> [uT,vT]` | add concrete file adapter when files exist |
| `advection_diffusion` | `[u0,b_x,b_y,kappa,uT,b_x,b_y,kappa]` synthetic/future layout | params + initial -> final + params | add concrete file adapter when files exist |
| `steady_heat_conduction` | `[f,u_D,u,u_D]` synthetic/future layout | `[f,u_D] -> [u,u_D]` | add concrete file adapter when files exist |

## Tasks

- `forward`: coefficient/source/initial state to solution/final/trajectory.
- `inverse`: solution/final state to coefficient/source/initial state.
- `sparse_solution`: sparse target observations to full target reconstruction.
- `sparse_inverse`: sparse target observations to inverse input reconstruction.
- `both`/`joint`: full pair reconstruction; primarily for compatibility.

All sparse tasks use `baselines/common/sensors.py` to create reusable masks, observations, noisy observations, masked grids, and Voronoi-like grids.

## Baseline Status

| Baseline | Local wrapper | Data input | Status |
| --- | --- | --- | --- |
| FNO | `baselines/methods/fno.py` | full or masked grid | implemented local compact FNO |
| DeepONet | `baselines/methods/deeponet.py` | flattened field or sensor values + query coords | implemented local branch/trunk DeepONet |
| iFNO | `baselines/methods/ifno.py` | forward/inverse grid | implemented local bidirectional iFNO core |
| RecFNO | `baselines/methods/recfno.py` | `[masked/voronoi grid, mask, coords]` | implemented local RecFNO with mask and Voronoi embeddings |
| Senseiver | `baselines/methods/senseiver.py` | `(obs_coords, obs_values)` + query coords | implemented local Perceiver-IO/Senseiver variant |
| VoronoiCNN | `baselines/methods/voronoicnn.py` | `[voronoi_grid, mask, coords]` | implemented local VoronoiCNN |
| PINN-Sparse | `baselines/methods/pinn_sparse.py` | per-instance sparse observations | implemented per-instance neural-field PINN |
| PC-BNN | `baselines/methods/pc_bnn.py` | per-instance sparse observations | implemented SVGD particle PC-BNN |
| PDE-Opt | `baselines/methods/pde_opt.py` | per-instance grid optimization | implemented PDE-constrained optimization |
| 4D-Var | `baselines/methods/var4d.py` | per-instance assimilation | implemented full-space weak 4D-Var |
| VIVID | `baselines/methods/vivid.py` | Voronoi inverse operator + variational refinement | implemented full-space VIVID |

Detailed paper/code sources are in `baselines/BASELINE_SOURCES.md`.

Official implementation reuse:

- `fno` first tries the vendored `offical/neuraloperator` FNO. If optional dependencies such as `tensorly` are unavailable, it falls back to the vendored RecFNO `VoronoiFNO2d` adapter, and only then to the local compact FNO.
- `recfno` uses the vendored `offical/RecFNO/model/fno.py::VoronoiFNO2d` as its network core; the local code only builds the sparse mask/Voronoi input tensor expected by the current data adapter.
- `voronoicnn` uses the vendored RecFNO `UNet` on normal image resolutions; tiny smoke-test grids use the local compact CNN because the official UNet downsamples too deeply for 8x8 inputs.
- `deeponet` and `pinn_sparse` prefer DeepXDE PyTorch `DeepONetCartesianProd`/`FNN` when DeepXDE optional dependencies are installed. In the default lightweight test environment, they fall back to the local API-compatible networks.
- `senseiver` and `pc_bnn` prefer their vendored official modules when their optional dependencies and problem-specific channel assumptions are satisfied. Otherwise they keep the local generic Perceiver/SVGD adapters.
- `ifno`, `pde_opt`, `var4d`, and `vivid` remain API-compatible adaptations because the vendored official scripts are tied to command-line globals, fixed datasets, or external checkpoint/data layouts rather than importable model components.

## Physics Loss Status

`baselines/common/physics.py` exposes structured physics losses:

```python
physics_losses(pred, pde_name, metadata) -> {
    "interior": tensor,
    "bc": tensor,
    "ic": tensor,
    "total": tensor,
    "residual": tensor | None,
    "mode": str,
}
```

`residual_loss(...)` is kept for backward compatibility and still returns the interior PDE residual loss. `physics_loss(...)` returns the weighted total. `pde_residual_metric`, `bc_residual_metric`, and `ic_residual_metric` record the three scalar terms separately.

Implemented equations and condition terms:

- Darcy: `-div(a grad p) - 1`, interior grid only, homogeneous Dirichlet loss on `p`, no IC term.
- Poisson: `-Delta phi - f`, interior grid only, homogeneous Dirichlet loss on `phi`, no IC term.
- Helmholtz: `(-Delta - k^2) psi - f`, interior grid only, homogeneous Dirichlet loss on `psi`, no IC term.
- Burgers: `u_t + u u_x - nu u_xx`, periodic x-boundary loss, IC loss against `initial_1d` or the first stored frame.
- Navier-Stokes nonbounded: periodic vorticity transport `omega_t + u dot grad omega - nu Delta omega - forcing`, Fourier streamfunction velocity recovery, periodic x/y boundary loss, IC loss against `omega0`.
- Reaction-Diffusion: FitzHugh-Nagumo `[u,v]` residual on `[-1,1]^2`, homogeneous Neumann boundary loss, IC loss against the segment input state.
- Shallow Water: conservative `[h,hu,hv]` mass and momentum residuals on `[-2.5,2.5]^2`, zero-order Neumann boundary loss, IC loss against the input state.
- Heat: `u_t - alpha Delta u`, periodic boundary loss by default with homogeneous Neumann available through metadata `bc="neumann"`, IC loss against `u0`.
- Wave: first-order system residual `[u_t-v, v_t-c^2 Delta u]`, periodic boundary loss by default, IC loss against `[u0,v0]`.
- Advection-Diffusion: `u_t + b_x u_x + b_y u_y - kappa Delta u`, periodic boundary loss and IC loss against `u0`.
- Steady Heat Conduction: nonlinear residual `-div(lambda(u) grad u)-f` with `lambda(u)=1+0.05(u-298)`, bottom Dirichlet `u_D` plus zero-Neumann top/left/right boundary loss.

Time-dependent residuals use `mode="full_trajectory"` when `pred` is a trajectory tensor such as `[N,C,T,H,W]`. If a task only predicts the final state, the code builds a two-frame segment from the input/background state and the prediction and returns `mode="two_level"`. This is useful for optimization and diagnostics, but it is not equivalent to a full multi-step dynamics residual.

PINN-Sparse, PC-BNN, PDE-Opt, 4D-Var, and VIVID use structured physics loss by default with `physics_loss_mode: "total"`. Supported weights are `lambda_int` or backward-compatible `lambda_pde`, plus `lambda_bc` and `lambda_ic`. If only `lambda_pde` is present, all three physics terms use that same scale.

For unknown PDEs or genuinely missing source/solution metadata, residual calls return `nan` through metric wrappers with an explicit warning instead of silently computing an invalid quantity.

## Running Smoke Tests

Use the requested conda environment:

```bash
conda run -n FM4PDEbaseline pytest
conda run -n FM4PDEbaseline bash scripts/baselines/smoke_all.sh
```

The smoke script uses deterministic synthetic data by default so it is fast and does not load multi-GB MAT/HDF5 files.

Single command examples:

```bash
conda run -n FM4PDEbaseline python -m baselines.run --baseline fno --pde darcy --task forward --dry-run --synthetic-data
conda run -n FM4PDEbaseline python -m baselines.run --baseline recfno --pde poisson --task sparse_solution --num-sensors 500 --dry-run --synthetic-data
conda run -n FM4PDEbaseline python -m baselines.run --baseline senseiver --pde shallow_water --task sparse_solution --num-sensors 500 --dry-run --synthetic-data
```

To dry-run against real small test files, omit `--synthetic-data` and use `--prefer-test`:

```bash
conda run -n FM4PDEbaseline python -m baselines.run --baseline fno --pde darcy --task forward --data-root /home/tat512/C01Python/PDEdata --prefer-test --dry-run
```

## Running Main Experiments

Scripts:

- `scripts/baselines/run_sparse_main.sh`
- `scripts/baselines/run_full_operator.sh`
- `scripts/baselines/run_physics_opt.sh`
- `scripts/baselines/run_time_dependent_da.sh`

Environment variables can override defaults:

```bash
DATA_ROOT=/home/tat512/C01Python/PDEdata TRAIN_SIZE=1024 BATCH_SIZE=8 EPOCHS=5 \
conda run -n FM4PDEbaseline bash scripts/baselines/run_sparse_main.sh
```

Results are written to JSONL and CSV with fields:

`pde`, `task`, `baseline`, `seed`, `train_size`, `num_sensors`, `sensor_mode`, `noise_level`, `relative_l2_input_or_coeff`, `relative_l2_solution`, `obs_mse`, `pde_residual`, `bc_residual`, `ic_residual`, `physics_loss`, `residual_mode`, `train_time`, `inference_time`, `inference_optimization_time`, `num_params`, `config_path`, `checkpoint_path`, `commit_hash`, `mask_id`.

## Adding New PDE Adapters

1. Add a loader function in `baselines/common/data_adapter.py` returning:
   `{"full_tensor": tensor, "channel_names": [...], "metadata": {...}}`.
2. Register it in `_register_defaults()` with `PDESpec`.
3. Preserve raw trajectory dimensions in `full_tensor` instead of collapsing time.
4. Define `input_indices`, `target_indices`, and task split logic in `_split_task()` if the default is insufficient.
5. Add a tiny native-format fixture in `tests/conftest.py`.
6. Update this README's channel table and `BASELINE_SOURCES.md` if a method has PDE-specific limitations.
