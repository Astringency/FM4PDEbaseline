# Baseline Sources and Enabled Scope

This file is the source-key reference used by `baselines/capabilities.py`. Do not describe a local/adapted fallback as official.

## FNO

- `source_key`: `fno`
- Paper: Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021.
- Official code/components: `https://github.com/zongyi-li/fourier_neural_operator`; maintained `https://github.com/neuraloperator/neuraloperator`.
- Vendored path: `offical/neuraloperator`.
- Wrapper: `baselines/methods/fno.py`.
- Standard capability: full-grid supervised forward operator learning.
- Enabled main tasks: `forward` with official `neuraloperator` FNO or vendored official FNO component.
- Supplement only: supervised `inverse` as FNO inverse-operator adaptation.
- Unsupported main tasks: sparse reconstruction and sparse inverse. A masked-grid FNO can be reported only as explicitly adapted supplement.

## DeepONet

- `source_key`: `deeponet`
- Paper: Lu et al., "Learning nonlinear operators via DeepONet...", Nature Machine Intelligence 2021.
- Official code/components: DeepONet/DeepXDE, `https://github.com/lululxvi/deepxde`.
- Vendored path: `offical/deepxde`.
- Wrapper: `baselines/methods/deeponet.py`.
- Standard capability: supervised operator learning.
- Enabled main tasks: full `forward` using DeepXDE `DeepONetCartesianProd` when importable.
- Supplement only: supervised full `inverse`; sensor values as branch input for sparse reconstruction.
- Unsupported main tasks: sparse inverse and any local MLP branch/trunk result in official paper mode.

## iFNO

- `source_key`: `ifno`
- Paper: Long et al., "Invertible Fourier Neural Operators for Tackling Both Forward and Inverse Problems", AISTATS 2025.
- Official code: `https://github.com/BayesianAIGroup/iFNO`.
- Vendored path: `offical/iFNO`.
- Wrapper: `baselines/methods/ifno.py`.
- Standard capability: full forward and full inverse operator learning.
- Enabled main tasks: `forward` and `inverse` only when a reliable official/importable iFNO adapter is available.
- Current status: vendored scripts parse command-line globals at import time, so paper official mode fails/skips rather than using the simplified local coupling block.
- Unsupported main tasks: sparse reconstruction and sparse inverse.

## RecFNO

- `source_key`: `recfno`
- Paper: Zhao et al., "RecFNO: a resolution-invariant flow and heat field reconstruction method from sparse observations via Fourier neural operator", 2023.
- Official code: `https://github.com/zhaoxiaoyu1995/recfno`.
- Vendored path: `offical/RecFNO`.
- Wrapper: `baselines/methods/recfno.py`.
- Standard capability: sparse sensor global field reconstruction using mask/Voronoi embeddings and FNO.
- Enabled main tasks: `sparse_solution`/`sparse_reconstruction` with vendored `VoronoiFNO2d`.
- Supplement only: target-change sparse inverse adaptation.
- Unsupported main tasks: full forward and full inverse.

## Senseiver

- `source_key`: `senseiver`
- Paper: Santos et al., "Development of the Senseiver for efficient field reconstruction from sparse observations", Nature Machine Intelligence 2023.
- Official code: `https://github.com/OrchardLANL/Senseiver`.
- Vendored path: `offical/Senseiver`.
- Wrapper: `baselines/methods/senseiver.py`.
- Standard capability: sparse/irregular sensor reconstruction with query coordinates; time-varying sensors when trajectory observations and query coordinates are present.
- Enabled main tasks: sparse reconstruction; time-varying DA reconstruction in full-trajectory settings.
- Supplement only: sparse coefficient/source reconstruction as supervised adaptation.
- Unsupported main tasks: full forward and full inverse.

## VoronoiCNN

- `source_key`: `voronoicnn`
- Paper: Fukami et al., "Global field reconstruction from sparse sensors with Voronoi tessellation-assisted deep learning", Nature Machine Intelligence 2021.
- Official code: `https://github.com/kfukami/Voronoi-CNN`.
- Vendored path: `offical/Voronoi-CNN`.
- Wrapper: `baselines/methods/voronoicnn.py`.
- Standard capability: sparse sensor global field reconstruction from Voronoi-filled fields using CNN/UNet.
- Enabled main tasks: sparse reconstruction.
- Implementation note: if original Keras scripts cannot be imported directly, PyTorch architecture reuse is labeled `official_architecture_reimplementation`, not official binary/code reuse.
- Supplement only: supervised sparse inverse target-change adaptation.
- Unsupported main tasks: full forward and full inverse.

## PINN-Sparse

- `source_key`: `pinn_sparse`
- Paper: Raissi et al., "Physics-informed neural networks", JCP 2019; DeepXDE reference implementation.
- Official code/components: `https://github.com/lululxvi/deepxde`.
- Vendored path: `offical/deepxde`.
- Wrapper: `baselines/methods/pinn_sparse.py`.
- Standard capability: per-instance PDE fitting from sparse/noisy observations.
- Enabled main tasks: sparse state reconstruction; static sparse inverse for `poisson`, `helmholtz`, `darcy`, and `steady_heat_conduction`.
- Implementation note: DeepXDE FNN is the preferred architecture. The inverse objective and residuals are local canonical PDE objectives and are disclosed as "PINN-style official architecture + local PDE objective".
- Unsupported main tasks: supervised full operator learning; time-dependent sparse inverse until unknown initial/parameter residuals are explicit.

## PC-BNN

- `source_key`: `pc_bnn`
- Paper: Sun and Wang, "Physics-Constrained Bayesian Neural Network for Fluid Flow Reconstruction with Sparse and Noisy Data".
- Official code: `https://github.com/Jianxun-Wang/Physics-constrained-Bayesian-deep-learning`.
- Vendored path: `offical/PC-BNN`.
- Wrapper: `baselines/methods/pc_bnn.py`.
- Standard capability: physics-constrained Bayesian neural network for sparse/noisy flow reconstruction under the official channel/PDE assumptions.
- Enabled main tasks: none by default for scalar FM4PDE main tables unless official target channel assumptions are matched.
- Supplement only: generic local SVGD particles with local residuals.
- Unsupported main tasks: sparse inverse without explicit parameter posterior objective; supervised full operators.

## PDE-Opt

- `source_key`: `pde_opt`
- Paper/source: canonical PDE-constrained optimization/data assimilation baseline; no single official codebase is claimed.
- Wrapper: `baselines/methods/pde_opt.py`.
- Standard capability: per-instance canonical PDE-constrained optimization.
- Enabled main tasks: sparse reconstruction; static sparse inverse for `poisson`, `helmholtz`, `darcy`, and `steady_heat_conduction` by jointly optimizing solution and unknown coefficient/source.
- Unsupported main tasks: supervised full operator learning; time-dependent sparse inverse without explicit trajectory/parameter objective.

## 4D-Var

- `source_key`: `var4d`
- Paper/source: canonical 4D-Var data assimilation.
- Wrapper: `baselines/methods/var4d.py`.
- Standard capability: time-dependent data assimilation over a trajectory with observation, background, and model-dynamics residual.
- Enabled main tasks: time-varying sparse DA for `nsnonbounded`, `burger`, `reaction_diffusion`, and `shallow_water` when full trajectory or multi-time observations are loaded.
- Supplement only: endpoint/two-level surrogate or static state surrogate.
- Unsupported main tasks: static PDE main tables and supervised full operators.

## VIVID

- `source_key`: `vivid`
- Paper/source: deep variational data assimilation with learned inverse observation operators.
- Official code: `https://github.com/DL-WG/VIVID`; related inverse-observation repository `https://github.com/googleinterns/invobs-data-assimilation`.
- Vendored paths: `offical/VIVID`, `offical/invobs-data-assimilation`.
- Wrapper: `baselines/methods/vivid.py`.
- Standard capability: learned inverse-observation initialization plus variational refinement for sparse, unstructured, time-varying sensors.
- Enabled main tasks: time-varying DA only when a trained/loaded inverse observation operator is used and full trajectory/multi-time observations are available.
- Supplement only: Voronoi initialization plus variational refinement without inverse-operator training, labeled `VIVID-style`.
- Unsupported main tasks: static PDE main tables, endpoint-only main DA, and supervised full operators.
