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
- Official files consulted: `offical/iFNO/darcy_curve.py`, `offical/iFNO/ns.py`, `offical/iFNO/utils.py`, `offical/iFNO/vanilla_vae.py`, `offical/iFNO/run.py`.
- Standard capability: full forward and full inverse operator learning.
- Enabled main tasks: `forward` and `inverse` using direct official components if they become importable, or the current `official_aligned` reimplementation.
- Reimplementation scope: `baselines/methods/ifno_official_aligned.py` follows the official p1/p2 coordinate-augmented lift, q1/q2 pointwise projections, multiplicative Softplus FNO coupling blocks, shared forward/backward invertible backbone, reconstruction terms, bidirectional supervised loss, and cycle consistency. It does not claim VAE posterior inference from the official scripts.
- Supplement only: explicit `implementation_mode: adapted` uses a simplified local/debug coupling path labeled `local_debug_ifno`.
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
- Standard capability: sparse sensor global field reconstruction from Voronoi-filled fields using the published CNN stack.
- Enabled main tasks: sparse reconstruction.
- Implementation note: the original Keras/TensorFlow scripts are not imported as a library. The main-table path uses a PyTorch reimplementation of the published seven-layer Conv2D architecture and is labeled `official_architecture_reimplementation`, not official binary/code reuse.
- Not official: using RecFNO's UNet as a VoronoiCNN surrogate is labeled `recfno_unet_as_voronoi_cnn_adaptation` and is supplement-only.
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
- Official files consulted: `offical/PC-BNN/code/FCN.py`, `offical/PC-BNN/code/BayesNN.py`, `offical/PC-BNN/code/SVGD.py`, `offical/PC-BNN/code/cases.py`, `offical/PC-BNN/code/mainsolve.py`.
- Standard capability: physics-constrained Bayesian neural network for sparse/noisy flow reconstruction under the official channel/PDE assumptions.
- Enabled main tasks: conditional `sparse_solution`/`sparse_reconstruction` for 2D three-channel shallow-water fields, labeled `official_aligned_pcbnn_reimplementation`.
- Reimplementation scope: coordinate-to-three-field Swish MLP particles, particle posterior approximation with SVGD RBF-kernel updates, observation likelihood on sparse/noisy samples, predictive mean/std, and physics-constrained residual terms. Scalar Poisson/Darcy/Helmholtz-style fields do not match the official flow/channel assumption and remain supplement-only.
- Supplement only: generic local SVGD particles with local residuals, labeled `local_generic_svgd_pcbnn` or `fallback_generic_svgd_pcbnn`.
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
- Official files consulted: `offical/VIVID/VIVID.py`, `offical/VIVID/VCNN_training.py`, `offical/VIVID/VIVID-POD.py`, `offical/VIVID/voronoi_preprocessing.py`, `offical/VIVID/shallow_water.py`, `offical/invobs-data-assimilation/run_train_inverse_observations.py`, `offical/invobs-data-assimilation/run_data_assimilation.py`, `offical/invobs-data-assimilation/da_methods.py`, `offical/invobs-data-assimilation/ml_methods.py`, `offical/invobs-data-assimilation/kolmogorov_ml.py`.
- Standard capability: learned inverse-observation initialization plus variational refinement for sparse, unstructured, time-varying sensors.
- Enabled main tasks: time-varying DA for `nsnonbounded`, `burger`, `reaction_diffusion`, and `shallow_water` when full trajectory/multi-time observations are available, using `official_aligned_vivid_invobs_reimplementation`.
- Reimplementation scope: `baselines/methods/vivid_official_aligned.py` trains a Voronoi/sparse-observation inverse operator with VIVID-style convolutional reconstruction and invobs-style time-space convolution semantics, then `baselines/methods/vivid.py` performs variational refinement against observation, inverse/background, and dynamics/PDE residual losses. It does not claim direct TensorFlow/Keras/Flax checkpoint reuse.
- Supplement only: explicit `implementation_mode: adapted` locally trains VoronoiCNN initialization plus refinement, labeled `VIVID-style`.
- Unsupported main tasks: static PDE main tables, endpoint-only main DA, and supervised full operators.
