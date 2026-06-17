# Baseline Sources

Baseline: FNO
Paper: Fourier Neural Operator for Parametric Partial Differential Equations, Li et al., ICLR 2021.
Official code: https://github.com/zongyi-li/fourier_neural_operator and maintained neuraloperator library https://github.com/neuraloperator/neuraloperator
Local wrapper: `baselines/methods/fno.py`
Adaptation notes: Uses a compact PyTorch FNO2d implementation in `baselines/methods/shared.py` to avoid imposing the full neuraloperator dependency chain in smoke tests. Existing local reference code also exists under `neuraloperator/` and FM4PDE's `data/DataGen/time_dependent/DataGen/pdebench/models/fno/`. The wrapper supports full-grid and sparse masked-grid inputs through the shared adapter.
License: See upstream repositories.
Status: implemented local baseline.

Baseline: DeepONet
Paper: Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators, Lu et al., Nature Machine Intelligence 2021.
Official code: https://github.com/lululxvi/deeponet and DeepXDE https://github.com/lululxvi/deepxde
Local wrapper: `baselines/methods/deeponet.py`
Adaptation notes: Implements a branch/trunk DeepONet in PyTorch. Branch accepts either flattened full input fields or sparse sensor values; trunk uses dense query coordinates from the shared adapter.
License: Upstream DeepONet is CC BY-NC-SA 4.0; DeepXDE license is in local `deepxde/`.
Status: implemented local baseline.

Baseline: iFNO
Paper: Invertible Fourier Neural Operators for Tackling Both Forward and Inverse Problems, Long et al., AISTATS 2025.
Official code: https://github.com/BayesianAIGroup/iFNO
Local wrapper: `baselines/methods/ifno.py`; source snapshot also exists in local `iFNO/`.
Adaptation notes: Uses invertible coupling/Fourier blocks, a bidirectional forward/inverse objective, and cycle consistency. Sparse inverse uses an explicit FNO fallback because sparse observations are not bijective. The optional VAE/posterior branch from some iFNO experiments is not part of this unified baseline.
License: See upstream repository.
Status: implemented local iFNO core.

Baseline: RecFNO
Paper: RecFNO: a resolution-invariant flow and heat field reconstruction method from sparse observations via Fourier neural operator, Zhao et al., arXiv 2023.
Official code: https://github.com/zhaoxiaoyu1995/recfno
Local wrapper: `baselines/methods/recfno.py`; source snapshot exists in local `RecFNO/`.
Adaptation notes: Implements mask and Voronoi embeddings with Fourier layers. MLP embedding remains available as an extension point, while the two published sparse-grid embeddings required for the JMLR comparison are enabled.
License: See upstream repository.
Status: implemented local baseline with two embeddings.

Baseline: Senseiver
Paper: Development of the Senseiver for efficient field reconstruction from sparse observations, Santos et al., Nature Machine Intelligence 2023.
Official code: https://github.com/OrchardLANL/Senseiver
Local wrapper: `baselines/methods/senseiver.py`; source snapshot exists in local `Senseiver/`.
Adaptation notes: Implements a Perceiver-IO/Senseiver-style cross-attention model accepting `(obs_coords, obs_values)` and dense query coordinates. Fixed sensor-count batches are supported directly; padding masks can be added for variable-count batching without changing the data adapter.
License: See upstream repository.
Status: implemented local baseline.

Baseline: VoronoiCNN
Paper: Global field reconstruction from sparse sensors with Voronoi tessellation-assisted deep learning, Fukami et al., Nature Machine Intelligence 2021.
Official code: https://github.com/kfukami/Voronoi-CNN
Local wrapper: `baselines/methods/voronoicnn.py`; source snapshot exists in local `Voronoi-CNN/`.
Adaptation notes: Uses shared `baselines/common/voronoi.py` nearest-sensor Voronoi-like filling plus a PyTorch CNN. The original Keras scripts are retained as source/reference; the local wrapper uses the shared sensor/noise protocol.
License: Upstream notes academic/research use; see local `Voronoi-CNN/LICENSE`.
Status: implemented local baseline.

Baseline: PINN-Sparse
Paper: Physics-informed neural networks, Raissi et al., JCP 2019; DeepXDE reference implementation.
Official code: DeepXDE https://github.com/lululxvi/deepxde
Local wrapper: `baselines/methods/pinn_sparse.py`
Adaptation notes: Per-instance neural-field optimization over query coordinates with shared sparse observation losses and PDE residuals. Implemented residual paths include Poisson, Darcy, Helmholtz, Burgers, Navier-Stokes vorticity transport, Reaction-Diffusion, and conservative Shallow Water. Supports Adam and LBFGS.
License: See local `deepxde/`.
Status: implemented local baseline.

Baseline: PC-BNN
Paper: Physics-Constrained Bayesian Neural Network for Fluid Flow Reconstruction with Sparse and Noisy Data, Sun and Wang.
Official code: https://github.com/Jianxun-Wang/Physics-constrained-Bayesian-deep-learning
Local wrapper: `baselines/methods/pc_bnn.py`
Adaptation notes: Implements multiple Bayesian neural-field particles with an RBF-kernel SVGD update, sparse observation likelihood, PDE residual likelihood, predictive mean, and predictive standard deviation. Uses the same residual implementation as PINN-Sparse.
License: See upstream repository.
Status: implemented local SVGD particle baseline.

Baseline: PDE-Opt
Paper: Classical PDE-constrained optimization / variational inverse problem baseline.
Official code: No single official implementation; implemented directly from objective definition.
Local wrapper: `baselines/methods/pde_opt.py`
Adaptation notes: Per-instance grid-variable optimization with observation, PDE residual, and smoothness regularization terms. Static inverse/forward experiments use Poisson, Darcy, and Helmholtz directly; other PDEs reuse the shared residual registry when metadata is available.
License: Local project code.
Status: implemented local baseline.

Baseline: 4D-Var
Paper: Standard weak/strong 4D-Var data assimilation.
Official code: No single official implementation selected for this framework.
Local wrapper: `baselines/methods/var4d.py`
Adaptation notes: Implements full-space weak 4D-Var by optimizing a trajectory state with background mismatch, observation mismatch, and model-dynamics residual. If a dataset only exposes initial/final states, it optimizes the available segment and documents that this is not a full multi-time assimilation experiment.
License: Local project code.
Status: implemented local baseline.

Baseline: VIVID
Paper: Variational Data Assimilation with a Learned Inverse Observation Operator; VIVID deep variational assimilation.
Official code: https://github.com/DL-WG/VIVID and related learned inverse observation operator repository https://github.com/googleinterns/invobs-data-assimilation
Local wrapper: `baselines/methods/vivid.py`
Adaptation notes: Trains/uses VoronoiCNN as an inverse observation operator, then performs a variational refinement step with observation, background, inverse-operator mismatch, and dynamics residual terms. The POD-reduced VIVID variant is reserved as an extension; the implemented version is full-space.
License: See upstream repositories.
Status: implemented local full-space baseline.
