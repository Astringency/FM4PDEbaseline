# Official-Aligned Implementation Appendix

This appendix documents paths that are main-table eligible as `implementation_mode_effective=official_aligned`. They are import-safe reimplementations aligned to official method definitions; they are not direct official code execution.

## iFNO

- Official files consulted: `offical/iFNO/darcy_curve.py`, `offical/iFNO/ns.py`, `offical/iFNO/utils.py`, `offical/iFNO/vanilla_vae.py`, `offical/iFNO/run.py`.
- Reimplemented components: coordinate-augmented p1/p2 lifts, q1/q2 pointwise projections, multiplicative Softplus FNO coupling blocks, bidirectional forward/inverse maps, reconstruction penalties, supervised forward/inverse losses, and cycle consistency.
- Omitted components: script-global training harnesses, VAE posterior inference, checkpoint-specific preprocessing, and command-line side effects.
- Input/output adapter: FM4PDE `PDEBatch.input_fields` and `target_fields` are mapped to iFNO forward/inverse tensors; sparse tasks remain unsupported.
- Training loss: supervised forward and inverse MSE plus cycle consistency and reconstruction penalties, with optional train-set normalization.

## VIVID

- Official files consulted: `offical/VIVID/VIVID.py`, `offical/VIVID/VCNN_training.py`, `offical/VIVID/VIVID-POD.py`, `offical/VIVID/voronoi_preprocessing.py`, `offical/VIVID/shallow_water.py`, `offical/invobs-data-assimilation/run_train_inverse_observations.py`, `run_data_assimilation.py`, `da_methods.py`, `ml_methods.py`, and `kolmogorov_ml.py`.
- Inverse observation operator: `baselines/methods/vivid_official_aligned.py` implements sparse/Voronoi observation ingestion and time-space convolutional inverse reconstruction.
- Variational refinement: `baselines/methods/vivid.py` refines a trajectory with observation, inverse/background, and structured dynamics losses.
- Trajectory handling: for `nsnonbounded`, learned future-only trajectories `[B,1,T_future,H,W]` are inserted into the optimized `[initial,future]` state before refinement. Physics loss prepends the true/background initial frame before evaluating full-trajectory NS residuals.
- Omitted components: direct TensorFlow/Keras/Flax execution, POD checkpoints, dataset-specific official config runners, and official checkpoint reuse.

## PC-BNN

- Official files consulted: `offical/PC-BNN/code/FCN.py`, `BayesNN.py`, `SVGD.py`, `cases.py`, and `mainsolve.py`.
- Matched assumption: main-table PC-BNN is enabled only for 2D three-channel shallow-water sparse reconstruction, matching the official flow-style coordinate-to-multiple-fields setup.
- Scalar PDE scope: scalar Darcy/Poisson/Helmholtz-style fields do not match the official channel/PDE assumption and remain supplement/debug only.
- Reimplemented components: Swish coordinate MLP particles, SVGD RBF-kernel particle updates, sparse/noisy observation likelihood, predictive mean/std, and physics-constrained residual loss.
- Direct official option: `implementation_mode=official` may use the vendored official `Net` class when it imports and the PDE/channel assumptions match; `implementation_mode=official_aligned` explicitly uses the local aligned Net.

## Declaration

Rows labeled `official_aligned` must have `official_import_success=false`, `official_reimplementation_success=true`, and an adapter status naming the aligned reimplementation. Direct official rows must use `implementation_mode_effective=official` and `official_import_success=true`.
