# PC-BNN adaptation contract

The FM4PDE method is reported as **PC-BNN-adapted**, not as an official-native
reproduction. The official stenotic-flow implementation maps coordinates to
the incompressible-flow fields `(u, v, p)`. Static FM4PDE sparse forward and
inverse tasks instead require a joint posterior over `(a, u)`.

The posterior engine preserves the method-defining PC-BNN mechanisms:

- three-hidden-layer Swish particles;
- Gaussian sparse-observation likelihood with learned noise precision;
- Student-t marginal weight prior;
- Gamma prior over noise precision;
- PDE-residual likelihood;
- SVGD particle updates and posterior mean/standard deviation.

Training follows the official optimizer structure: particles are independently
Kaiming-initialized, noise precisions are sampled from the Gamma prior, the
SVGD direction is written back as each particle's gradient, and independent
Adam optimizers update network weights and noise precision with separate
learning rates. The default official-aligned configuration uses five particles,
2000 steps, `lr=1e-3`, `lr_noise=1e-5`, and equation variance `1e-4`.

The task seam is implemented by sparse-forward and sparse-inverse Adapters.
They select the observed field, returned field, and PDE residual semantics.
Darcy coefficients use a positive `softplus(a) + epsilon` parameterization.
Navier-Stokes is intentionally excluded because endpoint-only adaptation would
require a space-time posterior and is outside the experiment specification.

Each evaluated sample is stored using schema
`fm4pde-evaluation-sample-v1`. The run summary records the artifact directory,
manifest, count, and PDF path. `torch.load(..., weights_only=True)` is used by
the public loader, and the manifest records a SHA-256 digest for each sample.
