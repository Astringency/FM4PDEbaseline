# Native Capability Matrix

`native` and `official_adapter` entries are eligible for main tables only when the implementation mode also satisfies the required official/canonical backend. `adapted` entries are supplement-only. `unsupported` entries are skipped in paper mode.

| Baseline | full_forward | full_inverse | sparse_reconstruction | sparse_inverse | time_varying_DA | official_status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FNO | native | adapted | unsupported | unsupported | unsupported | official required | Use neuraloperator/vendored official FNO for paper; local compact FNO is smoke/adapted only. |
| DeepONet | native | adapted | adapted | unsupported | unsupported | official required | Sensor-branch DeepONet is supplement-only. |
| iFNO | native | native | unsupported | unsupported | unsupported | official required | Vendored scripts are not safely importable yet; official paper mode skips/fails instead of using simplified local coupling. |
| RecFNO | unsupported | unsupported | official_adapter | adapted | unsupported | official required | Native for mask/Voronoi sparse field reconstruction. |
| Senseiver | unsupported | unsupported | official_adapter | adapted | official_adapter | official required | Time-varying DA requires trajectory observations and query coordinates. |
| VoronoiCNN | unsupported | unsupported | official_adapter | adapted | unsupported | official architecture allowed | Main path is a PyTorch reimplementation of the published Voronoi-CNN Conv2D stack; RecFNO UNet surrogate is supplement-only. |
| PINN-Sparse | unsupported | unsupported | official_adapter | official_adapter for static PDEs | unsupported | DeepXDE architecture preferred; local PDE objective disclosed | Static sparse inverse enabled for Poisson, Helmholtz, Darcy, and steady heat conduction. |
| PC-BNN | unsupported | unsupported | adapted by default | unsupported | unsupported | official assumptions required | Generic SVGD particles are supplement-only unless official channel/PDE assumptions match. |
| PDE-Opt | unsupported | unsupported | native | native for static PDEs | unsupported | canonical_math | No official code claim; canonical PDE-constrained optimization. |
| 4D-Var | unsupported | unsupported | adapted for endpoint/time-dependent surrogate | unsupported | native | canonical_math | Main only for full trajectory or multi-time observations. |
| VIVID | unsupported | unsupported | adapted for VIVID-style surrogate | unsupported | conditional official_adapter | official VIVID/invobs import required | Current vendored snapshots lack a stable importable adapter, so VIVID-style is supplement and native VIVID skips under `official_or_skip`. |

Static sparse inverse PDEs: `poisson`, `helmholtz`, `darcy`, `steady_heat_conduction`.

Time-varying DA PDEs: `nsnonbounded`, `burger`, `reaction_diffusion`, `shallow_water`.
