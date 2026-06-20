# Native Capability Matrix

`native` and `official_adapter` entries are eligible for main tables only when the implementation mode also satisfies the required official/canonical backend. `implementation_mode=official` means direct official code/component import, while `official_aligned` and `official_architecture` are explicit capability-approved reimplementations. `official_or_skip` may fall back to official-aligned only when the row's capability allows it. `adapted` entries are supplement-only. `unsupported` entries are skipped in paper mode.

| Baseline | full_forward | full_inverse | sparse_reconstruction | sparse_inverse | time_varying_DA | official_status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FNO | native | adapted | unsupported | unsupported | unsupported | official required | Use neuraloperator/vendored official FNO for paper; local compact FNO is smoke/adapted only. |
| DeepONet | native | adapted | adapted | unsupported | unsupported | official required | Sensor-branch DeepONet is supplement-only. |
| iFNO | native | native | unsupported | unsupported | unsupported | official_aligned available | Full forward/full inverse use the import-safe official-aligned invertible FNO architecture/objective reimplementation when direct official scripts are not importable. |
| RecFNO | unsupported | unsupported | official_adapter | adapted | unsupported | official required | Native for mask/Voronoi sparse field reconstruction. |
| Senseiver | unsupported | unsupported | official_adapter | adapted | official_adapter | official required | Time-varying DA requires trajectory observations and query coordinates. |
| VoronoiCNN | unsupported | unsupported | official_adapter | adapted | unsupported | official architecture allowed | Main path is a PyTorch reimplementation of the published Voronoi-CNN Conv2D stack; RecFNO UNet surrogate is supplement-only. |
| PINN-Sparse | unsupported | unsupported | official_adapter | official_adapter for static PDEs | unsupported | DeepXDE architecture preferred; local PDE objective disclosed | Static sparse inverse enabled for Poisson, Helmholtz, Darcy, and steady heat conduction. |
| PC-BNN | unsupported | unsupported | conditional official_adapter | unsupported | unsupported | official or official_aligned for matched fields | Shallow-water three-channel 2D sparse reconstruction matches the official flow-like setting; strict official may use the vendored official Net, while explicit aligned uses the local SVGD/Net reimplementation. Scalar generic fields remain supplement-only. |
| PDE-Opt | unsupported | unsupported | native | native for static PDEs | unsupported | canonical_math | No official code claim; canonical PDE-constrained optimization. |
| 4D-Var | unsupported | unsupported | adapted for endpoint/time-dependent surrogate | unsupported | native | canonical_math | Main only for full trajectory or multi-time observations. |
| VIVID | unsupported | unsupported | adapted for VIVID-style surrogate | unsupported | official_adapter | official_aligned available | Time-varying DA uses the official-aligned inverse-observation plus variational refinement path; explicit VIVID-style adapted runs remain supplement-only. |

Static sparse inverse PDEs: `poisson`, `helmholtz`, `darcy`, `steady_heat_conduction`.

Time-varying DA PDEs: `nsnonbounded`, `burger`, `reaction_diffusion`, `shallow_water`.
