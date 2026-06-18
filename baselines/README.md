# FM4PDE External Baseline Framework

This directory contains the reviewer-facing external baseline layer for FM4PDE. The framework is intentionally conservative: paper-mode results must use official code/components when a baseline claims official implementation status, and local compact implementations are never allowed to masquerade as native paper baselines.

## Capability Status

Every baseline/PDE/task/sensor-mode combination is resolved through `baselines.capabilities.resolve_capability(...)` and recorded in every result row.

| Status | Meaning | Main table |
| --- | --- | --- |
| `native` | The task is inside the method's standard published capability. | Eligible when implementation is official/canonical as required. |
| `official_adapter` | Official model components are used through this repository's data adapter, or canonical PINN architecture is paired with a local PDE objective. | Eligible when disclosed and not a fallback. |
| `adapted` | Local adaptation, target change, endpoint surrogate, style reimplementation, or debug/supplement method. | Supplement only. |
| `unsupported` | Outside the baseline's standard capability or missing required trajectory/objective. | Skipped in paper mode. |

`implementation_required` is one of:

- `official`: paper mode must use official code or official model components.
- `canonical_math`: no single official repository is claimed; the method is a canonical mathematical baseline.
- `adapted_allowed`: only smoke/debug or explicitly labeled supplement runs.
- `unsupported`: paper mode skips the combination.

Paper rows include:

`implementation_mode_requested`, `implementation_mode_effective`, `implementation_source`, `official_repo`, `official_commit_or_version`, `official_import_path`, `official_import_success`, `adapter_status`, `capability_status`, `implementation_required`, `unsupported_reason`, and `paper_table_eligible`.

## Implementation Modes

Paper configs default to:

```yaml
method:
  implementation_mode: official_or_skip
```

This means an official-capability baseline must import official code/components or be skipped without stopping the full paper script. To run local debug implementations, use smoke/debug mode or explicitly set `implementation_mode: adapted`; those rows are marked supplement-only.

`implementation_mode: official` is strict: if official code/components cannot be imported, the run raises and the command fails. `implementation_mode: official_or_skip` is the paper-script default: predictable official dependency/import failures are written to `skipped_combinations.jsonl` with implementation and capability metadata, and the shell script continues to later combinations. `implementation_mode: adapted` permits local implementations only for smoke/debug or supplement rows; it never sets `paper_table_eligible=true`.

`implementation_mode: official_architecture` is reserved for methods whose capability explicitly permits architecture reimplementation. At present this is used for VoronoiCNN's disclosed PyTorch reimplementation of the published Keras Conv2D stack. It is not a claim of official binary/code reuse.

Legacy `official_backend` is still accepted for compatibility, but new paper configs should use `implementation_mode`.

## Formal Native Commands

```bash
DATA_ROOT=/home/tat512/C01Python/PDEdata DEVICE=cuda:0 \
  bash scripts/baselines/run_paper_native_full_operator.sh

DATA_ROOT=/home/tat512/C01Python/PDEdata DEVICE=cuda:0 \
  bash scripts/baselines/run_paper_native_sparse_reconstruction.sh

DATA_ROOT=/home/tat512/C01Python/PDEdata DEVICE=cuda:0 \
  bash scripts/baselines/run_paper_static_sparse_inverse.sh

DATA_ROOT=/home/tat512/C01Python/PDEdata DEVICE=cuda:0 \
  bash scripts/baselines/run_paper_time_varying_da.sh

DATA_ROOT=/home/tat512/C01Python/PDEdata DEVICE=cuda:0 \
  bash scripts/baselines/run_paper_all_native.sh
```

`run_paper_all.sh` is retained as a compatibility entry point and delegates to `run_paper_all_native.sh`. Older per-task scripts remain available for supplement/debug runs, but adapted rows are not included in the main aggregate table. VIVID-style time-varying DA is available through `run_paper_time_varying_da_supplement.sh`; it is not part of the native time-varying DA script unless a real official VIVID/invobs adapter is added.

## Aggregation

Use:

```bash
OUT=outputs/baselines/paper bash scripts/baselines/aggregate_paper_results.sh
```

The aggregator writes:

- `summary_main.csv/json`: rows that pass both `paper_table_eligible=true` and aggregator rechecks for allowed implementation mode, no fallback, clean adapter status, and official import success when official code is required.
- `summary_supplement.csv/json`: adapted, surrogate, local, style, fallback, or suspicious rows. Rows that claimed main eligibility but fail the recheck are downgraded with `aggregation_warning`.
- `skipped_combinations.csv/json`: skipped paper combinations.
- `baseline_capability_matrix.csv/json`: full registry dump.
- `latex_table.csv/tex`: generated only from the main summary.

## Matrix Dump

```bash
python -m baselines.experiment_matrix --dump-matrix --output outputs/baselines/capability_matrix
```

This is the source for appendix capability tables and skip auditing.

## Native Scope Summary

- FNO: native full-grid supervised forward operator learning. Supervised inverse is adapted supplement; sparse tasks are unsupported unless explicitly named as masked-FNO adaptation.
- DeepONet: native supervised full forward operator learning. Full inverse and sensor-branch reconstruction are adapted supplement.
- iFNO: native full forward and full inverse operator learning only. Sparse reconstruction/inverse is unsupported.
- RecFNO: native sparse-sensor global field reconstruction with mask/Voronoi embedding. Full operators and sparse inverse are not native.
- Senseiver: native sparse/irregular reconstruction and time-varying sensor reconstruction when trajectory observations are present. Full operators and sparse inverse are not native.
- VoronoiCNN: native sparse reconstruction through Voronoi tessellation plus CNN. PyTorch reimplementation of the published Voronoi-CNN Conv2D stack is labeled `official_architecture_reimplementation`. RecFNO UNet used as a VoronoiCNN surrogate is an adaptation and cannot enter the main table.
- PINN-Sparse: native per-instance sparse observation fitting. Static sparse inverse is enabled for `poisson`, `helmholtz`, `darcy`, and `steady_heat_conduction` through local PDE objectives.
- PC-BNN: official assumptions are narrow; generic SVGD particles are adapted supplement unless the target PDE/channel assumptions match.
- PDE-Opt: canonical per-instance PDE-constrained optimization; supports sparse reconstruction and static sparse inverse.
- 4D-Var: main-table only for time-varying DA with full trajectory or multi-time observations. Endpoint-only variants are surrogate supplement.
- VIVID: main-table only for time-varying DA with an actually imported/used official VIVID or invobs inverse-observation component plus variational refinement. Locally trained Voronoi initialization plus refinement is `VIVID-style` supplement, even when `train_inverse_operator=true`.

Simplified iFNO coupling blocks, compact/local FNO, VIVID-style refinement, and RecFNO-UNet-as-VoronoiCNN are never main-table baselines.

## Data Interface Recording

Each run records raw/native input shape, target shape, observation and predicted field names, scalar PDE parameter availability, whether scalar parameters are part of the model input, whether full trajectory data is loaded, and the effective sensor mode.

Supervised full-operator scripts materialize future-PDE scalar parameters by default unless `SCALAR_PARAM_MODE` is explicitly set. Per-instance physics methods may read scalar parameters from metadata and record that fact.

## Smoke Tests

Smoke/debug runs can use local adapted fallbacks:

```bash
bash scripts/baselines/smoke_all.sh
python -m pytest -q
```

In this environment, invoke tests with `python -m pytest` so the active conda Python is used.
