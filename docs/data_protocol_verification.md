# Data protocol verification

`configs/experiments/experiment_plan_v2.yaml`,
`outputs/experiment_plan_v2/`, and `outputs/experiment_plan_v2_summary.xlsx`
are historical audit evidence. They are not rewritten. Corrected runs use the separate
`experiment_plan_v2_corrected` namespace defined in
`configs/experiments/experiment_plan_v2_corrected.yaml`.

The corrected protocol makes sparse layouts explicit with
`random_per_sample`, removes the undefined Burger `sparse_inverse` rows, and
removes PINN/PDEOpt `sparse_solution` rows because the unified sensor-only task
does not expose the hidden full trajectory to any method. Burger full inverse
uses the separate `u(T) -> u(0)` contract. Burger `sparse_solution` uses a
total budget over the full `T x X` trajectory. Burger `sparse_forward` is
excluded because its 1-D `u0(x)` source cannot share the 500-point 2-D sparse
source protocol without duplicating observations across time.

Protocol v3 additionally makes the document's task contract executable:
Poisson/Helmholtz/Darcy/Navier--Stokes sparse reconstruction is
`[O_a,O_u] -> [a,u]`, Navier--Stokes full/sparse operator tasks use `u(T)`
rather than flattened future trajectories, and Burgers has both 500 scattered
space-time observations and five complete time slices. The five-slice layouts
are sampled per sample and refreshed per training epoch.

Before a smoke run, perform a bounded read-only check (32 samples per split by
default):

```bash
python scripts/verify_data_protocol.py \
  --config configs/experiments/experiment_plan_v2_corrected.yaml \
  --data-root /path/to/PDEdata
```

For paper validation, explicitly request complete coverage and exact hashes:

```bash
python scripts/verify_data_protocol.py \
  --config configs/experiments/experiment_plan_v2_corrected.yaml \
  --data-root /path/to/PDEdata \
  --full
```

Verification is config-specific: the report records the exact SHA-256 of the
experiment YAML. A report generated for one config must not be reused for a
different config, even when their PDE lists overlap. The multi-matrix launcher
therefore expects one report per config under `DATA_MANIFEST_DIR`:

```text
outputs/data_protocol/<config-name>/full/data_protocol_report.json
```

After running `--full` separately for each experiment config, build each matrix
with the unified generator:

```bash
python scripts/build_experiment_matrix.py \
  --config configs/experiments/main_results.yaml \
  --output-root outputs/baselines_large \
  --data-manifest outputs/data_protocol/main_results/full/data_protocol_report.json
```

The same exact report is used for two-GPU execution; GPU selection belongs to
the runner rather than the matrix generator:

```bash
python scripts/run_experiments.py \
  outputs/baselines_large/matrices/main_results.jsonl \
  --data-root /path/to/PDEdata --gpus 0,1 --jobs-per-gpu 2
```

The verifier mirrors the runner's validation policy: `train_size` is the total
training/validation budget, so `50000` with `val_size=1000` always fits on
49000 samples. It prefers an independent validation file and otherwise uses
the final 1000 samples of that budget as a deterministic validation tail. It reports
missing or duplicate global IDs, cross-split ID overlap, and (when hashing is
enabled) cross-split field/content duplicates. `--full` fails if any configured
split is only partially scanned.

Outputs are written below
`outputs/data_protocol/<experiment-name>/<full|bounded_hashes|bounded_ids>/`,
so a later bounded check cannot overwrite full-validation evidence:

- `data_protocol_report.json`: configuration fingerprint, split decisions,
  overlap evidence, and pass/fail status.
- `data_protocol_splits.csv`: one provenance row per PDE/split.
- `sample_manifest.jsonl` and `sample_manifest.csv`: sample IDs and optional
  SHA-256 hashes.
- `cache/`: reusable per-split manifests. A cache entry is accepted only while
  its scan identity and every source file's size, nanosecond mtime, and
  nanosecond ctime match.

The field hash covers the canonical `full_tensor` sample. The complete-content
hash also incorporates sample-indexed physical parameters and metadata (for
example Burger's initial condition). No synthetic fallback is permitted.

After a full data-protocol pass, generate the corrected unified/adapted matrix
in its own output root:

```bash
python scripts/build_experiment_matrix.py \
  --config configs/experiments/experiment_plan_v2_corrected.yaml \
  --output-root outputs/experiment_plan_v2_corrected \
  --matrix-name experiment_plan_v2_corrected \
  --comparison-track unified_adapted \
  --data-manifest outputs/data_protocol/experiment_plan_v2_corrected/full/data_protocol_report.json
```

The corrected design runs seed `[1]` once per experiment, producing 76 active
rows. Inspect and run it with the unified runner:

```bash
python scripts/run_experiments.py \
  outputs/experiment_plan_v2_corrected/matrices/experiment_plan_v2_corrected.jsonl \
  --data-root /path/to/PDEdata --gpus 0,1 --dry-run
python scripts/run_experiments.py \
  outputs/experiment_plan_v2_corrected/matrices/experiment_plan_v2_corrected.jsonl \
  --data-root /path/to/PDEdata --gpus 0,1
```

It fails closed if given the historical `experiment_plan_v2` matrix or a row
whose artifacts live in that namespace. It never moves historical artifacts;
those remain immutable audit evidence.

Select a matrix row by zero-based index with `scripts/run_experiments.py`.
Every row is bound to the full config SHA-256, the full data-protocol report
SHA-256, and the exact repository revision; changing code, config, or report
requires regenerating the matrix. Formal paper runs fail before data loading if
the full-report SHA is missing. A row is complete only after its schema-v2
`summary.json` passes the same identity checks.

```bash
python scripts/run_experiments.py \
  outputs/experiment_plan_v2_corrected/matrices/experiment_plan_v2_corrected.jsonl \
  --data-root /path/to/PDEdata --indices 0
```

The `official_native` track is deliberately separate and fail-closed. At this
revision no audited method has an end-to-end upstream-native data, training,
and evaluation recipe, so this command produces skipped evidence rather than
mislabeling component reuse as an official reproduction:

```bash
python scripts/build_experiment_matrix.py \
  --config configs/experiments/experiment_plan_v2_corrected.yaml \
  --output-root outputs/experiment_plan_v2_official_native \
  --matrix-name experiment_plan_v2_official_native \
  --comparison-track official_native \
  --data-manifest outputs/data_protocol/experiment_plan_v2_corrected/full/data_protocol_report.json
```

Finally, export only a complete, single-provenance cohort. Missing, legacy,
eval-only-mismatched, or mixed-revision rows block publication and are written
to a quarantine workbook instead:

```bash
python scripts/collect_results.py \
  outputs/experiment_plan_v2_corrected/matrices/experiment_plan_v2_corrected.jsonl \
  --output-dir outputs/experiment_plan_v2_corrected/aggregate
```
