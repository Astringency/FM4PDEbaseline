# Data protocol verification

`experiment_plan_v2` and `outputs/experiment_plan_v2*` are historical audit
evidence. They are not rewritten. Corrected runs use the separate
`experiment_plan_v2_corrected` namespace defined in
`configs/experiments/experiment_plan_v2_corrected.yaml`.

The corrected protocol makes sparse layouts explicit with
`random_per_sample`, removes the undefined Burger `sparse_inverse` rows, and
removes PINN/PDEOpt `sparse_solution` rows because the unified sensor-only task
does not expose the hidden full trajectory to any method. Burger full inverse
uses the separate `u(T) -> u(0)` contract.

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

The verifier mirrors the runner's validation policy: it prefers an independent
validation file, otherwise reserves the configured training tail. It reports
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
python scripts/experiments/build_matrix.py \
  --config configs/experiments/experiment_plan_v2_corrected.yaml \
  --output-root outputs/experiment_plan_v2_corrected \
  --matrix-name experiment_plan_v2_corrected \
  --comparison-track unified_adapted \
  --data-manifest outputs/data_protocol/experiment_plan_v2_corrected/full/data_protocol_report.json
```

Run a matrix row by zero-based index with `scripts/experiments/run_one.py`.
Every row is bound to the full config SHA-256, the full data-protocol report
SHA-256, and the exact repository revision; changing code, config, or report
requires regenerating the matrix. Formal paper runs fail before data loading if
the full-report SHA is missing. A row is complete only after its schema-v2
`summary.json` passes the same identity checks.

```bash
python scripts/experiments/run_one.py \
  outputs/experiment_plan_v2_corrected/matrices/experiment_plan_v2_corrected.jsonl \
  0
```

The `official_native` track is deliberately separate and fail-closed. At this
revision no audited method has an end-to-end upstream-native data, training,
and evaluation recipe, so this command produces skipped evidence rather than
mislabeling component reuse as an official reproduction:

```bash
python scripts/experiments/build_matrix.py \
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
python scripts/export_results_xlsx.py \
  --matrix outputs/experiment_plan_v2_corrected/matrices/experiment_plan_v2_corrected.jsonl \
  --output outputs/experiment_plan_v2_corrected_summary.xlsx
```
