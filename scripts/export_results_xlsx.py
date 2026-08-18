#!/usr/bin/env python
"""Export a provenance-validated experiment matrix to XLSX.

Legacy, mismatched, unreadable, and missing summaries are written to a separate
quarantine artifact and block publication.  A formal workbook is emitted only
for a complete, single-provenance cohort.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.provenance import (
    HistoricalExperimentError,
    cohort_id,
    cohort_signature,
    reject_historical_experiment_path,
    summary_validation_reasons,
)


OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", ROOT / "outputs"))
CORRECTED_NAMESPACE = "experiment_plan_v2_corrected"
MATRIX = OUTPUT_ROOT / CORRECTED_NAMESPACE / "matrices" / f"{CORRECTED_NAMESPACE}.jsonl"
OUT_XLSX = OUTPUT_ROOT / f"{CORRECTED_NAMESPACE}_summary.xlsx"
IMMUTABLE_HISTORICAL_WORKBOOK = ROOT / "outputs" / "experiment_plan_v2_summary.xlsx"

PROVENANCE_COLUMNS = [
    "publishable",
    "validation_status",
    "cohort_id",
    "status",
    "matrix_schema_version",
    "summary_schema_version",
    "execution_mode",
    "eval_only",
    "source_train_run_id",
    "source_train_run_fingerprint",
    "source_train_seed",
    "run_id",
    "run_fingerprint",
    "config_content_sha256",
    "experiment_config_sha256",
    "config_hash",
    "config_path",
    "commit_hash",
    "checkpoint_path",
    "checkpoint_sha256",
    "task_protocol_version",
    "sensor_protocol_version",
    "comparison_track",
    "data_manifest_sha256",
    "data_manifest_path",
    "data_manifest_hash",
    "data_root",
    "file_paths_summary",
    "experiment_mode",
    "dry_run",
    "synthetic_data",
]

DESIGN_COLUMNS = [
    "experiment_kind",
    "ablation_factor",
    "task_group",
    "pde",
    "task",
    "baseline",
    "seed",
    "sensor_seed",
    "train_size",
    "train_requested_size",
    "train_size_requested",
    "effective_train_size",
    "train_size_loaded_for_fit",
    "val_size",
    "val_requested_size",
    "val_split_source",
    "val_from_train_offset",
    "test_size",
    "test_requested_size",
    "train_shards",
    "batch_size",
    "epochs",
    "device",
    "num_sensors",
    "requested_sensor_mode",
    "sensor_budget_mode_requested",
    "effective_sensor_mode",
    "sensor_mode",
    "mask_id",
    "mask_ids_unique_count",
    "split_mask_manifest",
    "noise_level",
    "scalar_param_mode",
    "scalar_param_mode_requested",
    "data_loading_mode_requested",
    "effective_data_loading_mode",
    "num_workers",
    "pin_memory_requested",
    "persistent_workers_requested",
    "prefetch_factor_requested",
    "dataloader_num_workers",
    "dataloader_pin_memory",
    "dataloader_persistent_workers",
    "dataloader_prefetch_factor",
    "load_full_trajectory",
    "loaded_full_trajectory",
    "steps",
    "refine_steps",
    "particles",
]

METRIC_COLUMNS = [
    "best_val_loss",
    "best_epoch",
    "mse_mean",
    "mse_std",
    "mae_mean",
    "mae_std",
    "relative_l2_solution_mean",
    "relative_l2_solution_std",
    "relative_l2_input_or_coeff_mean",
    "relative_l2_input_or_coeff_std",
    "obs_mse_clean_mean",
    "obs_mse_noisy_mean",
    "pde_residual_mean",
    "bc_residual_mean",
    "ic_residual_mean",
    "physics_loss_mean",
]

TIMING_COLUMNS = [
    "train_time",
    "fit_setup_time",
    "amortized_training",
    "test_time_optimization",
    "inference_time_total",
    "inference_time_per_sample",
    "inference_optimization_time_total",
    "inference_optimization_time_per_sample",
    "num_params",
    "num_params_storage",
    "num_real_dofs",
    "parameter_count_convention",
]

IMPLEMENTATION_COLUMNS = [
    "backend_used",
    "official_backend",
    "capability_status",
    "implementation_required",
    "implementation_mode_requested",
    "implementation_mode_effective",
    "implementation_source",
    "fallback_used",
    "paper_table_eligible",
    "unified_comparison_eligible",
    "official_native_eligible",
    "adapter_status",
    "citation_key",
    "official_commit_or_version",
    "official_local_modifications",
]

COLUMNS = PROVENANCE_COLUMNS + DESIGN_COLUMNS + METRIC_COLUMNS + TIMING_COLUMNS + IMPLEMENTATION_COLUMNS
SORT_KEYS = ["task_group", "pde", "task", "baseline", "seed"]


class ExportValidationError(RuntimeError):
    """Raised when a workbook would silently combine unpublishable results."""


@dataclass(frozen=True)
class ExportCollection:
    records: list[dict[str, Any]]
    quarantine: list[dict[str, Any]]
    missing_run_ids: list[str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export a provenance-validated experiment workbook")
    parser.add_argument("--matrix", default=str(MATRIX))
    parser.add_argument("--output", default=str(OUT_XLSX))
    parser.add_argument("--quarantine-output", default="")
    parser.add_argument("--quarantine-jsonl", default="")
    return parser.parse_args(argv)


def remap_output_dir(row: dict[str, Any], output_root: str | Path = OUTPUT_ROOT) -> dict[str, Any]:
    remapped = dict(row)
    output_root = Path(output_root)
    for key in ("output_dir", "log_dir", "status_file"):
        value = remapped.get(key)
        if isinstance(value, str) and value.startswith("outputs/"):
            remapped[key] = str(output_root / value[len("outputs/"):])
    return remapped


def load_rows(matrix: str | Path = MATRIX, *, output_root: str | Path = OUTPUT_ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(matrix).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("skip_reason"):
                rows.append(remap_output_dir(row, output_root))
    return rows


def collect_results(rows: list[dict[str, Any]]) -> ExportCollection:
    records: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        summary_path = Path(row["output_dir"]) / "summary.json"
        if not summary_path.exists():
            missing.append(str(row.get("run_id", "?")))
            quarantine.append(_quarantine_record(row, summary_path, ["summary_missing"]))
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            quarantine.append(_quarantine_record(row, summary_path, ["summary_unreadable"], error=str(exc)))
            continue
        if not isinstance(summary, dict):
            quarantine.append(_quarantine_record(row, summary_path, ["summary_not_object"]))
            continue
        reasons = summary_validation_reasons(row, summary)
        if reasons:
            quarantine.append(_quarantine_record(row, summary_path, reasons, summary=summary))
            continue
        records.append(_export_record(row, summary, summary_path))
    return ExportCollection(records=records, quarantine=quarantine, missing_run_ids=missing)


def require_single_cohort(records: list[dict[str, Any]]) -> str:
    cohorts = {str(record.get("cohort_id", "")) for record in records}
    cohorts.discard("")
    if len(cohorts) != 1:
        raise ExportValidationError(
            f"multiple provenance cohorts would be mixed: {sorted(cohorts)}"
            if cohorts
            else "no provenance cohort is available"
        )
    return next(iter(cohorts))


def _export_record(row: dict[str, Any], summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    record = {column: summary.get(column) for column in COLUMNS}
    record.update(
        {
            "publishable": True,
            "validation_status": "valid",
            "cohort_id": cohort_id(summary),
            "summary_path": str(summary_path),
            "matrix_config": row.get("config"),
            "matrix_train_size": row.get("train_size"),
            "matrix_val_size": row.get("val_size"),
            "matrix_test_size": row.get("test_size"),
            "matrix_batch_size": row.get("batch_size"),
            "matrix_epochs": row.get("epochs"),
            "matrix_device": row.get("device"),
        }
    )
    # Design fields that are immutable in the fingerprint may not be repeated
    # by older runner internals.  For a validated v2 summary the matrix is the
    # authoritative fallback; metrics and effective-size fields remain summary-only.
    for column in (*DESIGN_COLUMNS, *IMPLEMENTATION_COLUMNS):
        if record.get(column) is None and column in row:
            record[column] = row[column]
    return record


def _quarantine_record(
    row: dict[str, Any],
    summary_path: Path,
    reasons: list[str],
    *,
    summary: dict[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    summary = summary or {}
    return {
        "publishable": False,
        "validation_status": "quarantined",
        "validation_reasons": reasons,
        "run_id": row.get("run_id"),
        "task_group": row.get("task_group"),
        "pde": row.get("pde"),
        "task": row.get("task"),
        "baseline": row.get("baseline"),
        "seed": row.get("seed"),
        "summary_path": str(summary_path),
        "error": error,
        "observed_run_id": summary.get("run_id"),
        "observed_run_fingerprint": summary.get("run_fingerprint"),
        "observed_config_content_sha256": summary.get("config_content_sha256"),
        "observed_experiment_config_sha256": summary.get(
            "experiment_config_sha256"
        ),
        "observed_data_manifest_sha256": summary.get("data_manifest_sha256"),
        "observed_data_manifest_path": summary.get("data_manifest_path"),
        "observed_config_hash": summary.get("config_hash"),
        "observed_commit_hash": summary.get("commit_hash"),
        "observed_execution_mode": summary.get("execution_mode"),
        "observed_checkpoint_path": summary.get("checkpoint_path"),
    }


def _json_safe_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


def _dataframe(records: list[dict[str, Any]], columns: list[str] | None = None) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=columns or [])
    dataframe = pd.DataFrame.from_records(records)
    dataframe = dataframe.map(_json_safe_cell)
    if columns:
        ordered = list(dict.fromkeys(columns + list(dataframe.columns)))
        dataframe = dataframe.reindex(columns=ordered)
    available_sort_keys = [key for key in SORT_KEYS if key in dataframe.columns]
    if available_sort_keys:
        dataframe = dataframe.sort_values(available_sort_keys, kind="stable").reset_index(drop=True)
    return dataframe


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _write_quarantine(
    path: Path,
    jsonl_path: Path,
    quarantine: list[dict[str, Any]],
    validated_preview: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    _write_jsonl(jsonl_path, quarantine)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _dataframe(quarantine).to_excel(writer, sheet_name="quarantine", index=False)
        _dataframe(validated_preview, COLUMNS).to_excel(writer, sheet_name="validated_preview", index=False)
        _dataframe([manifest]).to_excel(writer, sheet_name="manifest", index=False)


def _archive_existing_publication(output: Path, quarantine_output: Path) -> str:
    """Remove a stale workbook from the publishable path without deleting it."""
    if not output.exists():
        return ""
    if output.resolve() == quarantine_output.resolve():
        raise ExportValidationError("output and quarantine-output must be different paths")
    if not output.is_file():
        raise ExportValidationError(f"refusing to replace non-file publication target: {output}")

    quarantine_output.parent.mkdir(parents=True, exist_ok=True)
    stem = f"{quarantine_output.stem}_previous_publication"
    archived = quarantine_output.with_name(f"{stem}{output.suffix}")
    suffix = 1
    while archived.exists():
        archived = quarantine_output.with_name(f"{stem}_{suffix}{output.suffix}")
        suffix += 1
    shutil.move(str(output), str(archived))
    return str(archived)


def _is_immutable_historical_workbook(path: Path) -> bool:
    return path.expanduser().resolve() == IMMUTABLE_HISTORICAL_WORKBOOK.expanduser().resolve()


def _prepare_blocked_publication(
    output: Path,
    quarantine_output: Path,
    manifest: dict[str, Any],
) -> None:
    """Archive stale corrected output while leaving historical evidence in place."""
    if _is_immutable_historical_workbook(output):
        manifest["immutable_historical_workbook_preserved"] = str(output)
        return
    archived = _archive_existing_publication(output, quarantine_output)
    if archived:
        manifest["previous_publication_quarantined_as"] = archived


def _require_mutable_publication_target(output: Path) -> None:
    if _is_immutable_historical_workbook(output):
        raise ExportValidationError(
            "refusing to overwrite immutable historical workbook "
            f"{output}; publish the corrected cohort to {OUT_XLSX}"
        )


def _require_safe_artifact_targets(
    output: Path,
    quarantine_output: Path,
    quarantine_jsonl: Path,
) -> None:
    targets = {
        "output": output,
        "quarantine-output": quarantine_output,
        "quarantine-jsonl": quarantine_jsonl,
    }
    resolved = {name: path.expanduser().resolve() for name, path in targets.items()}
    if len(set(resolved.values())) != len(resolved):
        raise ExportValidationError("output and quarantine artifact paths must be different")
    for name in ("quarantine-output", "quarantine-jsonl"):
        if _is_immutable_historical_workbook(targets[name]):
            raise ExportValidationError(
                f"refusing to overwrite immutable historical workbook {targets[name]} as {name}"
            )
    for name, target in targets.items():
        try:
            reject_historical_experiment_path(target, field=name)
        except HistoricalExperimentError as exc:
            raise ExportValidationError(str(exc)) from exc


def export_results(
    rows: list[dict[str, Any]],
    *,
    output: str | Path,
    quarantine_output: str | Path,
    quarantine_jsonl: str | Path,
    matrix_path: str | Path = "",
) -> dict[str, Any]:
    output = Path(output)
    quarantine_output = Path(quarantine_output)
    quarantine_jsonl = Path(quarantine_jsonl)
    _require_safe_artifact_targets(output, quarantine_output, quarantine_jsonl)
    collection = collect_results(rows)
    manifest: dict[str, Any] = {
        "matrix": str(matrix_path),
        "matrix_rows": len(rows),
        "validated_rows": len(collection.records),
        "quarantined_rows": len(collection.quarantine),
        "missing_summaries": len(collection.missing_run_ids),
        "missing_run_ids": collection.missing_run_ids,
    }
    if collection.quarantine:
        manifest["publication_blocked"] = True
        manifest["block_reason"] = "legacy_or_mismatched_summaries"
        _prepare_blocked_publication(output, quarantine_output, manifest)
        _write_quarantine(
            quarantine_output,
            quarantine_jsonl,
            collection.quarantine,
            collection.records,
            manifest,
        )
        raise ExportValidationError(
            f"publication blocked: {len(collection.quarantine)} legacy or mismatched summaries were quarantined"
        )
    if not collection.records:
        manifest["publication_blocked"] = True
        manifest["block_reason"] = "no_validated_summaries"
        _prepare_blocked_publication(output, quarantine_output, manifest)
        _write_quarantine(quarantine_output, quarantine_jsonl, [], [], manifest)
        raise ExportValidationError("No provenance-validated summary.json files found.")

    try:
        cohort = require_single_cohort(collection.records)
    except ExportValidationError:
        mixed = [
            _quarantine_record(
                row=record,
                summary_path=Path(str(record.get("summary_path", ""))),
                reasons=["mixed_provenance_cohort"],
                summary=record,
            )
            for record in collection.records
        ]
        manifest["publication_blocked"] = True
        manifest["block_reason"] = "mixed_provenance_cohorts"
        manifest["cohort_signatures"] = [cohort_signature(record) for record in collection.records]
        _prepare_blocked_publication(output, quarantine_output, manifest)
        _write_quarantine(quarantine_output, quarantine_jsonl, mixed, collection.records, manifest)
        raise

    _require_mutable_publication_target(output)
    manifest.update({"publication_blocked": False, "cohort_id": cohort})
    dataframe = _dataframe(collection.records, COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, sheet_name="runs", index=False)
        _dataframe([manifest]).to_excel(writer, sheet_name="manifest", index=False)
        if collection.missing_run_ids:
            _dataframe([{"run_id": run_id} for run_id in collection.missing_run_ids]).to_excel(
                writer, sheet_name="missing", index=False
            )
    return {"output": str(output), "rows": len(dataframe), "columns": len(dataframe.columns), **manifest}


def _derived_quarantine_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_quarantine{output.suffix}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    matrix = Path(args.matrix)
    output = Path(args.output)
    quarantine_output = Path(args.quarantine_output) if args.quarantine_output else _derived_quarantine_path(output)
    quarantine_jsonl = (
        Path(args.quarantine_jsonl)
        if args.quarantine_jsonl
        else quarantine_output.with_suffix(".jsonl")
    )
    rows = load_rows(matrix, output_root=OUTPUT_ROOT)
    try:
        report = export_results(
            rows,
            output=output,
            quarantine_output=quarantine_output,
            quarantine_jsonl=quarantine_jsonl,
            matrix_path=matrix,
        )
    except ExportValidationError as exc:
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "output_written": False,
                    "quarantine_output": str(quarantine_output),
                    "quarantine_jsonl": str(quarantine_jsonl),
                },
                indent=2,
            )
        )
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
