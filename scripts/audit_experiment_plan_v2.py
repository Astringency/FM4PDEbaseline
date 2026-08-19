#!/usr/bin/env python
"""Fail-closed audit for the historical ``experiment_plan_v2`` cohort.

The audit is deliberately separate from the result exporter.  It reads the
matrix, every referenced ``summary.json``, and the original xlsx workbook, but
never updates any of them.  The known 87-row cohort is classified with the
issues confirmed during the baseline implementation audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "outputs/experiment_plan_v2/matrices/experiment_plan_v2.jsonl"
DEFAULT_WORKBOOK = ROOT / "outputs/experiment_plan_v2_summary.xlsx"
DEFAULT_SENSOR_EVIDENCE = ROOT / "outputs/sensor_generalization_retrain"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/experiment_plan_v2_audit"
DEFAULT_DOC = ROOT / "docs/experiment_plan_v2_audit.md"

EXPECTED_TOTAL_ROWS = 87
EXPECTED_REASON_COUNTS = {
    "stale_eval_only_artifact": 33,
    "sparse_forward_protocol_invalid": 21,
    "burger_inverse_target_leakage": 4,
    "sparse_solution_physics_hidden_truth": 10,
    "legacy_summary_schema": 14,
}
EXPECTED_HARD_EXCLUDED_UNIQUE = 74
EXPECTED_LEGACY_OVERLAP = 8
EXPECTED_DETAIL_REASON_COUNTS = {
    "sparse_forward_wrong_input_normalization": 15,
    "sparse_forward_physics_hidden_target": 6,
    "sparse_forward_voronoicnn_hidden_target": 2,
}

HARD_REASON_DESCRIPTIONS = {
    "stale_eval_only_artifact": (
        "train_time 恰为 0：该行从旧 checkpoint 重新评估，并非由当前矩阵运行完成训练"
    ),
    "sparse_forward_protocol_invalid": (
        "全部 sparse-forward 行使用了无效协议；细分标签记录已独立确认的归一化和隐藏目标机制"
    ),
    "burger_inverse_target_leakage": (
        "审计版本的 Burger inverse 输入暴露了 t=0 目标；部分适配器训练时广播、评估时裁剪"
    ),
    "sparse_solution_physics_hidden_truth": (
        "PINN/PDEOpt sparse-solution 行读取完整隐藏物理场，违反 sensor-only 比较协议"
    ),
    "legacy_summary_schema": (
        "summary 早于强制 provenance schema，缺少 commit_hash"
    ),
}

DETAIL_REASON_DESCRIPTIONS = {
    "sparse_forward_wrong_input_normalization": (
        "15 个 amortized sparse-forward 行用 target/solution 而不是 input/source 统计量"
        "归一化输入观测"
    ),
    "sparse_forward_physics_hidden_target": (
        "6 个 per-instance sparse-forward 行生成时，target fields 尚未与物理方法 metadata 隔离"
    ),
    "sparse_forward_voronoicnn_hidden_target": (
        "2 个 VoronoiCNN sparse-forward 行由 commit 3340de0 生成，其 Voronoi 构造可读取"
        "稀疏目标解"
    ),
}

OFFICIAL_CLASSIFICATIONS = {
    "fno": {
        "classification": "official_component_reuse",
        "exact_end_to_end_official": False,
        "finding": (
            "复用了 NeuralOperator FNO 组件，但训练器、优化器、归一化、早停和评估均为本地协议。"
        ),
    },
    "deeponet": {
        "classification": "official_component_reuse",
        "exact_end_to_end_official": False,
        "finding": (
            "复用了 DeepXDE Cartesian-product 网络组件，但未使用官方 Model.compile/Model.train "
            "实验协议。"
        ),
    },
    "ifno": {
        "classification": "adapted_reimplementation",
        "exact_end_to_end_official": False,
        "finding": "本地 official-aligned 架构实现，不是固定 revision 的直接官方运行。",
    },
    "recfno": {
        "classification": "adapted_reimplementation_major_input_divergence",
        "exact_end_to_end_official": False,
        "finding": (
            "审计适配器使用 zero-filled field + mask + coordinates，width/modes/loss/训练设置"
            "均不同于 vendored recipe。"
        ),
    },
    "senseiver": {
        "classification": "adapted_reimplementation_major_encoding_divergence",
        "exact_end_to_end_official": False,
        "finding": (
            "审计适配器使用 raw coordinates 和不同 latent/layer 设置，没有采用 vendored "
            "Fourier positional encoding recipe。"
        ),
    },
    "voronoicnn": {
        "classification": "architecture_reimplementation",
        "exact_end_to_end_official": False,
        "finding": (
            "复现了核心层类型，但 width、batch size、epochs 和统一任务协议不同于 vendored 实验。"
        ),
    },
    "pinn_sparse": {
        "classification": "project_canonical_custom_baseline",
        "exact_end_to_end_official": False,
        "finding": "项目定义的 sparse PINN 协议；未固定可对应的上游完整实验。",
    },
    "pde_opt": {
        "classification": "project_custom_baseline",
        "exact_end_to_end_official": False,
        "finding": "项目定义的 per-instance PDE optimization baseline。",
    },
    "pc_bnn": {
        "classification": "official_component_or_aligned_reimplementation",
        "exact_end_to_end_official": False,
        "finding": (
            "仅 shallow-water 三通道 sparse reconstruction 与论文假设匹配；标量 PDE 的通用 "
            "SVGD 版本只允许作为独立 debug 方法，且 v2 表未运行该方法。"
        ),
    },
    "var4d": {
        "classification": "project_canonical_math_baseline",
        "exact_end_to_end_official": False,
        "finding": (
            "项目本地 4D-Var/两层动力学实现；只有显式 time-varying trajectory 协议可进入统一比较，"
            "v2 表未运行。"
        ),
    },
    "vivid": {
        "classification": "official_component_or_aligned_adapter",
        "exact_end_to_end_official": False,
        "finding": (
            "需要显式 inverse-observation operator 与完整时变轨迹；统一训练/评估仍为本地适配，"
            "v2 表未运行。"
        ),
    },
}

PROVENANCE_FIELDS = (
    "run_name",
    "checkpoint_path",
    "commit_hash",
    "config_hash",
    "config_path",
    "file_paths_summary",
    "data_root",
    "official_repo",
    "official_backend",
    "official_commit_or_version",
    "official_local_modifications",
    "official_alignment_level",
    "official_vendored_path",
    "official_import_path",
    "official_import_success",
    "official_metadata_note",
    "implementation_mode_effective",
    "implementation_source",
    "backend_used",
    "backend_warning",
    "adapter_status",
    "fallback_used",
    "paper_table_eligible",
    "mask_id",
    "requested_sensor_mode",
    "effective_sensor_mode",
    "train_time",
    "best_epoch",
    "best_val_loss",
    "train_size_requested",
    "effective_train_size",
    "val_requested_size",
    "test_size",
    "inference_optimization_time_total",
    "inference_optimization_time_per_sample",
    "inference_time_total",
    "inference_time_per_sample",
    "num_params",
    "input_channel_names",
    "target_channel_names",
    "observation_field_name",
    "predicted_field_name",
    "native_input_shape",
    "target_shape_native",
)

METRIC_FIELDS = (
    "mse_mean",
    "mae_mean",
    "relative_l2_solution_mean",
    "relative_l2_input_or_coeff_mean",
    "obs_mse_clean_mean",
    "pde_residual_mean",
    "physics_loss_mean",
)


class AuditInputError(RuntimeError):
    """The source cohort is incomplete or is not the audited 87-row cohort."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference)
    if letters is None:
        raise AuditInputError(f"Invalid xlsx cell reference: {cell_reference!r}")
    result = 0
    for char in letters.group(0):
        result = result * 26 + (ord(char) - ord("A") + 1)
    return result - 1


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter() if node.tag.endswith("}t"))
    value_node = next((node for node in cell if node.tag.endswith("}v")), None)
    if value_node is None or value_node.text is None:
        return ""
    if cell_type == "s":
        return shared_strings[int(value_node.text)]
    return value_node.text


def inspect_xlsx(path: Path) -> dict[str, Any]:
    """Inspect a workbook with the standard library; never modify it."""

    if not path.is_file():
        raise AuditInputError(f"Source workbook not found: {path}")
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in names:
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root:
                shared_strings.append(
                    "".join(
                        node.text or ""
                        for node in item.iter()
                        if node.tag.endswith("}t")
                    )
                )

        worksheets = sorted(
            name
            for name in names
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        if not worksheets:
            raise AuditInputError(f"No worksheet XML found in {path}")
        sheet_root = ET.fromstring(archive.read(worksheets[0]))
        rows: list[list[str]] = []
        has_formulas = False
        for row_node in sheet_root.iter():
            if not row_node.tag.endswith("}row"):
                continue
            values: dict[int, str] = {}
            for cell in row_node:
                if not cell.tag.endswith("}c"):
                    continue
                has_formulas = has_formulas or any(
                    child.tag.endswith("}f") for child in cell
                )
                index = _column_index(cell.attrib.get("r", ""))
                values[index] = _xlsx_cell_value(cell, shared_strings)
            if values:
                rows.append([values.get(index, "") for index in range(max(values) + 1)])

    if not rows:
        raise AuditInputError(f"No rows found in source workbook: {path}")
    headers = rows[0]
    run_ids: list[str] = []
    if "run_id" in headers:
        run_id_index = headers.index("run_id")
        run_ids = [row[run_id_index] for row in rows[1:] if len(row) > run_id_index]
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "worksheet_xml": worksheets[0],
        "data_rows": len(rows) - 1,
        "columns": len(headers),
        "headers": headers,
        "run_ids": run_ids,
        "has_formulas": has_formulas,
    }


def _load_matrix(matrix_path: Path) -> list[dict[str, Any]]:
    if not matrix_path.is_file():
        raise AuditInputError(f"Matrix not found: {matrix_path}")
    rows = []
    for line_number, line in enumerate(matrix_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("skip_reason"):
            continue
        if not row.get("run_id") or not row.get("output_dir"):
            raise AuditInputError(f"Matrix row {line_number} lacks run_id/output_dir")
        rows.append(row)
    return rows


def _resolve_repo_path(value: str, *, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _hard_reasons(matrix_row: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if summary.get("train_time") == 0:
        reasons.append("stale_eval_only_artifact")
    if str(matrix_row.get("task_group", "")).startswith("sparse_forward"):
        reasons.append("sparse_forward_protocol_invalid")
    if matrix_row.get("pde") == "burger" and "inverse" in str(
        matrix_row.get("task_group", "")
    ):
        reasons.append("burger_inverse_target_leakage")
    if matrix_row.get("task_group") == "sparse_solution_main_physics":
        reasons.append("sparse_solution_physics_hidden_truth")
    if not summary.get("commit_hash"):
        reasons.append("legacy_summary_schema")
    return reasons


def _conditional_reasons(
    matrix_row: dict[str, Any], *, cohort_seed_count: int
) -> list[str]:
    reasons = ["requires_corrected_protocol_rerun"]
    if matrix_row.get("sensor_mode") == "random":
        reasons.append("fixed_sensor_layout_shared_across_splits")
    if cohort_seed_count == 1:
        reasons.append("single_seed_only")
    return reasons


def _detail_reasons(
    matrix_row: dict[str, Any], summary: dict[str, Any]
) -> list[str]:
    task_group = str(matrix_row.get("task_group", ""))
    reasons: list[str] = []
    if task_group == "sparse_forward_main_amortized":
        reasons.append("sparse_forward_wrong_input_normalization")
    if task_group == "sparse_forward_main_physics":
        reasons.append("sparse_forward_physics_hidden_target")
    if (
        task_group == "sparse_forward_main_amortized"
        and matrix_row.get("baseline") == "voronoicnn"
        and str(summary.get("commit_hash", "")).startswith("3340de0")
    ):
        reasons.append("sparse_forward_voronoicnn_hidden_target")
    return reasons


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_sensor_generalization(root: Path | None) -> list[dict[str, Any]]:
    if root is None or not root.is_dir():
        return []
    raw: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/eval_testseed*/summary.json")):
        match = re.search(r"eval_testseed(\d+)", str(summary_path))
        if match is None:
            continue
        data = json.loads(summary_path.read_text())
        raw.append(
            {
                "baseline": data.get("baseline")
                or summary_path.parents[1].name.split("_")[0],
                "test_sensor_seed": int(match.group(1)),
                "relative_l2_solution_mean": _finite_or_none(
                    data.get("relative_l2_solution_mean")
                ),
                "mse_mean": _finite_or_none(data.get("mse_mean")),
                "mask_id": data.get("mask_id"),
                "checkpoint_path": data.get("checkpoint_path"),
                "summary_path": str(summary_path),
                "summary_sha256": _sha256(summary_path),
            }
        )
    reference = {
        row["baseline"]: row["relative_l2_solution_mean"]
        for row in raw
        if row["test_sensor_seed"] == 1
    }
    for row in raw:
        base = reference.get(row["baseline"])
        value = row["relative_l2_solution_mean"]
        row["degradation_vs_seed1"] = (
            value / base if base not in (None, 0) and value is not None else None
        )
    return raw


def build_audit(
    *,
    matrix_path: Path,
    source_workbook_path: Path,
    sensor_generalization_root: Path | None = None,
    repository_root: Path = ROOT,
    assert_known_cohort: bool = True,
) -> dict[str, Any]:
    """Build an in-memory row-level audit from immutable source artifacts."""

    matrix_path = Path(matrix_path)
    source_workbook_path = Path(source_workbook_path)
    workbook = inspect_xlsx(source_workbook_path)
    matrix_rows = _load_matrix(matrix_path)
    matrix_run_ids = [str(row["run_id"]) for row in matrix_rows]

    if workbook["data_rows"] != len(matrix_rows):
        raise AuditInputError(
            "Source workbook/matrix row mismatch: "
            f"{workbook['data_rows']} != {len(matrix_rows)}"
        )
    if not workbook["run_ids"]:
        raise AuditInputError("Source workbook has no auditable run_id column")
    if len(workbook["run_ids"]) != len(matrix_run_ids):
        raise AuditInputError("Source workbook run_id count does not match the matrix")
    if set(workbook["run_ids"]) != set(matrix_run_ids):
        raise AuditInputError("Source workbook run_ids do not match the matrix")
    if len(set(matrix_run_ids)) != len(matrix_run_ids):
        raise AuditInputError("Matrix contains duplicate run_id values")

    seed_count = len({row.get("seed") for row in matrix_rows})
    audit_rows: list[dict[str, Any]] = []
    missing_summaries: list[str] = []
    for matrix_row in matrix_rows:
        summary_path = _resolve_repo_path(
            str(matrix_row["output_dir"]), root=repository_root
        ) / "summary.json"
        if not summary_path.is_file():
            missing_summaries.append(str(matrix_row["run_id"]))
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("run_id") != matrix_row.get("run_id"):
            raise AuditInputError(
                f"Summary run_id mismatch for {matrix_row.get('run_id')}: "
                f"{summary.get('run_id')!r}"
            )
        hard_reasons = _hard_reasons(matrix_row, summary)
        detail_reasons = _detail_reasons(matrix_row, summary)
        conditional_reasons = _conditional_reasons(
            matrix_row, cohort_seed_count=seed_count
        )
        record: dict[str, Any] = {
            "run_id": matrix_row["run_id"],
            "task_group": matrix_row.get("task_group"),
            "task": matrix_row.get("task"),
            "pde": matrix_row.get("pde"),
            "baseline": matrix_row.get("baseline"),
            "seed": matrix_row.get("seed"),
            "hard_excluded": bool(hard_reasons),
            "audit_status": (
                "hard_excluded" if hard_reasons else "conditional_rerun_required"
            ),
            "publishable": False,
            "hard_reason_tags": hard_reasons,
            "detail_reason_tags": detail_reasons,
            "conditional_reason_tags": conditional_reasons,
            "reason_tags": hard_reasons + detail_reasons + conditional_reasons,
            "reason_evidence": [
                HARD_REASON_DESCRIPTIONS[tag] for tag in hard_reasons
            ]
            + [DETAIL_REASON_DESCRIPTIONS[tag] for tag in detail_reasons],
            "official_consistency_classification": OFFICIAL_CLASSIFICATIONS.get(
                str(matrix_row.get("baseline")), {}
            ).get("classification"),
            "matrix_path": str(matrix_path),
            "matrix_output_dir": matrix_row.get("output_dir"),
            "summary_path": str(summary_path),
            "summary_sha256": _sha256(summary_path),
            "source_workbook_path": str(source_workbook_path),
            "source_workbook_sha256": workbook["sha256"],
            "matrix_train_size": matrix_row.get("train_size"),
            "matrix_val_size": matrix_row.get("val_size"),
            "matrix_test_size": matrix_row.get("test_size"),
            "matrix_sensor_mode": matrix_row.get("sensor_mode"),
            "matrix_num_sensors": matrix_row.get("num_sensors"),
        }
        for field in PROVENANCE_FIELDS + METRIC_FIELDS:
            record[field] = _finite_or_none(summary.get(field))
        audit_rows.append(record)

    if missing_summaries:
        raise AuditInputError(
            f"Missing {len(missing_summaries)} summaries: {missing_summaries[:5]}"
        )

    reason_counts = {
        reason: sum(reason in row["hard_reason_tags"] for row in audit_rows)
        for reason in HARD_REASON_DESCRIPTIONS
    }
    detail_reason_counts = {
        reason: sum(reason in row["detail_reason_tags"] for row in audit_rows)
        for reason in DETAIL_REASON_DESCRIPTIONS
    }
    hard_excluded = sum(row["hard_excluded"] for row in audit_rows)
    legacy_overlap = sum(
        "legacy_summary_schema" in row["hard_reason_tags"]
        and len(row["hard_reason_tags"]) > 1
        for row in audit_rows
    )
    row_counts = {
        "total": len(audit_rows),
        "hard_excluded_unique": hard_excluded,
        "conditional_only": len(audit_rows) - hard_excluded,
        "publishable": sum(row["publishable"] for row in audit_rows),
    }
    conditional_only_rows = [row for row in audit_rows if not row["hard_excluded"]]
    conditional_only_breakdown = {
        "fixed_sensor_layout_shared_across_splits": sum(
            "fixed_sensor_layout_shared_across_splits"
            in row["conditional_reason_tags"]
            for row in conditional_only_rows
        ),
        "non_sensor_single_seed": sum(
            "fixed_sensor_layout_shared_across_splits"
            not in row["conditional_reason_tags"]
            for row in conditional_only_rows
        ),
    }
    sensor_evidence = _load_sensor_generalization(
        Path(sensor_generalization_root)
        if sensor_generalization_root is not None
        else None
    )
    if assert_known_cohort:
        mismatches = []
        if len(audit_rows) != EXPECTED_TOTAL_ROWS:
            mismatches.append(f"total={len(audit_rows)}")
        if reason_counts != EXPECTED_REASON_COUNTS:
            mismatches.append(f"reason_counts={reason_counts}")
        if detail_reason_counts != EXPECTED_DETAIL_REASON_COUNTS:
            mismatches.append(f"detail_reason_counts={detail_reason_counts}")
        if hard_excluded != EXPECTED_HARD_EXCLUDED_UNIQUE:
            mismatches.append(f"hard_excluded={hard_excluded}")
        if legacy_overlap != EXPECTED_LEGACY_OVERLAP:
            mismatches.append(f"legacy_overlap={legacy_overlap}")
        sensor_signature = {
            (row["baseline"], row["test_sensor_seed"]) for row in sensor_evidence
        }
        expected_sensor_signature = {
            (baseline, seed)
            for baseline in ("recfno", "senseiver")
            for seed in (1, 2, 3)
        }
        if sensor_signature != expected_sensor_signature:
            mismatches.append(f"sensor_evidence={sorted(sensor_signature)}")
        for baseline in ("recfno", "senseiver"):
            method_evidence = sorted(
                (row for row in sensor_evidence if row["baseline"] == baseline),
                key=lambda row: row["test_sensor_seed"],
            )
            values = [row["relative_l2_solution_mean"] for row in method_evidence]
            if (
                len(values) != 3
                or any(value is None for value in values)
                or not all(value > values[0] for value in values[1:])
            ):
                mismatches.append(f"sensor_generalization_values[{baseline}]={values}")
            if len({row["mask_id"] for row in method_evidence}) != 3:
                mismatches.append(f"sensor_masks[{baseline}] are not distinct")
            if len({row["checkpoint_path"] for row in method_evidence}) != 1:
                mismatches.append(f"sensor_checkpoints[{baseline}] do not match")
        if mismatches:
            raise AuditInputError(
                "Inputs are not the confirmed experiment_plan_v2 cohort: "
                + "; ".join(mismatches)
            )

    return {
        "summary": {
            "schema_version": "experiment_plan_v2_audit/v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy": "fail_closed; historical rows require corrected-protocol rerun",
            "row_counts": row_counts,
            "confirmed_reason_counts": reason_counts,
            "confirmed_detail_reason_counts": detail_reason_counts,
            "legacy_schema_overlap_with_other_hard_exclusions": legacy_overlap,
            "conditional_only_breakdown": conditional_only_breakdown,
            "source": {
                "matrix_path": str(matrix_path),
                "matrix_sha256": _sha256(matrix_path),
                "workbook": workbook,
            },
            "data_content_verification": {
                "status": "not_performed",
                "reason": (
                    "The historical PDE data files are not part of the mounted audit evidence; "
                    "split/content overlap must be checked separately with verify_data_protocol.py --full."
                ),
            },
            "sensor_generalization_evidence": sensor_evidence,
            "official_consistency": OFFICIAL_CLASSIFICATIONS,
        },
        "rows": sorted(
            audit_rows,
            key=lambda row: (
                str(row["task_group"]),
                str(row["pde"]),
                str(row["baseline"]),
                int(row["seed"] or 0),
            ),
        ),
    }


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    return _finite_or_none(value)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return value


def _excel_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if isinstance(value, str) and len(value) > 32_767:
        return value[:32_740] + "…[truncated for xlsx]"
    return value


def _write_xlsx_if_available(
    audit: dict[str, Any], path: Path
) -> str | None:
    try:
        from openpyxl import Workbook
    except ModuleNotFoundError:
        return None

    rows = audit["rows"]
    workbook = Workbook()
    row_sheet = workbook.active
    row_sheet.title = "rows"
    if rows:
        headers = list(rows[0])
        row_sheet.append(headers)
        for row in rows:
            row_sheet.append([_excel_value(row.get(header)) for header in headers])
        row_sheet.freeze_panes = "A2"
        row_sheet.auto_filter.ref = row_sheet.dimensions

    summary_sheet = workbook.create_sheet("summary")
    summary_sheet.append(["key", "value"])
    for key, value in audit["summary"].items():
        summary_sheet.append([key, _excel_value(value)])
    summary_sheet.freeze_panes = "A2"
    workbook.save(path)
    return str(path)


def write_audit_artifacts(
    audit: dict[str, Any], output_dir: Path
) -> dict[str, str | None]:
    """Write JSON and CSV derivatives without touching any source artifact."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    rows_json_path = output_dir / "rows.json"
    rows_csv_path = output_dir / "rows.csv"
    rows_xlsx_path = output_dir / "rows.xlsx"
    summary_path.write_text(
        json.dumps(
            _json_compatible(audit["summary"]),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    rows_json_path.write_text(
        json.dumps(
            _json_compatible(audit["rows"]),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = audit["rows"]
    if rows:
        with rows_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(
                {key: _csv_value(value) for key, value in row.items()} for row in rows
            )
    rows_xlsx = _write_xlsx_if_available(audit, rows_xlsx_path)
    return {
        "summary_json": str(summary_path),
        "rows_json": str(rows_json_path),
        "rows_csv": str(rows_csv_path),
        "rows_xlsx": rows_xlsx,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    counts = summary["row_counts"]
    reasons = summary["confirmed_reason_counts"]
    detail_reasons = summary["confirmed_detail_reason_counts"]
    evidence = summary["sensor_generalization_evidence"]
    conditional = summary["conditional_only_breakdown"]
    legacy_overlap = summary["legacy_schema_overlap_with_other_hard_exclusions"]
    lines = [
        "# experiment_plan_v2 baseline 审计",
        "",
        "> 结论：当前 87 行中 **0 行可直接发表**。本报告描述的是已确认的客观实现/协议问题，"
        "不对修改者的主观意图作判断。",
        "",
        "## 结果处置",
        "",
        f"- 硬性排除（去重）：{counts['hard_excluded_unique']}/{counts['total']} 行。",
        f"- 仅条件性排除：{counts['conditional_only']}/{counts['total']} 行；仍须按修复后的协议重新运行。",
        f"  其中 {conditional['fixed_sensor_layout_shared_across_splits']} 行是固定传感器布局的"
        " sparse-inverse，另 1 行是非传感器 full-forward 的单 seed 结果。",
        f"- 可发布：{counts['publishable']}/{counts['total']} 行。",
        f"- 原始工作簿只读校验：`{summary['source']['workbook']['sha256']}`；"
        f"{summary['source']['workbook']['data_rows']} 行、"
        f"{summary['source']['workbook']['columns']} 列、"
        f"公式={'有' if summary['source']['workbook']['has_formulas'] else '无'}。",
        "",
        "五类硬性原因允许在同一行上叠加，不能把类别计数直接相加：",
        "",
        "| 原因标签 | 行数 | 证据与影响 |",
        "|---|---:|---|",
    ]
    for reason, description in HARD_REASON_DESCRIPTIONS.items():
        lines.append(f"| `{reason}` | {reasons[reason]} | {description} |")
    lines.extend(
        [
            "",
            f"legacy schema 的 14 行中有 {legacy_overlap} 行"
            "同时命中另一项硬性问题；五类原因的去重并集为 74 行。",
            "",
            "sparse-forward 的 21 行进一步分解如下（子类可以重叠）：",
            "",
            "| 细分原因标签 | 行数 | 证据与影响 |",
            "|---|---:|---|",
        ]
    )
    for reason, description in DETAIL_REASON_DESCRIPTIONS.items():
        lines.append(f"| `{reason}` | {detail_reasons[reason]} | {description} |")
    lines.extend(
        [
            "",
            "## 为什么稀疏结果会过好",
            "",
            "审计版本中的 `random` 并非逐样本随机：每个 split 都用相同 seed 构造一次 mask，"
            "训练、验证和测试因而共享同一传感器布局。这测量的是同布局插值，不是未知布局泛化。"
            "以下证据使用同一 checkpoint，只改变测试 mask seed：",
            "",
            "| 方法 | test mask seed | relative L2 | 相对 seed=1 恶化倍数 | mask id |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in evidence:
        ratio = row["degradation_vs_seed1"]
        ratio_text = "—" if ratio is None else f"{ratio:.2f}×"
        value = row["relative_l2_solution_mean"]
        value_text = "—" if value is None else f"{value:.6f}"
        lines.append(
            f"| {row['baseline']} | {row['test_sensor_seed']} | {value_text} | "
            f"{ratio_text} | `{row['mask_id']}` |"
        )
    lines.extend(
        [
            "",
            "此外，15 个 amortized sparse-forward 行把 input observation/Voronoi 元数据按 target "
            "solution 统计量归一化；6 个 physics 行存在隐藏 target 读取，15 个 amortized 行中的 2 个 "
            "VoronoiCNN 行也存在该问题。这些机制的去重并集覆盖全部 21 个 sparse-forward 行。"
            "Burger inverse 则把目标初态放进了输入，影响 4 行。",
            "",
            "PINN/PDEOpt 的 `fit()` 是空操作，微秒级 `train_time` 不是实际优化成本；其优化发生在"
            "预测阶段，应使用 `inference_optimization_time_total/per_sample` 比较。",
            "",
            "FNO 的极小 MSE 也不应单独解读为近乎完美：部分解场量纲很小，必须以无量纲 relative L2 "
            "和清晰的任务字段共同报告。当前指标公式本身未发现直接读取测试标签的问题。",
            "",
            "样本内容方面，当前挂载的审计证据不包含原始 PDE 数据文件，因此尚未完成 train/val/test "
            "的全量内容哈希与交集核验，不能排除物理文件重复或内容泄漏。修复后的正式运行必须先通过 "
            "`scripts/verify_data_protocol.py --full`，并把该报告的 SHA-256 绑定进矩阵、checkpoint 和 summary。",
            "",
            "## 与 official 实现的一致性",
            "",
            "`official` 在此项目中常表示导入官方网络组件，并不表示复现官方数据、训练器、损失、"
            "超参数和评估的完整实验流程。分类如下：",
            "vendored source metadata 中的 upstream revision 和 local modifications 仍为 unknown；"
            "在固定 tree/commit 并完成差异核对前，不能声称 exact official reproduction。",
            "",
            "| 方法 | 审计分类 | 完整官方复现 | 说明 |",
            "|---|---|---:|---|",
        ]
    )
    for baseline, item in OFFICIAL_CLASSIFICATIONS.items():
        exact = "是" if item["exact_end_to_end_official"] else "否"
        lines.append(
            f"| {baseline} | `{item['classification']}` | {exact} | {item['finding']} |"
        )
    lines.extend(
        [
            "",
            "## 本次实现已修复什么",
            "",
            "当前代码已实现上述协议修复：`random_per_sample` 为 batch-aware mask，训练按 epoch 换布局；"
            "sparse-forward 改用 input-side normalization；loss/metric 禁止广播与裁剪；Burger full inverse "
            "改为 `u(T)->u(0)` 并禁用 sparse-inverse；模型评估视图不再包含 target/full truth。",
            "",
            "RecFNO 当前使用 Voronoi-filled field + mask + coordinates，Senseiver 当前恢复 Fourier positional "
            "encoding；FNO/DeepONet/RecFNO/Senseiver 均明确标成 official component + unified adapted training，"
            "iFNO 标成 adapted reimplementation。旧 checkpoint 因输入/架构或 provenance 不兼容必须隔离重训，"
            "这些修复不会追溯性地使历史 87 行有效。",
            "",
            "runner/checkpoint/export pipeline 现记录并校验 schema、execution mode、run/config fingerprint、"
            "protocol versions、checkpoint SHA-256 与 comparison track；legacy/mismatch summary 会进入 quarantine。",
            "",
            "## 修复与重新验收",
            "",
            "1. 使用真正的 `random_per_sample`：训练按样本/epoch 采样，验证与测试按 split 和样本 ID "
            "确定性生成不同布局；保留 `fixed` 作为单独控制组。",
            "2. sparse-forward 只用 input-side statistics；训练和评估严格要求 prediction/target shape "
            "完全相同，禁止广播与静默裁剪。",
            "3. Burger full inverse 定义为末态到初态，并移除 Burger sparse-inverse；sensor-only 主表"
            "不允许 PINN/PDEOpt 访问隐藏完整场。",
            "4. 将方法标为 component reuse、architecture reimplementation 或 adapted protocol；只有"
            "固定 upstream revision 且端到端原生协议一致时才标 exact official。",
            "5. 新运行必须保存并校验 commit、完整 config 指纹、checkpoint hash、数据 manifest、split "
            "和 sensor policy；缺失 provenance 或 eval-only 复用不得进入主表。",
            "6. 重新运行至少多个训练 seed，并分别报告 fixed-layout、unseen-layout 和 random-per-sample "
            "面板。历史 87 行只保留为审计证据，不覆盖、不回填。",
            "",
            "## 逐行证据",
            "",
            "机器可读逐行原因和 provenance 位于 `outputs/experiment_plan_v2_audit/rows.json`、"
            "`rows.csv` 与 `rows.xlsx`；汇总位于同目录的 `summary.json`。`hard_reason_tags` 支持"
            "一行多个原因，"
            "`publishable` 对本历史 cohort 始终为 `false`。",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown(audit: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(audit), encoding="utf-8")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--source-workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument(
        "--sensor-generalization-root", type=Path, default=DEFAULT_SENSOR_EVIDENCE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--docs-output", type=Path, default=DEFAULT_DOC)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    audit = build_audit(
        matrix_path=args.matrix,
        source_workbook_path=args.source_workbook,
        sensor_generalization_root=args.sensor_generalization_root,
    )
    outputs = write_audit_artifacts(audit, args.output_dir)
    outputs["markdown"] = str(write_markdown(audit, args.docs_output))
    print(
        json.dumps(
            {
                "row_counts": audit["summary"]["row_counts"],
                "confirmed_reason_counts": audit["summary"]["confirmed_reason_counts"],
                "outputs": outputs,
                "source_workbook_preserved": True,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
