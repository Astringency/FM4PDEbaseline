#!/usr/bin/env python
"""Build the isolated sparse-solution multicondition ablation tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


TASK = "sparse_solution_multicondition"
CONDITIONS = ("a_only", "u_only", "both")
IDENTITY_FIELDS = (
    "checkpoint_path",
    "checkpoint_sha256",
    "train_run_fingerprint",
    "normalization_stats_sha256",
    "test_sample_set_sha256",
    "base_mask_manifest_sha256",
)
LONG_COLUMNS = (
    "pde",
    "baseline",
    "seed",
    "condition_mode",
    "rel_l2_a_mean",
    "rel_l2_a_median",
    "rel_l2_a_p90",
    "rel_l2_u_mean",
    "rel_l2_u_median",
    "rel_l2_u_p90",
    "joint_rel_l2_mean",
    "joint_rel_l2_median",
    "joint_rel_l2_p90",
    "observed_mse_a_mean",
    "observed_mse_u_mean",
    "num_sensor_locations",
    "num_scalar_observations_a",
    "num_scalar_observations_u",
    "num_scalar_observations_total",
    "checkpoint_path",
    "checkpoint_sha256",
    "train_run_fingerprint",
    "normalization_stats_sha256",
    "test_sample_set_sha256",
    "base_mask_manifest_sha256",
    "summary_path",
)
WIDE_COLUMNS = (
    "pde",
    "baseline",
    "seed",
    "a_only_rel_l2_a_mean",
    "a_only_rel_l2_u_mean",
    "u_only_rel_l2_a_mean",
    "u_only_rel_l2_u_mean",
    "both_rel_l2_a_mean",
    "both_rel_l2_u_mean",
    "checkpoint_sha256",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and report sparse_solution_multicondition evaluation views."
    )
    parser.add_argument(
        "--input-root",
        default="results/ablations/sparse_solution_multicondition",
        help="Root containing eval-only summary.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Destination; defaults to <input-root>/report.",
    )
    return parser.parse_args(argv)


def load_evaluation_summaries(input_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(input_root.rglob("summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read {path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("task") != TASK:
            continue
        if payload.get("execution_mode") != "eval_only":
            continue
        if payload.get("status") != "success":
            raise RuntimeError(f"evaluation summary is not successful: {path}")
        row = dict(payload)
        row["summary_path"] = str(path)
        summaries.append(row)
    if not summaries:
        raise RuntimeError(f"no completed {TASK} eval-only summaries found under {input_root}")
    return summaries


def validate_and_build_rows(
    summaries: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        key = (
            str(summary.get("pde", "")),
            str(summary.get("baseline", "")),
            int(summary.get("seed", 0)),
        )
        groups[key].append(summary)

    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for key in sorted(groups):
        pde, baseline, seed = key
        by_condition: dict[str, dict[str, Any]] = {}
        for summary in groups[key]:
            condition = str(
                summary.get("evaluation_condition_mode")
                or summary.get("condition_mode")
                or ""
            )
            if condition not in CONDITIONS:
                raise RuntimeError(f"{key} has invalid evaluation condition {condition!r}")
            if condition in by_condition:
                raise RuntimeError(f"{key} has duplicate {condition!r} evaluation summaries")
            by_condition[condition] = summary
        missing = sorted(set(CONDITIONS) - set(by_condition))
        if missing:
            raise RuntimeError(f"{key} is missing evaluation conditions {missing}")

        identity: dict[str, str] = {}
        for field in IDENTITY_FIELDS:
            values = {str(by_condition[mode].get(field, "") or "") for mode in CONDITIONS}
            if "" in values:
                raise RuntimeError(f"{key} has an empty required provenance field {field!r}")
            if len(values) != 1:
                raise RuntimeError(
                    f"{key} condition views do not share {field}: {sorted(values)}"
                )
            identity[field] = next(iter(values))

        sample_counts = {
            int(by_condition[mode].get("test_size_loaded", by_condition[mode].get("test_size", 0)) or 0)
            for mode in CONDITIONS
        }
        if len(sample_counts) != 1:
            raise RuntimeError(f"{key} condition views use different test sample counts: {sample_counts}")

        for mode in CONDITIONS:
            summary = by_condition[mode]
            _validate_sensor_budget(key, mode, summary)
            _validate_metrics(key, mode, summary)
            long_rows.append({column: summary.get(column, "") for column in LONG_COLUMNS})

        wide_rows.append(
            {
                "pde": pde,
                "baseline": baseline,
                "seed": seed,
                "a_only_rel_l2_a_mean": by_condition["a_only"]["rel_l2_a_mean"],
                "a_only_rel_l2_u_mean": by_condition["a_only"]["rel_l2_u_mean"],
                "u_only_rel_l2_a_mean": by_condition["u_only"]["rel_l2_a_mean"],
                "u_only_rel_l2_u_mean": by_condition["u_only"]["rel_l2_u_mean"],
                "both_rel_l2_a_mean": by_condition["both"]["rel_l2_a_mean"],
                "both_rel_l2_u_mean": by_condition["both"]["rel_l2_u_mean"],
                "checkpoint_sha256": identity["checkpoint_sha256"],
            }
        )
        audits.append(
            {
                "pde": pde,
                "baseline": baseline,
                "seed": seed,
                "conditions": list(CONDITIONS),
                "test_sample_count": next(iter(sample_counts)),
                **identity,
                "single_checkpoint_verified": True,
                "same_test_samples_verified": True,
                "same_base_masks_verified": True,
                "same_normalization_verified": True,
            }
        )
    return long_rows, wide_rows, audits


def _validate_sensor_budget(
    key: tuple[str, str, int], mode: str, summary: dict[str, Any]
) -> None:
    locations = int(summary.get("num_sensor_locations", -1))
    count_a = int(summary.get("num_scalar_observations_a", -1))
    count_u = int(summary.get("num_scalar_observations_u", -1))
    total = int(summary.get("num_scalar_observations_total", -1))
    expected = {
        "a_only": (locations, 0, locations),
        "u_only": (0, locations, locations),
        "both": (locations, locations, 2 * locations),
    }[mode]
    if locations <= 0 or (count_a, count_u, total) != expected:
        raise RuntimeError(
            f"{key}/{mode} has invalid sensor counts: locations={locations}, "
            f"a={count_a}, u={count_u}, total={total}, expected={expected}"
        )


def _validate_metrics(
    key: tuple[str, str, int], mode: str, summary: dict[str, Any]
) -> None:
    for field in (
        "rel_l2_a_mean",
        "rel_l2_a_median",
        "rel_l2_a_p90",
        "rel_l2_u_mean",
        "rel_l2_u_median",
        "rel_l2_u_p90",
        "joint_rel_l2_mean",
        "joint_rel_l2_median",
        "joint_rel_l2_p90",
    ):
        try:
            value = float(summary[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{key}/{mode} is missing numeric metric {field}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"{key}/{mode} metric {field} is not finite: {value}")
    for modality in ("a", "u"):
        field = f"observed_mse_{modality}_mean"
        value = float(summary.get(field, float("nan")))
        active = mode == "both" or mode.startswith(modality)
        if active and not math.isfinite(value):
            raise RuntimeError(f"{key}/{mode} active observed metric {field} is not finite")
        if not active and not math.isnan(value):
            raise RuntimeError(
                f"{key}/{mode} unobserved metric {field} must be NaN/not-applicable, got {value}"
            )


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        return f"{value:.6g}"
    return str(value)


def _strict_json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strict_json_value(item) for item in value]
    return value


def markdown_table(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    labels = {
        "pde": "PDE",
        "baseline": "Baseline",
        "seed": "Seed",
        "condition_mode": "Condition",
        "rel_l2_a_mean": "RelL2-a Mean",
        "rel_l2_u_mean": "RelL2-u Mean",
        "joint_rel_l2_mean": "Joint RelL2 Mean",
        "checkpoint_sha256": "Checkpoint SHA",
        "a_only_rel_l2_a_mean": "a-only: a",
        "a_only_rel_l2_u_mean": "a-only: u",
        "u_only_rel_l2_a_mean": "u-only: a",
        "u_only_rel_l2_u_mean": "u-only: u",
        "both_rel_l2_a_mean": "both: a",
        "both_rel_l2_u_mean": "both: u",
    }
    header = [labels.get(column, column) for column in columns]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    long_rows: list[dict[str, Any]],
    wide_rows: list[dict[str, Any]],
    audits: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    long_csv = output_dir / "sparse_solution_multicondition_long.csv"
    wide_csv = output_dir / "sparse_solution_multicondition_wide.csv"
    markdown = output_dir / "sparse_solution_multicondition_report.md"
    summary_json = output_dir / "sparse_solution_multicondition_summary.json"
    write_csv(long_csv, long_rows, LONG_COLUMNS)
    write_csv(wide_csv, wide_rows, WIDE_COLUMNS)
    compact_long = (
        "pde",
        "baseline",
        "seed",
        "condition_mode",
        "rel_l2_a_mean",
        "rel_l2_u_mean",
        "joint_rel_l2_mean",
        "checkpoint_sha256",
    )
    markdown.write_text(
        "# Sparse Solution Multicondition Ablation\n\n"
        "All three condition views in every row group passed the single-checkpoint, "
        "test-sample, base-mask, and normalization-statistics identity checks.\n\n"
        "## Long table\n\n"
        + markdown_table(long_rows, compact_long)
        + "\n\n## Paper wide table\n\n"
        + markdown_table(wide_rows, WIDE_COLUMNS)
        + "\n",
        encoding="utf-8",
    )
    payload = {
        "task": TASK,
        "num_training_groups": len(audits),
        "num_evaluation_results": len(long_rows),
        "expected_evaluations_per_training_group": len(CONDITIONS),
        "validations": {
            "single_checkpoint": True,
            "same_test_sample_set": True,
            "same_base_sensor_masks": True,
            "same_normalization_statistics": True,
        },
        "long_rows": long_rows,
        "wide_rows": wide_rows,
        "provenance_audit": audits,
        "outputs": {
            "long_csv": str(long_csv),
            "wide_csv": str(wide_csv),
            "markdown": str(markdown),
            "json": str(summary_json),
        },
    }
    summary_json.write_text(
        json.dumps(
            _strict_json_value(payload), indent=2, sort_keys=True, allow_nan=False
        ),
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir) if args.output_dir else input_root / "report"
    summaries = load_evaluation_summaries(input_root)
    long_rows, wide_rows, audits = validate_and_build_rows(summaries)
    payload = write_report(output_dir, long_rows, wide_rows, audits)
    print(json.dumps(payload["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
