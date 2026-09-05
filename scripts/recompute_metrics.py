#!/usr/bin/env python
"""Recompute prediction errors on CPU from complete saved evaluation samples.

Updates results_raw.jsonl and run summary JSON/JSONL/CSV files. Physics metrics
and measured runtimes retain their original values. Sample tensors, manifests,
and the historical metrics embedded in sample .pt files are not rewritten.
No model, checkpoint, training data, or inference is needed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.common.sample_artifacts import load_evaluation_sample
from baselines.run import (
    _joint_reconstruction_relative_l2_values,
    _json_safe,
    _mae_values,
    _mean_list,
    _metric_stats,
    _mse_values,
    _multicondition_metric_stats,
    _multicondition_metrics,
    _relative_l2_input_or_coeff_values,
    _relative_l2_values,
    _solution_metric_fields,
)
from scripts.run_eval import detect_output_root, filter_rows, load_matrix, source_run_dir, split_selection


_MULTICONDITION_KEYS = {"rel_l2_a", "rel_l2_u", "joint_rel_l2", "observed_mse_a", "observed_mse_u"}


def _jsonl(text: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL records must be objects")
    return rows


def _jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(_json_safe(row), sort_keys=True) + "\n" for row in rows)


def _prediction_errors(sample: dict[str, Any]) -> dict[str, float]:
    pred = sample["prediction"].unsqueeze(0)
    target = sample["target_fields"].unsqueeze(0)
    if pred.shape != target.shape or target.numel() == 0:
        raise ValueError("saved prediction and target must have identical non-empty shapes")
    task = sample["task"]
    metadata = sample["metadata"]
    if sample["pde_name"] == "burger" and task in {"sparse_solution", "sparse_reconstruction"}:
        if target.ndim != 4 or target.shape[1] != 1 or target.shape[2] < 2:
            raise ValueError("Burgers reconstruction requires a complete [1,1,T,X] trajectory")
        metadata = {**metadata, "joint_reconstruction": True, "joint_split_axis": 2, "joint_input_extent": 1}
    if task in {"sparse_solution", "sparse_reconstruction"} and metadata.get("joint_reconstruction"):
        solution, initial = _joint_reconstruction_relative_l2_values(pred, target, metadata)
    else:
        solution = [float("nan")] if task in {"inverse", "sparse_inverse"} else _relative_l2_values(pred, target)
        initial = _relative_l2_input_or_coeff_values(task, pred, target)
    metrics = {
        "relative_l2_solution": solution[0],
        "relative_l2_input_or_coeff": initial[0],
        "mse": _mse_values(pred, target)[0],
        "mae": _mae_values(pred, target)[0],
    }
    if task == "sparse_solution_multicondition":
        values = _multicondition_metrics(pred, target, SimpleNamespace(metadata=metadata, mask=sample["mask"]))
        metrics.update({key: values[f"{key}_values"][0] for key in _MULTICONDITION_KEYS})
    return metrics


def recompute_run(
    run_dir: str | Path,
    *,
    dry_run: bool = False,
    workers: int = 4,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Validate an entire run before replacing any metric output files."""
    if workers < 1:
        raise ValueError("workers must be positive")
    directory = Path(run_dir).expanduser().resolve()
    originals: dict[Path, str] = {}

    def read(path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        originals[path] = text
        return text

    summary_path = directory / "summary.json"
    summary = json.loads(read(summary_path))
    if not isinstance(summary, dict) or summary.get("status") != "success":
        raise ValueError(f"a successful source summary is required: {summary_path}")
    run_id = str(summary.get("run_id", ""))
    count = int(summary.get("test_size", 0) or 0)
    if count <= 0 or int(summary.get("sample_artifact_count", 0) or 0) != count:
        raise ValueError(f"saved sample count does not match test_size: {directory}")
    # Recorded absolute paths may belong to another host. Only read this run's
    # local sample directory, never an unrelated run at the old mount.
    sample_name = Path(str(summary.get("sample_artifact_dir") or "samples")).name
    sample_dir = directory / sample_name
    manifest_path = sample_dir / "manifest.jsonl"
    manifest = _jsonl(read(manifest_path))
    if len(manifest) != count:
        raise ValueError(f"incomplete sample manifest: expected {count}, found {len(manifest)}")

    raw_path = directory / "results_raw.jsonl"
    raw_rows = _jsonl(read(raw_path))
    batches = [row for row in raw_rows if str(row.get("run_id", "")) == run_id]
    expected_samples: list[tuple[int, int, str]] = []
    for index, batch in enumerate(batches):
        if batch.get("batch_index") != index:
            raise ValueError("raw batch indices must be contiguous")
        ids = json.loads(batch["global_sample_ids"])
        if not isinstance(ids, list) or len(ids) != int(batch["sample_count"]):
            raise ValueError("raw sample IDs do not match the batch sample count")
        expected_samples.extend((index, item, str(value)) for item, value in enumerate(ids))
    if len(expected_samples) != count or len({item[2] for item in expected_samples}) != count:
        raise ValueError("raw results must cover exactly test_size unique samples")

    def evaluate_sample(ordinal: int) -> tuple[str, dict[str, float]]:
        record = manifest[ordinal]
        batch_index, batch_item_index, sample_id = expected_samples[ordinal]
        identity = {
            "sample_ordinal": ordinal,
            "batch_index": batch_index,
            "batch_item_index": batch_item_index,
            "global_sample_id": sample_id,
        }
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError(f"manifest does not match raw sample identity at ordinal {ordinal}")
        filename = Path(record["artifact_path"]).name
        if filename != f"sample_{ordinal:06d}.pt" or not record.get("artifact_sha256"):
            raise ValueError(f"invalid sample filename or checksum at ordinal {ordinal}")
        path = sample_dir / filename
        sample = load_evaluation_sample(path, expected_sha256=record["artifact_sha256"])
        if any(sample.get(key) != value for key, value in identity.items()):
            raise ValueError(f"saved sample identity mismatch: {path}")
        expected_run = {key: summary[key] for key in ("run_id", "baseline", "pde", "task", "seed")}
        if any(sample.get("run_metadata", {}).get(key) != value for key, value in expected_run.items()):
            raise ValueError(f"saved sample belongs to a different run: {path}")
        if sample.get("pde_name") != summary["pde"] or sample.get("task") != summary["task"]:
            raise ValueError(f"saved sample task mismatch: {path}")
        batch = batches[batch_index]
        for field, shape_field in (("prediction", "pred_shape"), ("target_fields", "target_shape")):
            expected_shape = json.loads(batch[shape_field])[1:]
            if list(sample[field].shape) != expected_shape:
                raise ValueError(f"saved sample shape does not match raw results: {path}: {field}")
        return sample_id, _prediction_errors(sample)

    # Bound concurrent I/O and discard each tensor payload after computing its
    # scalar errors. map preserves manifest order for reproducible aggregation.
    by_sample: dict[str, dict[str, float]] = {}
    executor = ThreadPoolExecutor(max_workers=min(workers, count))
    try:
        for ordinal, (sample_id, values) in enumerate(executor.map(evaluate_sample, range(count))):
            by_sample[sample_id] = values
            if progress and ((ordinal + 1) % 25 == 0 or ordinal + 1 == count):
                progress(f"{directory.name}: verified {ordinal + 1}/{count} samples")
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    metrics = list(next(iter(by_sample.values())))
    updates: dict[str, Any] = {
        "prediction_metrics_source": "saved_samples",
        **_solution_metric_fields(SimpleNamespace(pde=summary["pde"], task=summary["task"])),
    }
    for key in metrics:
        values = [item[key] for item in by_sample.values()]
        stats = _multicondition_metric_stats(values) if key in _MULTICONDITION_KEYS else _metric_stats(values)
        updates.update({f"{key}_{suffix}": value for suffix, value in stats.items()})
    updates = _json_safe(updates)
    for batch in batches:
        ids = json.loads(batch["global_sample_ids"])
        for key in metrics:
            values = [by_sample[str(sample_id)][key] for sample_id in ids]
            batch[key] = _json_safe(_mean_list(values))
            # Match the runner's JSON-encoded arrays, including NaN entries;
            # its resume/aggregation code expects numbers rather than nulls.
            batch[f"{key}_values"] = json.dumps(values)
        batch.update({key: value for key, value in updates.items() if key in {
            "prediction_metrics_source", "relative_l2_solution_scope"
        }})

    updated_summary = {**summary, **updates}
    replacements = {raw_path: _jsonl_text(raw_rows)}
    # Preserve other runs in shared JSONL/CSV files, and retain all original
    # timing/provenance/physics fields in records for this run.
    for path in directory.glob("*_summary.json"):
        item = json.loads(read(path))
        if isinstance(item, dict) and str(item.get("run_id", "")) == run_id:
            replacements[path] = json.dumps({**item, **updates}, indent=2, sort_keys=True) + "\n"
    jsonl_path = directory / "results_summary.jsonl"
    summary_rows = _jsonl(read(jsonl_path)) if jsonl_path.is_file() else [summary]
    found = any(str(item.get("run_id", "")) == run_id for item in summary_rows)
    summary_rows = [
        {**item, **updates} if str(item.get("run_id", "")) == run_id else item
        for item in summary_rows
    ]
    if not found:
        summary_rows.append(updated_summary)
    replacements[jsonl_path] = _jsonl_text(summary_rows)
    csv_paths = {directory / "results_summary.csv", *directory.glob("*results_summary_latest.csv")}
    for path in sorted(csv_paths):
        items = list(csv.DictReader(io.StringIO(read(path)))) if path.is_file() else []
        found = False
        for item in items:
            if item.get("run_id", "") == run_id:
                item.update(updates)
                found = True
        if not found:
            items.append(updated_summary)
        columns = list(dict.fromkeys(key for item in items for key in item))
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(items)
        replacements[path] = output.getvalue()
    replacements[summary_path] = json.dumps(updated_summary, indent=2, sort_keys=True) + "\n"

    if not dry_run:
        # Complete validation and stage every replacement before publishing;
        # the canonical summary is committed last. Repeating this is idempotent.
        staged: list[tuple[Path, Path]] = []
        try:
            for path, text in replacements.items():
                temporary = path.with_name(f".{path.name}.recompute-{os.getpid()}.tmp")
                staged.append((temporary, path))
                temporary.write_text(text, encoding="utf-8")
            if any(path.read_text(encoding="utf-8") != text for path, text in originals.items()):
                raise ValueError("evaluation files changed during recomputation; no results were replaced")
            for temporary, path in staged:
                temporary.replace(path)
        finally:
            for temporary, _path in staged:
                temporary.unlink(missing_ok=True)
    return {
        "run_dir": str(directory),
        "samples": count,
        "dry_run": dry_run,
        "relative_l2_solution_mean_before": summary.get("relative_l2_solution_mean"),
        "relative_l2_solution_mean_after": updates["relative_l2_solution_mean"],
        "metrics": metrics,
    }


def select_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        return list(dict.fromkeys(path.expanduser().resolve() for path in args.run_dir))
    root = Path(args.output_root).expanduser().resolve() if args.output_root else detect_output_root()
    matrix_path = Path(args.matrix).expanduser().resolve() if args.matrix else root / "matrices/main_results.jsonl"
    matrix = load_matrix(matrix_path)
    pdes = split_selection(args.pde) or list(dict.fromkeys(row["pde"] for row in matrix))
    distributions = split_selection(args.distributions)
    if not distributions or set(distributions) - {"main", "smooth", "id", "rough"}:
        raise ValueError("distributions must be selected from main,smooth,id,rough")
    rows = filter_rows(matrix, pdes=pdes, baselines=split_selection(args.baselines))
    if not rows:
        raise ValueError("no matrix rows match the selected PDEs/baselines")
    directories = []
    for row in rows:
        for distribution in distributions:
            if distribution == "main":
                directory = source_run_dir(row, root / "runs/main_results", root)
            else:
                directory = (
                    root / "runs/evaluations" / distribution
                    / f"task_group={row['task_group']}" / f"pde={row['pde']}"
                    / f"baseline={row['baseline']}" / f"seed={row['seed']}"
                    / f"run=eval_{distribution}_{row['run_id']}"
                )
            directories.append(directory)
    return list(dict.fromkeys(directories))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", default=[], help="Recompute a specific run; repeatable.")
    parser.add_argument("--output-root", default=os.environ.get("OUT_ROOT") or os.environ.get("FM_OUTPUT_ROOT", ""))
    parser.add_argument("--matrix", help="Source experiment matrix (default: OUTPUT_ROOT/matrices/main_results.jsonl).")
    parser.add_argument("--pde", action="append", default=[], help="PDE filter; repeatable or comma-separated.")
    parser.add_argument("--baselines", default="", help="Comma-separated method filter.")
    parser.add_argument("--distributions", default="main,smooth,id,rough")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent sample readers (default: 4).")
    parser.add_argument("--dry-run", action="store_true", help="Verify samples and calculate metrics without writing files.")
    args = parser.parse_args(argv)
    directories = select_run_dirs(args)
    for index, directory in enumerate(directories, start=1):
        print(f"[recompute_metrics] {index}/{len(directories)} {directory}", flush=True)
        result = recompute_run(directory, dry_run=args.dry_run, workers=args.workers, progress=lambda text: print(text, flush=True))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"[recompute_metrics] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc
