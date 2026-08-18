#!/usr/bin/env python
"""Render one PDF per stored evaluation sample."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.common.sample_artifacts import render_evaluation_sample_pdf
from scripts.experiments.run_one import load_matrix_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--matrix", type=Path, help="Experiment matrix whose completed runs should be plotted.")
    source.add_argument("--manifest", type=Path, help="One stored sample manifest to plot.")
    parser.add_argument("--output-dir", type=Path, help="PDF directory for --manifest; defaults beside the sample directory.")
    parser.add_argument("--force", action="store_true", help="Replace complete PDFs instead of skipping them.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Plot at most the first N samples from each run; 0 means all (default: 0).",
    )
    return parser.parse_args(argv)


def _pdf_complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 8:
        return False
    with path.open("rb") as handle:
        if handle.read(4) != b"%PDF":
            return False
        handle.seek(max(path.stat().st_size - 2048, 0))
        return b"%%EOF" in handle.read()


def _matrix_jobs(matrix: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    jobs: list[tuple[Path, Path]] = []
    errors: list[str] = []
    for row in load_matrix_rows(matrix):
        if row.get("skip_reason"):
            continue
        run_id = str(row.get("run_id", ""))
        summary_path = Path(str(row.get("output_dir", ""))) / "summary.json"
        if not summary_path.is_file():
            errors.append(f"summary missing for {run_id}: {summary_path}")
            continue
        try:
            summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"summary unreadable for {run_id}: {exc}")
            continue
        manifest = Path(str(summary.get("sample_manifest_path", "")))
        if not manifest.is_file():
            errors.append(f"sample manifest missing for {run_id}: {manifest}")
            continue
        jobs.append((manifest, manifest.parent.with_name(f"{manifest.parent.name}_pdf")))
    return jobs, errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_manifest(
    manifest: Path,
    output_dir: Path,
    *,
    force: bool,
    max_samples: int,
) -> dict[str, Any]:
    all_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all_rows:
        raise ValueError(f"sample manifest is empty: {manifest}")
    rows = all_rows[:max_samples] if max_samples > 0 else all_rows
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    skipped = 0
    errors: list[str] = []
    plot_records: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        ordinal = int(row.get("sample_ordinal", position))
        target = output_dir / f"sample_{ordinal:06d}.pdf"
        if not force and _pdf_complete(target):
            skipped += 1
        else:
            temporary = output_dir / f".sample_{ordinal:06d}.tmp.pdf"
            try:
                render_evaluation_sample_pdf(
                    row["artifact_path"],
                    temporary,
                    expected_sha256=str(row.get("artifact_sha256", "")) or None,
                )
                temporary.replace(target)
                rendered += 1
            except Exception as exc:
                errors.append(f"sample {ordinal}: {exc}")
                continue
        plot_records.append(
            {
                "sample_ordinal": ordinal,
                "global_sample_id": str(row.get("global_sample_id", "")),
                "artifact_path": str(row.get("artifact_path", "")),
                "pdf_path": str(target.resolve()),
                "pdf_sha256": _sha256(target),
            }
        )
    plot_manifest = output_dir / "manifest.jsonl"
    temporary_manifest = output_dir / ".manifest.tmp.jsonl"
    temporary_manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in plot_records),
        encoding="utf-8",
    )
    temporary_manifest.replace(plot_manifest)
    return {
        "source_sample_count": len(all_rows),
        "sample_count": len(rows),
        "rendered_count": rendered,
        "skipped_count": skipped,
        "pdf_dir": str(output_dir),
        "pdf_manifest": str(plot_manifest),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    if args.manifest is not None:
        manifest = args.manifest
        if not manifest.is_file():
            raise FileNotFoundError(f"sample manifest not found: {manifest}")
        jobs = [(manifest, args.output_dir or manifest.parent.with_name(f"{manifest.parent.name}_pdf"))]
        errors: list[str] = []
    else:
        if args.output_dir is not None:
            raise ValueError("--output-dir is only valid with --manifest")
        jobs, errors = _matrix_jobs(args.matrix)

    outputs: list[dict[str, Any]] = []
    for manifest, output_dir in jobs:
        print(f"[plot_results] render {manifest} -> {output_dir}", file=sys.stderr, flush=True)
        try:
            result = _render_manifest(
                manifest,
                output_dir,
                force=args.force,
                max_samples=args.max_samples,
            )
            outputs.append(result)
            errors.extend(result["errors"])
        except Exception as exc:  # Continue so one broken run does not hide the rest.
            errors.append(f"render failed for {manifest}: {exc}")

    report = {
        "jobs": len(jobs),
        "outputs": outputs,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
