#!/usr/bin/env python
"""Build one experiment matrix from a declarative YAML design."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.build_matrix import main as build_main


DEFAULT_CONFIG = "configs/experiments/main_results.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=os.environ.get("OUT_ROOT", "outputs/main_results"))
    parser.add_argument("--matrix-name", default="")
    parser.add_argument("--data-manifest", default=os.environ.get("DATA_MANIFEST", ""))
    parser.add_argument("--include-skipped", action="store_true")
    parser.add_argument("--comparison-track", choices=["unified_adapted", "official_native"], default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    protocol = str(config.get("task_protocol_version", ""))
    if protocol.startswith("fm4pde-task-contract-") and not args.data_manifest:
        raise ValueError(
            "--data-manifest is required for a formal FM4PDE matrix; run "
            "scripts/verify_data_protocol.py --full first"
        )
    matrix_name = args.matrix_name or str(Path(args.config).stem)
    forwarded = [
        "--config",
        args.config,
        "--output-root",
        args.output_root,
        "--matrix-name",
        matrix_name,
    ]
    if args.data_manifest:
        forwarded.extend(["--data-manifest", args.data_manifest])
    if args.include_skipped:
        forwarded.append("--include-skipped")
    if args.comparison_track:
        forwarded.extend(["--comparison-track", args.comparison_track])
    build_main(forwarded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
