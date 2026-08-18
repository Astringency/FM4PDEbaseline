from __future__ import annotations

import argparse
from pathlib import Path

from baselines.common.sample_artifacts import render_sample_manifest_pdf


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Reload FM4PDE evaluation samples and render a PDF")
    parser.add_argument("manifest", help="Path to a saved evaluation sample manifest.jsonl")
    parser.add_argument("--output", default="", help="Output PDF path (default: next to the manifest)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = Path(args.manifest)
    output = Path(args.output) if args.output else manifest.with_name("samples_reloaded.pdf")
    rendered = render_sample_manifest_pdf(manifest, output)
    print(rendered)


if __name__ == "__main__":
    main()
