from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TIME_DEPENDENT_PDES = {
    "nsnonbounded",
    "burger",
    "reaction_diffusion",
    "shallow_water",
    "heat",
    "wave",
    "advection_diffusion",
}


def compatibility_reason(baseline: str, pde: str, task: str) -> str:
    baseline = baseline.lower()
    pde = pde.lower()
    task = task.lower()
    if baseline in {"var4d", "vivid"} and pde not in TIME_DEPENDENT_PDES:
        return f"{baseline} is restricted to time-dependent PDEs in this matrix"
    if baseline == "ifno" and task.startswith("sparse"):
        return "iFNO is evaluated on full forward/inverse tasks, not sparse reconstruction"
    return ""


def is_supported(baseline: str, pde: str, task: str) -> bool:
    return compatibility_reason(baseline, pde, task) == ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("FM4PDE baseline/PDE compatibility filter")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--pde", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--skipped-path", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    reason = compatibility_reason(args.baseline, args.pde, args.task)
    if reason:
        row: dict[str, Any] = {"baseline": args.baseline, "pde": args.pde, "task": args.task, "reason": reason}
        if args.skipped_path:
            path = Path(args.skipped_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
