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

FULL_TRAJECTORY_SENSOR_PDES = {"nsnonbounded", "burger", "reaction_diffusion", "shallow_water"}
PER_INSTANCE_BASELINES = {"pinn_sparse", "pc_bnn", "pde_opt", "var4d", "vivid"}


def compatibility_reason(baseline: str, pde: str, task: str, sensor_mode: str = "") -> str:
    baseline = baseline.lower()
    pde = pde.lower()
    task = task.lower()
    sensor_mode = sensor_mode.lower()
    if baseline in {"var4d", "vivid"} and pde not in TIME_DEPENDENT_PDES:
        return f"{baseline} is restricted to time-dependent PDEs in this matrix"
    if baseline == "ifno" and task.startswith("sparse"):
        return "iFNO is evaluated on full forward/inverse tasks, not sparse reconstruction"
    if task == "sparse_inverse" and baseline in PER_INSTANCE_BASELINES:
        return f"{baseline} sparse_inverse would require a PDE forward solve from predicted coefficient/initial state to sensor observations"
    if sensor_mode == "time_varying":
        if task not in {"sparse_solution", "sparse_reconstruction"}:
            return "time_varying sensors are only defined for full trajectory sparse reconstruction/DA tasks"
        if pde not in FULL_TRAJECTORY_SENSOR_PDES:
            return f"time_varying sensors require explicit trajectory targets; {pde} uses final-state targets in this matrix"
    return ""


def is_supported(baseline: str, pde: str, task: str) -> bool:
    return compatibility_reason(baseline, pde, task) == ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("FM4PDE baseline/PDE compatibility filter")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--pde", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--sensor-mode", default="")
    parser.add_argument("--skipped-path", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    reason = compatibility_reason(args.baseline, args.pde, args.task, args.sensor_mode)
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
