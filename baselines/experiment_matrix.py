from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALL_PDES = [
    "darcy",
    "poisson",
    "helmholtz",
    "nsnonbounded",
    "burger",
    "reaction_diffusion",
    "shallow_water",
    "heat",
    "wave",
    "advection_diffusion",
    "steady_heat_conduction",
]

FUTURE_PDES = {"heat", "wave", "advection_diffusion", "steady_heat_conduction"}

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
FULL_OPERATOR_BASELINES = {"fno", "deeponet", "ifno"}
SPARSE_AMORTIZED_BASELINES = {"recfno", "senseiver", "voronoicnn", "fno", "deeponet"}
SPARSE_INVERSE_BASELINES = {"recfno", "senseiver", "voronoicnn", "fno", "deeponet"}
TIME_VARYING_SENSOR_BASELINES = {"var4d", "vivid", "senseiver"}
PER_INSTANCE_BASELINES = {"pinn_sparse", "pc_bnn", "pde_opt", "var4d", "vivid"}
ALL_BASELINES = sorted(
    FULL_OPERATOR_BASELINES
    | SPARSE_AMORTIZED_BASELINES
    | SPARSE_INVERSE_BASELINES
    | TIME_VARYING_SENSOR_BASELINES
    | PER_INSTANCE_BASELINES
)


def compatibility_reason(baseline: str, pde: str, task: str, sensor_mode: str = "", task_group: str = "") -> str:
    baseline = baseline.lower()
    pde = pde.lower()
    task = task.lower()
    sensor_mode = sensor_mode.lower()
    task_group = task_group.lower()
    if pde in {"diffusionpde", "diffusion_pde", "diffusion"}:
        return "DiffusionPDE is not part of the FM4PDE external-baseline matrix"
    if baseline in {"fm4pde", "fm4pde_debug", "fm4pde-ablation", "fm4pde_ablation"}:
        return "FM4PDE main-model/internal-ablation runs are outside this external-baseline matrix"
    if pde not in set(ALL_PDES):
        return f"unknown or unregistered PDE '{pde}' for this matrix"
    if baseline not in set(ALL_BASELINES):
        return f"unknown or unregistered baseline '{baseline}' for this matrix"
    if task in {"forward", "inverse"} and baseline not in FULL_OPERATOR_BASELINES:
        return f"{baseline} is not a full-operator baseline for {task}"
    if baseline in {"var4d", "vivid"} and pde not in TIME_DEPENDENT_PDES:
        return f"{baseline} is restricted to time-dependent PDEs in this matrix"
    if baseline == "ifno" and task.startswith("sparse"):
        return "iFNO is evaluated on full forward/inverse tasks, not sparse reconstruction"
    if task == "sparse_inverse" and baseline in PER_INSTANCE_BASELINES:
        return f"{baseline} sparse_inverse would require a PDE forward solve from predicted coefficient/initial state to sensor observations"
    if task == "sparse_inverse" and baseline not in SPARSE_INVERSE_BASELINES:
        return f"{baseline} is not enabled for sparse_inverse in this matrix"
    if sensor_mode == "time_varying":
        if task not in {"sparse_solution", "sparse_reconstruction"}:
            return "time_varying sensors are only defined for full trajectory sparse reconstruction/DA tasks"
        if pde not in FULL_TRAJECTORY_SENSOR_PDES:
            return f"time_varying sensors require explicit trajectory targets; {pde} uses final-state targets in this matrix"
        if baseline not in TIME_VARYING_SENSOR_BASELINES:
            return f"{baseline} is not adapted to 5D trajectory targets with time_varying sensors"
    if task_group == "time_varying" and sensor_mode != "time_varying":
        return "time_varying task_group requires sensor_mode=time_varying"
    return ""


def is_supported(baseline: str, pde: str, task: str, sensor_mode: str = "", task_group: str = "") -> bool:
    return compatibility_reason(baseline, pde, task, sensor_mode, task_group) == ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("FM4PDE baseline/PDE compatibility filter")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--pde", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--sensor-mode", default="")
    parser.add_argument("--task-group", default="")
    parser.add_argument("--skipped-path", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    reason = compatibility_reason(args.baseline, args.pde, args.task, args.sensor_mode, args.task_group)
    if reason:
        row: dict[str, Any] = {
            "baseline": args.baseline,
            "pde": args.pde,
            "task": args.task,
            "sensor_mode": args.sensor_mode,
            "task_group": args.task_group,
            "reason": reason,
        }
        if args.skipped_path:
            path = Path(args.skipped_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
