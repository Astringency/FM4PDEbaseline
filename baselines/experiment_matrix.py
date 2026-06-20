from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from baselines.capabilities import (
    ALL_BASELINES,
    ALL_PDES,
    FUTURE_PDES,
    TIME_DEPENDENT_PDES,
    TIME_VARYING_DA_PDES,
    Capability,
    iter_capability_matrix,
    resolve_capability,
    write_capability_matrix,
)


FULL_OPERATOR_BASELINES = {"fno", "deeponet", "ifno"}
SPARSE_AMORTIZED_BASELINES = {"recfno", "senseiver", "voronoicnn"}
SPARSE_INVERSE_BASELINES = {"pinn_sparse", "pde_opt"}
TIME_VARYING_SENSOR_BASELINES = {"var4d", "vivid", "senseiver"}
PER_INSTANCE_BASELINES = {"pinn_sparse", "pc_bnn", "pde_opt", "var4d", "vivid"}
FULL_TRAJECTORY_SENSOR_PDES = set(TIME_VARYING_DA_PDES)


def compatibility_reason(
    baseline: str,
    pde: str,
    task: str,
    sensor_mode: str = "",
    task_group: str = "",
    *,
    load_full_trajectory: bool | None = None,
    train_inverse_operator: bool | None = None,
    uses_official_inverse_observation_operator: bool | None = None,
    main_table_only: bool = False,
) -> str:
    capability = resolve_capability(
        baseline,
        pde,
        task,
        sensor_mode,
        task_group,
        load_full_trajectory=load_full_trajectory,
        train_inverse_operator=train_inverse_operator,
        uses_official_inverse_observation_operator=uses_official_inverse_observation_operator,
    )
    if capability.support_status == "unsupported":
        return capability.reason
    if main_table_only and not capability.paper_table_eligible:
        return main_table_skip_reason(capability)
    return ""


def capability_skip_row(
    capability: Capability,
    *,
    task_group: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "baseline": capability.baseline,
        "pde": capability.pde,
        "task": capability.task,
        "sensor_mode": capability.sensor_mode,
        "task_group": task_group,
        "capability_status": capability.support_status,
        "implementation_required": capability.implementation_required,
        "task_family": capability.task_family,
        "reason": capability.reason,
        "unsupported_reason": capability.unsupported_reason,
        "citation_key": capability.citation_key,
        "source_key": capability.source_key,
        "notes_for_paper": capability.notes_for_paper,
        "official_architecture_allowed": capability.official_architecture_allowed,
        "official_aligned_allowed": capability.official_aligned_allowed,
        "eligible_implementation_modes": list(capability.eligible_implementation_modes),
        "paper_table_eligible": capability.paper_table_eligible,
    }
    if extra:
        row.update(extra)
    return row


def main_table_skip_reason(capability: Capability) -> str:
    if capability.support_status == "unsupported":
        return capability.reason
    if not capability.paper_table_eligible:
        return (
            f"{capability.baseline}/{capability.task_family} is {capability.support_status} "
            f"({capability.implementation_required}) and is supplement-only for paper main-table mode: {capability.reason}"
        )
    return ""


def is_supported(
    baseline: str,
    pde: str,
    task: str,
    sensor_mode: str = "",
    task_group: str = "",
    *,
    main_table_only: bool = False,
) -> bool:
    return compatibility_reason(baseline, pde, task, sensor_mode, task_group, main_table_only=main_table_only) == ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("FM4PDE baseline/PDE compatibility filter")
    parser.add_argument("--baseline")
    parser.add_argument("--pde")
    parser.add_argument("--task")
    parser.add_argument("--sensor-mode", default="")
    parser.add_argument("--task-group", default="")
    parser.add_argument("--skipped-path", default="")
    parser.add_argument("--dump-matrix", action="store_true", help="Write the complete native capability matrix as CSV and JSON.")
    parser.add_argument("--output", default="outputs/baselines/capability_matrix")
    parser.add_argument("--load-full-trajectory", action="store_true")
    parser.add_argument("--train-inverse-operator", action="store_true")
    parser.add_argument("--uses-official-inverse-observation-operator", action="store_true")
    parser.add_argument("--main-table-only", action="store_true", help="Treat adapted/supplement-only capabilities as skipped.")
    parser.add_argument("--paper-mode", action="store_true", help="Alias for --main-table-only.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dump_matrix:
        csv_path, json_path = write_capability_matrix(args.output, iter_capability_matrix())
        print(json.dumps({"csv": str(csv_path), "json": str(json_path)}, indent=2, sort_keys=True))
        return
    missing = [name for name in ("baseline", "pde", "task") if not getattr(args, name)]
    if missing:
        raise SystemExit(f"Missing required arguments for compatibility check: {', '.join(missing)}")
    capability = resolve_capability(
        args.baseline,
        args.pde,
        args.task,
        args.sensor_mode,
        args.task_group,
        load_full_trajectory=bool(args.load_full_trajectory) if args.load_full_trajectory else None,
        train_inverse_operator=bool(args.train_inverse_operator) if args.train_inverse_operator else None,
        uses_official_inverse_observation_operator=(
            bool(args.uses_official_inverse_observation_operator)
            if args.uses_official_inverse_observation_operator
            else None
        ),
    )
    if capability.support_status == "unsupported":
        row = capability_skip_row(capability, task_group=args.task_group)
        if args.skipped_path:
            path = Path(args.skipped_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        raise SystemExit(1)
    if (args.main_table_only or args.paper_mode) and not capability.paper_table_eligible:
        row = capability_skip_row(
            capability,
            task_group=args.task_group,
            extra={"reason": main_table_skip_reason(capability), "unsupported_reason": main_table_skip_reason(capability)},
        )
        if args.skipped_path:
            path = Path(args.skipped_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
