#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.experiment_matrix import (
    ALL_BASELINES,
    ALL_PDES,
    FUTURE_PDES,
    PER_INSTANCE_BASELINES,
    TIME_VARYING_SENSOR_BASELINES,
    capability_skip_row,
    compatibility_reason,
    main_table_skip_reason,
    resolve_capability,
)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def progress(message: str) -> None:
    print(f"[{timestamp()}] {message}", file=sys.stderr, flush=True)


GROUP_TO_TASK = {
    "full_forward_main": "forward",
    "full_inverse_main": "inverse",
    "sparse_solution_main_amortized": "sparse_solution",
    "sparse_solution_main_physics": "sparse_solution",
    "sparse_inverse_main": "sparse_inverse",
    "time_varying_da_main": "sparse_solution",
    "sensor_count_ablation": "sparse_solution",
    "noise_ablation": "sparse_solution",
    "sensor_mode_ablation": "sparse_solution",
    "time_varying_sensor_ablation": "sparse_solution",
    "runtime_budget_ablation": "sparse_solution",
    "train_size_ablation": "sparse_solution",
}

DEFAULT_BASELINES_BY_GROUP = {
    "full_forward_main": ["fno", "deeponet", "ifno"],
    "full_inverse_main": ["fno", "deeponet", "ifno"],
    "sparse_solution_main_amortized": ["recfno", "senseiver", "voronoicnn"],
    "sparse_solution_main_physics": ["pinn_sparse", "pc_bnn", "pde_opt"],
    "sparse_inverse_main": ["pinn_sparse", "pde_opt"],
    "time_varying_da_main": ["senseiver", "var4d", "vivid"],
    "sensor_count_ablation": ["recfno", "senseiver", "voronoicnn", "pinn_sparse", "pde_opt"],
    "noise_ablation": ["recfno", "senseiver", "voronoicnn", "pinn_sparse", "pde_opt"],
    "sensor_mode_ablation": ["recfno", "senseiver", "voronoicnn", "pde_opt"],
    "time_varying_sensor_ablation": ["var4d", "vivid", "senseiver"],
    "runtime_budget_ablation": ["pinn_sparse", "pc_bnn", "pde_opt", "var4d", "vivid"],
    "train_size_ablation": ["fno", "deeponet", "recfno", "senseiver", "voronoicnn"],
}

VALID_EXPERIMENT_KINDS = {"main", "ablation"}
VALID_ABLATION_FACTORS = {
    "sensor_count",
    "noise_level",
    "sensor_mode",
    "time_varying_sensor_count",
    "runtime_budget",
    "train_size",
}
FACTOR_TO_VARIED_FIELDS = {
    "sensor_count": {"num_sensors"},
    "noise_level": {"noise_level"},
    "sensor_mode": {"sensor_mode"},
    "time_varying_sensor_count": {"num_sensors"},
    "runtime_budget": {"steps", "refine_steps", "particles"},
    "train_size": {"train_size"},
}

HASH_FIELDS = [
    "experiment_kind",
    "ablation_factor",
    "task_group",
    "task",
    "pde",
    "baseline",
    "seed",
    "train_size",
    "test_size",
    "num_sensors",
    "sensor_mode",
    "noise_level",
    "steps",
    "refine_steps",
    "particles",
    "scalar_param_mode",
    "data_loading_mode",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "prefetch_factor",
    "load_full_trajectory",
]

MATRIX_FIELDS = [
    "run_id",
    "run_name",
    "experiment_kind",
    "ablation_factor",
    "task_group",
    "task",
    "pde",
    "baseline",
    "seed",
    "train_size",
    "val_size",
    "test_size",
    "train_shards",
    "num_sensors",
    "sensor_mode",
    "noise_level",
    "scalar_param_mode",
    "data_loading_mode",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "prefetch_factor",
    "load_full_trajectory",
    "batch_size",
    "epochs",
    "steps",
    "refine_steps",
    "particles",
    "device",
    "config",
    "output_dir",
    "log_dir",
    "status_file",
    "skip_reason",
]

SUMMARY_DESIGN_FIELDS = [
    "task_group",
    "task",
    "pde",
    "baseline",
    "seed",
    "train_size",
    "num_sensors",
    "sensor_mode",
    "noise_level",
    "steps",
    "refine_steps",
    "particles",
    "load_full_trajectory",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build large-scale FM4PDE external-baseline experiment matrices.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default=os.environ.get("OUT_ROOT", "outputs/baselines_large"))
    parser.add_argument("--matrix-name", default="")
    parser.add_argument("--include-skipped", action="store_true", help="Also include unsupported rows in the matrix with skip_reason set.")
    parser.add_argument("--main-table-only", action="store_true", help="Skip adapted/supplement-only capabilities at matrix generation time.")
    parser.add_argument("--paper-mode", action="store_true", help="Alias for --main-table-only.")
    return parser.parse_args(argv)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_matrix(
    cfg: dict[str, Any],
    output_root: str | Path,
    matrix_name: str,
    include_skipped: bool = False,
    main_table_only: bool | None = None,
    emit_progress: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    output_root = Path(output_root)
    experiment_kind = _experiment_kind(cfg)
    ablation_factor = _ablation_factor(cfg, experiment_kind)
    if main_table_only is None:
        main_table_only = bool(cfg.get("main_table_only", False))
    allow_multi = _as_bool(cfg.get("allow_multi_factor_grid", False))
    task_groups = list(cfg.get("task_groups", []))
    if not task_groups:
        raise ValueError("config must define at least one task_group")

    rows: list[dict[str, Any]] = []
    skipped: dict[tuple[Any, ...], dict[str, Any]] = {}
    global_defaults = _global_defaults(cfg)

    for task_group in task_groups:
        if task_group not in GROUP_TO_TASK:
            raise ValueError(f"Unknown task_group {task_group!r}")
        group_cfg = dict(cfg.get("task_group_overrides", {}).get(task_group, {}) or {})
        task = str(group_cfg.get("task", GROUP_TO_TASK[task_group]))
        _validate_group_design(
            cfg=cfg,
            group_cfg=group_cfg,
            defaults=global_defaults,
            experiment_kind=experiment_kind,
            ablation_factor=ablation_factor,
            task_group=task_group,
            task=task,
            allow_multi=allow_multi,
        )
        pdes = _resolve_pdes(cfg, group_cfg, experiment_kind, ablation_factor)
        baselines = _resolve_baselines(cfg, task_group, task, group_cfg, experiment_kind, ablation_factor)
        extra_skip_baselines = list(group_cfg.get("extra_skip_baselines", []) or [])
        candidate_baselines = list(baselines)
        seeds = _env_list("SEEDS", group_cfg.get("seeds", global_defaults["seeds"]), int)
        expansion = _group_expansion(cfg, group_cfg, global_defaults, task_group, task)
        if emit_progress:
            progress(
                f"[matrix group] task_group={task_group} task={task} pdes={len(pdes)} "
                f"baselines={_short_list(baselines)} sensor_modes={_short_list(expansion['sensor_modes'])} "
                f"sensor_counts={_short_list(expansion['sensor_counts'])} noise_levels={_short_list(expansion['noise_levels'])} "
                f"seeds={_short_list(seeds)} train_sizes={_short_list(expansion['train_sizes'])}"
            )

        for pde in pdes:
            for baseline in extra_skip_baselines:
                if baseline in baselines:
                    continue
                budget_variants = _budget_variants(cfg, group_cfg, baseline, ablation_factor, experiment_kind)
                for sensor_mode in expansion["sensor_modes"]:
                    skipped_row = {
                        "matrix_name": matrix_name,
                        "experiment_kind": experiment_kind,
                        "ablation_factor": ablation_factor,
                        "task_group": task_group,
                        "task": task,
                        "pde": pde,
                        "baseline": baseline,
                        "sensor_mode": sensor_mode,
                        "reason": "baseline is intentionally skipped for this task_group",
                        "would_have_expanded": (
                            len(seeds)
                            * len(expansion["sensor_counts"])
                            * len(expansion["noise_levels"])
                            * len(expansion["train_sizes"])
                            * len(budget_variants)
                        ),
                    }
                    _merge_skip(skipped, skipped_row)
                    if include_skipped:
                        rows.append(_skipped_matrix_row(skipped_row, global_defaults, output_root))
            for baseline in candidate_baselines:
                budget_variants = _budget_variants(cfg, group_cfg, baseline, ablation_factor, experiment_kind)
                for sensor_mode in expansion["sensor_modes"]:
                    capability = resolve_capability(
                        baseline,
                        pde,
                        task,
                        "" if sensor_mode == "none" else sensor_mode,
                        task_group,
                        load_full_trajectory=_matrix_load_full_trajectory(group_cfg, global_defaults, baseline),
                        train_inverse_operator=_matrix_train_inverse_operator(cfg, group_cfg, baseline),
                        uses_official_inverse_observation_operator=_matrix_uses_official_inverse_observation_operator(cfg, group_cfg, baseline),
                    )
                    reason = capability.reason if capability.support_status == "unsupported" else ""
                    if not reason and main_table_only and not capability.paper_table_eligible:
                        reason = main_table_skip_reason(capability)
                    if reason:
                        skipped_row = capability_skip_row(
                            capability,
                            task_group=task_group,
                            extra={
                                "matrix_name": matrix_name,
                                "experiment_kind": experiment_kind,
                                "ablation_factor": ablation_factor,
                                "reason": reason,
                                "unsupported_reason": reason if capability.support_status == "unsupported" else "",
                                "would_have_expanded": (
                                    len(seeds)
                                    * len(expansion["sensor_counts"])
                                    * len(expansion["noise_levels"])
                                    * len(expansion["train_sizes"])
                                    * len(budget_variants)
                                ),
                            },
                        )
                        _merge_skip(skipped, skipped_row)
                        if include_skipped:
                            rows.append(_skipped_matrix_row(skipped_row, global_defaults, output_root))
                        continue
                    for seed in seeds:
                        for train_size in expansion["train_sizes"]:
                            for num_sensors in expansion["sensor_counts"]:
                                for noise_level in expansion["noise_levels"]:
                                    for budget in budget_variants:
                                        row = _make_run_row(
                                            cfg=cfg,
                                            group_cfg=group_cfg,
                                            defaults=global_defaults,
                                            output_root=output_root,
                                            matrix_name=matrix_name,
                                            experiment_kind=experiment_kind,
                                            ablation_factor=ablation_factor,
                                            task_group=task_group,
                                            task=task,
                                            pde=str(pde),
                                            baseline=str(baseline),
                                            seed=int(seed),
                                            train_size=int(train_size),
                                            num_sensors=int(num_sensors),
                                            sensor_mode=str(sensor_mode),
                                            noise_level=float(noise_level),
                                            budget=budget,
                                        )
                                        rows.append(row)

    _ensure_unique_run_ids(rows)
    skipped_rows = list(skipped.values())
    summary = _summary(rows, skipped_rows, matrix_name, experiment_kind, ablation_factor)
    return rows, skipped_rows, summary


def write_outputs(rows: list[dict[str, Any]], skipped_rows: list[dict[str, Any]], summary: dict[str, Any], output_root: str | Path, matrix_name: str) -> None:
    output_root = Path(output_root)
    matrix_dir = output_root / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = matrix_dir / f"{matrix_name}.jsonl"
    tsv_path = matrix_dir / f"{matrix_name}.tsv"
    summary_path = matrix_dir / f"{matrix_name}_summary.json"
    skipped_matrix_path = matrix_dir / f"{matrix_name}_skipped.jsonl"
    skipped_global_path = output_root / "skipped_combinations.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    _write_tsv(tsv_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    with skipped_matrix_path.open("w", encoding="utf-8") as f:
        for row in skipped_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    existing_global = _read_jsonl(skipped_global_path)
    preserved_global = [row for row in existing_global if str(row.get("matrix_name", "")) != matrix_name]
    skipped_global_path.parent.mkdir(parents=True, exist_ok=True)
    with skipped_global_path.open("w", encoding="utf-8") as f:
        for row in preserved_global + skipped_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _experiment_kind(cfg: dict[str, Any]) -> str:
    kind = str(cfg.get("experiment_kind", "") or "")
    if kind not in VALID_EXPERIMENT_KINDS:
        raise ValueError(f"experiment_kind must be one of {sorted(VALID_EXPERIMENT_KINDS)}, got {kind!r}")
    return kind


def _ablation_factor(cfg: dict[str, Any], experiment_kind: str) -> str:
    factor = str(cfg.get("ablation_factor", "") or "")
    if experiment_kind == "main":
        if factor:
            raise ValueError("main experiments must not set ablation_factor")
        return ""
    if factor not in VALID_ABLATION_FACTORS:
        raise ValueError(f"ablation_factor must be one of {sorted(VALID_ABLATION_FACTORS)}, got {factor!r}")
    return factor


def _global_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    train_size = _env_int("TRAIN_SIZE", int(cfg.get("train_size", 50000)))
    return {
        "seeds": _env_list("SEEDS", cfg.get("seeds", [1, 2, 3]), int),
        "train_size": train_size,
        "val_size": _env_int("VAL_SIZE", int(cfg.get("val_size", 0))),
        "test_size": _env_int("TEST_SIZE", int(cfg.get("test_size", 10000))),
        "train_shards": _env_int("TRAIN_SHARDS", int(cfg.get("train_shards", 5))),
        "sensor_counts": _env_list("SENSOR_COUNTS", _first_present(cfg, ["sensor_counts", "num_sensors", "sensor_count"], [500]), int),
        "sensor_modes": _env_list("SENSOR_MODES", _first_present(cfg, ["sensor_modes", "sensor_mode"], ["random"]), str),
        "noise_levels": _env_list("NOISE_LEVELS", _first_present(cfg, ["noise_levels", "noise_level"], [0.0]), float),
        "train_sizes": _env_list("TRAIN_SIZES", cfg.get("train_sizes", [train_size]), int),
        "data_loading_mode": str(cfg.get("data_loading_mode", "eager")),
        "num_workers": _env_int("NUM_WORKERS", int(cfg.get("num_workers", 4))),
        "pin_memory": _env_bool("PIN_MEMORY", bool(cfg.get("pin_memory", True))),
        "persistent_workers": _env_bool("PERSISTENT_WORKERS", bool(cfg.get("persistent_workers", True))),
        "prefetch_factor": _env_int("PREFETCH_FACTOR", int(cfg.get("prefetch_factor", 2))),
        "device": os.environ.get("DEVICE", str(cfg.get("device", "cuda"))),
        "config": str(cfg.get("config", "baselines/configs/paper.yaml")),
        "scalar_param_mode": os.environ.get("SCALAR_PARAM_MODE", ""),
        "load_full_trajectory": bool(cfg.get("load_full_trajectory", False)),
    }


def _matrix_load_full_trajectory(group_cfg: dict[str, Any], defaults: dict[str, Any], baseline: str) -> bool:
    resources = group_cfg.get("resources", {}) if isinstance(group_cfg.get("resources", {}), dict) else {}
    baseline_resources = resources.get(baseline, {}) if isinstance(resources.get(baseline, {}), dict) else {}
    return bool(group_cfg.get("load_full_trajectory", baseline_resources.get("load_full_trajectory", defaults.get("load_full_trajectory", False))))


def _matrix_train_inverse_operator(cfg: dict[str, Any], group_cfg: dict[str, Any], baseline: str) -> bool:
    resources = dict(cfg.get("resources", {}) or {})
    baseline_resources = dict(resources.get(baseline, {}) or {})
    group_resources = dict(group_cfg.get("resources", {}) or {})
    group_baseline_resources = dict(group_resources.get(baseline, {}) or {})
    return bool(group_cfg.get("train_inverse_operator", group_baseline_resources.get("train_inverse_operator", baseline_resources.get("train_inverse_operator", False))))


def _matrix_uses_official_inverse_observation_operator(cfg: dict[str, Any], group_cfg: dict[str, Any], baseline: str) -> bool:
    resources = dict(cfg.get("resources", {}) or {})
    baseline_resources = dict(resources.get(baseline, {}) or {})
    group_resources = dict(group_cfg.get("resources", {}) or {})
    group_baseline_resources = dict(group_resources.get(baseline, {}) or {})
    if baseline == "vivid":
        return bool(
            group_cfg.get(
                "uses_official_inverse_observation_operator",
                group_baseline_resources.get(
                    "uses_official_inverse_observation_operator",
                    baseline_resources.get("uses_official_inverse_observation_operator", True),
                ),
            )
        )
    return bool(
        group_cfg.get(
            "uses_official_inverse_observation_operator",
            group_baseline_resources.get(
                "uses_official_inverse_observation_operator",
                baseline_resources.get("uses_official_inverse_observation_operator", False),
            ),
        )
    )


def _resolve_pdes(cfg: dict[str, Any], group_cfg: dict[str, Any], experiment_kind: str, ablation_factor: str) -> list[str]:
    if experiment_kind == "ablation" and _env_flag("FULL_ABLATION_ALL"):
        if ablation_factor == "time_varying_sensor_count":
            return list(ALL_PDES)
        return list(ALL_PDES)
    return _resolve_list(group_cfg.get("pdes", cfg.get("pdes", ALL_PDES)), ALL_PDES)


def _resolve_baselines(
    cfg: dict[str, Any],
    task_group: str,
    task: str,
    group_cfg: dict[str, Any],
    experiment_kind: str,
    ablation_factor: str,
) -> list[str]:
    if experiment_kind == "ablation" and _env_flag("FULL_ABLATION_ALL"):
        if ablation_factor == "runtime_budget":
            budget_cfg = group_cfg.get("budget_by_baseline", {}) or group_cfg.get("runtime_budget_by_baseline", {}) or {}
            if budget_cfg:
                return list(budget_cfg)
        if ablation_factor == "time_varying_sensor_count":
            return sorted(TIME_VARYING_SENSOR_BASELINES)
        return _all_baselines_for_task(task)
    return list(group_cfg.get("baselines", cfg.get("baselines", DEFAULT_BASELINES_BY_GROUP[task_group])))


def _all_baselines_for_task(task: str) -> list[str]:
    if task in {"forward", "inverse"}:
        return ["fno", "deeponet", "ifno"]
    if task == "sparse_inverse":
        return ["recfno", "senseiver", "voronoicnn", "fno", "deeponet", "pinn_sparse", "pc_bnn", "pde_opt", "var4d", "vivid"]
    return list(ALL_BASELINES)


def _group_expansion(cfg: dict[str, Any], group_cfg: dict[str, Any], defaults: dict[str, Any], task_group: str, task: str) -> dict[str, list[Any]]:
    sparse = task.startswith("sparse")
    if sparse:
        sensor_counts = _env_list("SENSOR_COUNTS", _first_present(group_cfg, ["sensor_counts", "num_sensors", "sensor_count"], defaults["sensor_counts"]), int)
        sensor_modes = _env_list("SENSOR_MODES", _first_present(group_cfg, ["sensor_modes", "sensor_mode"], defaults["sensor_modes"]), str)
        noise_levels = _env_list("NOISE_LEVELS", _first_present(group_cfg, ["noise_levels", "noise_level"], defaults["noise_levels"]), float)
    else:
        sensor_counts = [0]
        sensor_modes = ["none"]
        noise_levels = [0.0]
    train_sizes = _env_list("TRAIN_SIZES", group_cfg.get("train_sizes", cfg.get("train_sizes", [group_cfg.get("train_size", defaults["train_size"])])), int)
    if task_group == "time_varying_sensor_ablation":
        sensor_modes = ["time_varying"]
    return {
        "sensor_counts": sensor_counts,
        "sensor_modes": sensor_modes,
        "noise_levels": noise_levels,
        "train_sizes": train_sizes,
    }


def _validate_group_design(
    cfg: dict[str, Any],
    group_cfg: dict[str, Any],
    defaults: dict[str, Any],
    experiment_kind: str,
    ablation_factor: str,
    task_group: str,
    task: str,
    allow_multi: bool,
) -> None:
    expansion = _group_expansion(cfg, group_cfg, defaults, task_group, task)
    lengths = {
        "num_sensors": len(set(expansion["sensor_counts"])),
        "sensor_mode": len(set(expansion["sensor_modes"])),
        "noise_level": len(set(expansion["noise_levels"])),
        "train_size": len(set(expansion["train_sizes"])),
    }
    if experiment_kind == "main":
        offenders = {field: length for field, length in lengths.items() if field in {"num_sensors", "sensor_mode", "noise_level"} and length > 1}
        if offenders:
            raise ValueError(f"main task_group {task_group!r} must use one sensor/noise/mode setting, got {offenders}")
        return

    budget_varied = _runtime_budget_varied_fields(group_cfg)
    multi_fields = {field for field, length in lengths.items() if length > 1}
    multi_fields.update(budget_varied)
    allowed = FACTOR_TO_VARIED_FIELDS[ablation_factor]
    illegal = multi_fields - allowed
    if illegal and not allow_multi:
        raise ValueError(
            f"ablation task_group {task_group!r} with factor {ablation_factor!r} varies unsupported fields "
            f"{sorted(illegal)}; set allow_multi_factor_grid: true only for explicit exhaustive debugging"
        )
    if not allow_multi and len(multi_fields & {"num_sensors", "sensor_mode", "noise_level", "train_size", "steps", "refine_steps", "particles"}) > len(allowed):
        active = sorted(multi_fields & {"num_sensors", "sensor_mode", "noise_level", "train_size", "steps", "refine_steps", "particles"})
        raise ValueError(f"ablation task_group {task_group!r} varies multiple factors: {active}")
    if ablation_factor == "runtime_budget":
        _validate_runtime_budget_group(group_cfg, task_group, allow_multi)
    elif not allow_multi and not (multi_fields & allowed):
        raise ValueError(f"ablation task_group {task_group!r} does not vary its declared factor {ablation_factor!r}")


def _runtime_budget_varied_fields(group_cfg: dict[str, Any]) -> set[str]:
    varied: set[str] = set()
    budget_cfg = group_cfg.get("budget_by_baseline", {}) or group_cfg.get("runtime_budget_by_baseline", {}) or {}
    for baseline_cfg in budget_cfg.values():
        for field in ("steps", "refine_steps", "particles"):
            if len(_as_list(baseline_cfg.get(field, []))) > 1:
                varied.add(field)
    for field in ("steps", "refine_steps", "particles"):
        if len(_as_list(group_cfg.get(field, []))) > 1:
            varied.add(field)
    return varied


def _validate_runtime_budget_group(group_cfg: dict[str, Any], task_group: str, allow_multi: bool) -> None:
    budget_cfg = group_cfg.get("budget_by_baseline", {}) or group_cfg.get("runtime_budget_by_baseline", {}) or {}
    if not budget_cfg and not any(group_cfg.get(field) is not None for field in ("steps", "refine_steps", "particles")):
        raise ValueError(f"runtime_budget ablation {task_group!r} must define budget_by_baseline or a budget field")
    if allow_multi:
        return
    for baseline, baseline_cfg in budget_cfg.items():
        varied = [field for field in ("steps", "refine_steps", "particles") if len(_as_list(baseline_cfg.get(field, []))) > 1]
        if len(varied) != 1:
            raise ValueError(f"runtime_budget baseline {baseline!r} must vary exactly one budget field, got {varied}")
    top_varied = [field for field in ("steps", "refine_steps", "particles") if len(_as_list(group_cfg.get(field, []))) > 1]
    if top_varied and len(top_varied) != 1:
        raise ValueError(f"runtime_budget task_group {task_group!r} must not cross budget fields, got {top_varied}")


def _budget_variants(
    cfg: dict[str, Any],
    group_cfg: dict[str, Any],
    baseline: str,
    ablation_factor: str,
    experiment_kind: str,
) -> list[dict[str, int]]:
    resources = _resource_config(cfg, baseline)
    base = {
        "steps": _method_steps(baseline, resources),
        "refine_steps": _method_refine_steps(baseline, resources),
        "particles": _method_particles(baseline, resources),
    }
    if experiment_kind != "ablation" or ablation_factor != "runtime_budget":
        return [base]

    budget_cfg = group_cfg.get("budget_by_baseline", {}) or group_cfg.get("runtime_budget_by_baseline", {}) or {}
    selected = budget_cfg.get(baseline)
    if selected is None:
        selected = {field: group_cfg[field] for field in ("steps", "refine_steps", "particles") if field in group_cfg}
    if not selected:
        return [base]
    varied = [field for field in ("steps", "refine_steps", "particles") if len(_as_list(selected.get(field, []))) > 1]
    if len(varied) > 1 and not _as_bool(cfg.get("allow_multi_factor_grid", False)):
        raise ValueError(f"runtime_budget baseline {baseline!r} crosses budget fields: {varied}")
    if not varied:
        merged = dict(base)
        for field in ("steps", "refine_steps", "particles"):
            if field in selected:
                merged[field] = int(selected[field])
        return [merged]
    field = varied[0]
    variants = []
    for value in _as_list(selected[field]):
        merged = dict(base)
        for fixed_field in ("steps", "refine_steps", "particles"):
            if fixed_field in selected and fixed_field != field and not isinstance(selected[fixed_field], list):
                merged[fixed_field] = int(selected[fixed_field])
        merged[field] = int(value)
        variants.append(merged)
    return variants


def _make_run_row(
    cfg: dict[str, Any],
    group_cfg: dict[str, Any],
    defaults: dict[str, Any],
    output_root: Path,
    matrix_name: str,
    experiment_kind: str,
    ablation_factor: str,
    task_group: str,
    task: str,
    pde: str,
    baseline: str,
    seed: int,
    train_size: int,
    num_sensors: int,
    sensor_mode: str,
    noise_level: float,
    budget: dict[str, int],
) -> dict[str, Any]:
    resources = _resource_config(cfg, baseline)
    val_size = _env_int("VAL_SIZE", int(group_cfg.get("val_size", defaults["val_size"])))
    test_size = _env_int("TEST_SIZE", int(group_cfg.get("test_size", defaults["test_size"])))
    train_shards = _env_int("TRAIN_SHARDS", int(group_cfg.get("train_shards", defaults["train_shards"])))
    batch_size = _env_int("BATCH_SIZE", int(group_cfg.get("batch_size", resources.get("batch_size", 16))))
    epochs = _env_int("EPOCHS", int(group_cfg.get("epochs", resources.get("epochs", 1 if baseline in PER_INSTANCE_BASELINES else 200))))
    load_full_trajectory = bool(
        group_cfg.get(
            "load_full_trajectory",
            resources.get("load_full_trajectory", defaults.get("load_full_trajectory", False)),
        )
    )
    if sensor_mode == "time_varying" or task_group == "time_varying_sensor_ablation":
        load_full_trajectory = True
    scalar_param_mode = _scalar_param_mode(defaults, pde, task)
    row: dict[str, Any] = {
        "run_id": "",
        "run_name": "",
        "experiment_kind": experiment_kind,
        "ablation_factor": ablation_factor,
        "task_group": task_group,
        "task": task,
        "pde": pde,
        "baseline": baseline,
        "seed": seed,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "train_shards": train_shards,
        "num_sensors": int(num_sensors),
        "sensor_mode": sensor_mode,
        "noise_level": float(noise_level),
        "scalar_param_mode": scalar_param_mode,
        "data_loading_mode": defaults["data_loading_mode"],
        "num_workers": int(defaults["num_workers"]),
        "pin_memory": bool(defaults["pin_memory"]),
        "persistent_workers": bool(defaults["persistent_workers"]),
        "prefetch_factor": int(defaults["prefetch_factor"]),
        "load_full_trajectory": load_full_trajectory,
        "batch_size": batch_size,
        "epochs": epochs,
        "steps": int(budget.get("steps", 0) or 0),
        "refine_steps": int(budget.get("refine_steps", 0) or 0),
        "particles": int(budget.get("particles", 0) or 0),
        "device": defaults["device"],
        "config": str(group_cfg.get("config", defaults["config"])),
        "output_dir": "",
        "log_dir": "",
        "status_file": "",
        "skip_reason": "",
    }
    row["run_id"] = _run_id(row)
    row["run_name"] = _run_name(row)
    row["output_dir"] = str(_run_output_dir(output_root, matrix_name, row))
    row["log_dir"] = str(_run_log_dir(output_root, matrix_name, row))
    row["status_file"] = str(Path(row["output_dir"]) / "run.status.json")
    return row


def _resource_config(cfg: dict[str, Any], baseline: str) -> dict[str, Any]:
    resources = dict(cfg.get("resources", {}) or {})
    base_name = "per_instance_default" if baseline in PER_INSTANCE_BASELINES else "amortized_default"
    merged = dict(resources.get(base_name, {}) or {})
    merged.update(resources.get(baseline, {}) or {})
    return merged


def _method_steps(baseline: str, resources: dict[str, Any]) -> int:
    if baseline == "pinn_sparse":
        return _env_int("PINN_STEPS", int(resources.get("steps", 1000)))
    if baseline == "pde_opt":
        return _env_int("PDEOPT_STEPS", int(resources.get("steps", 500)))
    if baseline == "var4d":
        return _env_int("VAR4D_STEPS", int(resources.get("steps", 500)))
    if baseline == "pc_bnn":
        return _env_int("PCBNN_STEPS", int(resources.get("steps", 500)))
    return int(resources.get("steps", 0) or 0)


def _method_refine_steps(baseline: str, resources: dict[str, Any]) -> int:
    if baseline == "vivid":
        return _env_int("VIVID_REFINE_STEPS", int(resources.get("refine_steps", 300)))
    return int(resources.get("refine_steps", 0) or 0)


def _method_particles(baseline: str, resources: dict[str, Any]) -> int:
    if baseline == "pc_bnn":
        return _env_int("PCBNN_PARTICLES", int(resources.get("particles", 8)))
    return int(resources.get("particles", 0) or 0)


def _scalar_param_mode(defaults: dict[str, Any], pde: str, task: str) -> str:
    explicit = str(defaults.get("scalar_param_mode", "") or "")
    if explicit:
        return explicit
    if pde in FUTURE_PDES and task in {"forward", "inverse", "sparse_inverse"}:
        return "materialize"
    return "metadata"


def _run_id(row: dict[str, Any]) -> str:
    payload = {key: row[key] for key in HASH_FIELDS}
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]
    prefix = f"{row['task_group']}_{row['baseline']}_{row['pde']}_s{row['seed']}"
    return _safe_name(f"{prefix}_{digest}")


def _run_name(row: dict[str, Any]) -> str:
    base = f"{row['task_group']}/{row['baseline']}/{row['pde']}/seed={row['seed']}"
    if row["task"].startswith("sparse"):
        base += f"/sensors={row['num_sensors']}/{row['sensor_mode']}/noise={row['noise_level']}"
    if row["ablation_factor"] == "train_size":
        base += f"/train_size={row['train_size']}"
    if row["ablation_factor"] == "runtime_budget":
        budget = _budget_label(row)
        if budget:
            base += f"/{budget}"
    return base


def _run_output_dir(output_root: Path, matrix_name: str, row: dict[str, Any]) -> Path:
    if row["experiment_kind"] == "main":
        parts = [
            "runs",
            matrix_name,
            f"task_group={_safe_name(row['task_group'])}",
            f"pde={_safe_name(row['pde'])}",
            f"baseline={_safe_name(row['baseline'])}",
            f"seed={row['seed']}",
            f"run={row['run_id']}",
        ]
        return output_root.joinpath(*parts)
    ablation_name = _ablation_dir_name(row["ablation_factor"], matrix_name)
    parts = [
        "runs",
        "ablations",
        f"ablation={_safe_name(ablation_name)}",
        f"pde={_safe_name(row['pde'])}",
        f"baseline={_safe_name(row['baseline'])}",
        f"seed={row['seed']}",
        f"sensors={row['num_sensors']}",
        f"mode={_safe_name(row['sensor_mode'])}",
        f"noise={_safe_float(row['noise_level'])}",
    ]
    if row["ablation_factor"] == "train_size":
        parts.append(f"train_size={row['train_size']}")
    if row["ablation_factor"] == "runtime_budget":
        budget = _budget_label(row)
        if budget:
            parts.append(_safe_name(budget))
    parts.append(f"run={row['run_id']}")
    return output_root.joinpath(*parts)


def _run_log_dir(output_root: Path, matrix_name: str, row: dict[str, Any]) -> Path:
    if row["experiment_kind"] == "main":
        return output_root / "logs" / matrix_name / f"task_group={_safe_name(row['task_group'])}" / f"run={row['run_id']}"
    ablation_name = _ablation_dir_name(row["ablation_factor"], matrix_name)
    return output_root / "logs" / "ablations" / f"ablation={_safe_name(ablation_name)}" / f"run={row['run_id']}"


def _ablation_dir_name(ablation_factor: str, matrix_name: str) -> str:
    return {
        "sensor_count": "sensor_count",
        "noise_level": "noise",
        "sensor_mode": "sensor_mode",
        "time_varying_sensor_count": "time_varying",
        "runtime_budget": "runtime_budget",
        "train_size": "train_size",
    }.get(ablation_factor, matrix_name)


def _budget_label(row: dict[str, Any]) -> str:
    baseline = str(row["baseline"])
    if baseline == "vivid" and int(row.get("refine_steps", 0) or 0) > 0:
        return f"refine_steps={int(row['refine_steps'])}"
    if baseline == "pc_bnn" and int(row.get("particles", 0) or 0) > 0 and int(row.get("steps", 0) or 0) <= 0:
        return f"particles={int(row['particles'])}"
    if int(row.get("steps", 0) or 0) > 0:
        return f"steps={int(row['steps'])}"
    if int(row.get("particles", 0) or 0) > 0:
        return f"particles={int(row['particles'])}"
    if int(row.get("refine_steps", 0) or 0) > 0:
        return f"refine_steps={int(row['refine_steps'])}"
    return ""


def _skipped_matrix_row(skip: dict[str, Any], defaults: dict[str, Any], output_root: Path) -> dict[str, Any]:
    row = {field: "" for field in MATRIX_FIELDS}
    row.update(
        {
            "run_id": _safe_name(f"skipped_{skip['task_group']}_{skip['baseline']}_{skip['pde']}_{skip['sensor_mode']}"),
            "experiment_kind": skip.get("experiment_kind", ""),
            "ablation_factor": skip.get("ablation_factor", ""),
            "task_group": skip["task_group"],
            "task": skip["task"],
            "pde": skip["pde"],
            "baseline": skip["baseline"],
            "seed": 0,
            "train_size": defaults["train_size"],
            "val_size": defaults["val_size"],
            "test_size": defaults["test_size"],
            "train_shards": defaults["train_shards"],
            "num_sensors": 0,
            "sensor_mode": skip["sensor_mode"],
            "noise_level": 0.0,
            "scalar_param_mode": "metadata",
            "data_loading_mode": defaults["data_loading_mode"],
            "num_workers": int(defaults["num_workers"]),
            "pin_memory": bool(defaults["pin_memory"]),
            "persistent_workers": bool(defaults["persistent_workers"]),
            "prefetch_factor": int(defaults["prefetch_factor"]),
            "load_full_trajectory": False,
            "batch_size": 0,
            "epochs": 0,
            "steps": 0,
            "refine_steps": 0,
            "particles": 0,
            "device": defaults["device"],
            "config": defaults["config"],
            "output_dir": str(output_root / "skipped" / _safe_name(skip["task_group"])),
            "log_dir": str(output_root / "logs" / "skipped"),
            "status_file": "",
            "skip_reason": skip["reason"],
        }
    )
    return row


def _merge_skip(skipped: dict[tuple[Any, ...], dict[str, Any]], row: dict[str, Any]) -> None:
    key = (
        row["matrix_name"],
        row.get("experiment_kind", ""),
        row.get("ablation_factor", ""),
        row["task_group"],
        row["task"],
        row["pde"],
        row["baseline"],
        row["sensor_mode"],
        row["reason"],
    )
    if key not in skipped:
        skipped[key] = dict(row)
    else:
        skipped[key]["would_have_expanded"] = int(skipped[key]["would_have_expanded"]) + int(row["would_have_expanded"])


def _summary(
    rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    matrix_name: str,
    experiment_kind: str,
    ablation_factor: str,
) -> dict[str, Any]:
    active_rows = [row for row in rows if not row.get("skip_reason")]
    by_group = Counter(row["task_group"] for row in active_rows)
    by_pde = Counter(row["pde"] for row in active_rows)
    by_baseline = Counter(row["baseline"] for row in active_rows)
    by_group_baseline = Counter((row["task_group"], row["baseline"]) for row in active_rows)
    skipped_by_group = Counter()
    skipped_total_expanded = 0
    for row in skipped_rows:
        skipped_by_group[row["task_group"]] += int(row.get("would_have_expanded", 1))
        skipped_total_expanded += int(row.get("would_have_expanded", 1))
    varied_fields, fixed_fields = _field_variation(active_rows, ablation_factor)
    return {
        "matrix_name": matrix_name,
        "experiment_kind": experiment_kind,
        "ablation_factor": ablation_factor,
        "run_count": len(active_rows),
        "skipped_combo_count": len(skipped_rows),
        "skipped_expanded_count": skipped_total_expanded,
        "by_task_group": dict(sorted(by_group.items())),
        "by_pde": dict(sorted(by_pde.items())),
        "by_baseline": dict(sorted(by_baseline.items())),
        "varied_fields": varied_fields,
        "fixed_fields": fixed_fields,
        "skipped_by_task_group": dict(sorted(skipped_by_group.items())),
        "by_task_group_baseline": {
            f"{group}/{baseline}": count for (group, baseline), count in sorted(by_group_baseline.items())
        },
    }


def _field_variation(rows: list[dict[str, Any]], ablation_factor: str) -> tuple[dict[str, Any], dict[str, Any]]:
    varied: dict[str, Any] = {}
    fixed: dict[str, Any] = {}
    if not rows:
        return varied, fixed
    fields = list(SUMMARY_DESIGN_FIELDS)
    if ablation_factor != "runtime_budget":
        fields = [field for field in fields if field not in {"steps", "refine_steps", "particles"}]
    for field in fields:
        values = sorted({str(row.get(field, "")) for row in rows})
        if len(values) == 1:
            fixed[field] = values[0]
        else:
            varied[field] = {"count": len(values), "values": values[:30]}
    return varied, fixed


def _ensure_unique_run_ids(rows: list[dict[str, Any]]) -> None:
    ids = [row["run_id"] for row in rows if not row.get("skip_reason")]
    duplicates = [run_id for run_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"run_id collision detected: {duplicates[:10]}")


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MATRIX_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_list(value: Any, all_values: list[str]) -> list[str]:
    if isinstance(value, str) and value.lower() == "all":
        return list(all_values)
    if isinstance(value, str):
        return shlex.split(value)
    return list(value)


def _first_present(mapping: dict[str, Any], keys: list[str], default: Any) -> Any:
    for key in keys:
        if key in mapping:
            value = mapping[key]
            if key in {"num_sensors", "sensor_count", "sensor_mode", "noise_level"} and not isinstance(value, list):
                return [value]
            return value
    return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in {None, ""} else int(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in {None, ""}:
        return bool(default)
    return str(value).lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: Any, caster) -> list[Any]:
    raw = os.environ.get(name)
    if raw not in {None, ""}:
        return [caster(item) for item in shlex.split(raw)]
    if isinstance(default, str):
        return [caster(item) for item in shlex.split(default)]
    return [caster(item) for item in list(default)]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _safe_name(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "="} else "_" for ch in text)


def _safe_float(value: Any) -> str:
    return _safe_name(f"{float(value):g}")


def _short_list(values: list[Any], max_items: int = 12) -> str:
    text_values = [str(value) for value in values]
    if len(text_values) <= max_items:
        return ",".join(text_values)
    shown = ",".join(text_values[:max_items])
    return f"{shown},...({len(text_values)} total)"


def _output_paths(output_root: str | Path, matrix_name: str) -> dict[str, Path]:
    output_root = Path(output_root)
    matrix_dir = output_root / "matrices"
    return {
        "jsonl": matrix_dir / f"{matrix_name}.jsonl",
        "tsv": matrix_dir / f"{matrix_name}.tsv",
        "summary": matrix_dir / f"{matrix_name}_summary.json",
        "skipped": matrix_dir / f"{matrix_name}_skipped.jsonl",
        "skipped_global": output_root / "skipped_combinations.jsonl",
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    matrix_name = args.matrix_name or str(cfg.get("name") or Path(args.config).stem)
    main_table_only = bool(args.main_table_only or args.paper_mode or cfg.get("main_table_only", False))
    task_groups = list(cfg.get("task_groups", []))
    progress(
        f"[matrix start] config={args.config} matrix_name={matrix_name} output_root={args.output_root} "
        f"experiment_kind={cfg.get('experiment_kind', '')} main_table_only={main_table_only} "
        f"task_groups={_short_list(task_groups)}"
    )
    rows, skipped, summary = build_matrix(
        cfg,
        args.output_root,
        matrix_name,
        include_skipped=args.include_skipped,
        main_table_only=main_table_only,
        emit_progress=True,
    )
    write_outputs(rows, skipped, summary, args.output_root, matrix_name)
    output_paths = _output_paths(args.output_root, matrix_name)
    progress(
        f"[matrix complete] rows={len(rows)} active_rows={summary['run_count']} skipped_combo_count={len(skipped)} "
        f"skipped_expanded_count={summary.get('skipped_expanded_count', 0)} "
        f"by_task_group={json.dumps(summary.get('by_task_group', {}), sort_keys=True)}"
    )
    progress(
        f"[matrix outputs] jsonl={output_paths['jsonl']} tsv={output_paths['tsv']} summary={output_paths['summary']} "
        f"skipped={output_paths['skipped']} skipped_global={output_paths['skipped_global']}"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
