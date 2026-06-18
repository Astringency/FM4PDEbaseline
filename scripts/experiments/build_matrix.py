#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.experiment_matrix import (
    ALL_PDES,
    FUTURE_PDES,
    PER_INSTANCE_BASELINES,
    compatibility_reason,
)


GROUP_TO_TASK = {
    "full_forward": "forward",
    "full_inverse": "inverse",
    "sparse_solution_amortized": "sparse_solution",
    "sparse_solution_physics": "sparse_solution",
    "sparse_inverse": "sparse_inverse",
    "time_varying": "sparse_solution",
}

DEFAULT_BASELINES_BY_GROUP = {
    "full_forward": ["fno", "deeponet", "ifno"],
    "full_inverse": ["fno", "deeponet", "ifno"],
    "sparse_solution_amortized": ["recfno", "senseiver", "voronoicnn", "fno", "deeponet", "ifno"],
    "sparse_solution_physics": ["pinn_sparse", "pc_bnn", "pde_opt", "var4d", "vivid"],
    "sparse_inverse": ["recfno", "senseiver", "voronoicnn", "fno", "deeponet", "pinn_sparse", "pc_bnn", "pde_opt", "var4d", "vivid"],
    "time_varying": ["var4d", "vivid", "senseiver"],
}

HASH_FIELDS = [
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
    "scalar_param_mode",
    "data_loading_mode",
    "load_full_trajectory",
]

MATRIX_FIELDS = [
    "run_id",
    "run_name",
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build large-scale FM4PDE external-baseline experiment matrices.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default=os.environ.get("OUT_ROOT", "outputs/baselines_large"))
    parser.add_argument("--matrix-name", default="")
    parser.add_argument("--include-skipped", action="store_true", help="Also include unsupported rows in the matrix with skip_reason set.")
    return parser.parse_args(argv)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_matrix(
    cfg: dict[str, Any],
    output_root: str | Path,
    matrix_name: str,
    include_skipped: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    output_root = Path(output_root)
    task_groups = list(cfg.get("task_groups", []))
    rows: list[dict[str, Any]] = []
    skipped: dict[tuple[Any, ...], dict[str, Any]] = {}
    global_defaults = _global_defaults(cfg)

    for task_group in task_groups:
        if task_group not in GROUP_TO_TASK:
            raise ValueError(f"Unknown task_group {task_group!r}")
        group_cfg = dict(cfg.get("task_group_overrides", {}).get(task_group, {}) or {})
        task = str(group_cfg.get("task", GROUP_TO_TASK[task_group]))
        pdes = _resolve_list(group_cfg.get("pdes", cfg.get("pdes", ALL_PDES)), ALL_PDES)
        baselines = list(group_cfg.get("baselines", DEFAULT_BASELINES_BY_GROUP[task_group]))
        extra_skip_baselines = list(group_cfg.get("extra_skip_baselines", []) or [])
        candidate_baselines = baselines + [baseline for baseline in extra_skip_baselines if baseline not in baselines]
        seeds = _env_list("SEEDS", group_cfg.get("seeds", global_defaults["seeds"]), int)

        sparse = task.startswith("sparse")
        if sparse:
            sensor_counts = _env_list("SENSOR_COUNTS", group_cfg.get("sensor_counts", global_defaults["sensor_counts"]), int)
            sensor_modes = _env_list("SENSOR_MODES", group_cfg.get("sensor_modes", global_defaults["sensor_modes"]), str)
            noise_levels = _env_list("NOISE_LEVELS", group_cfg.get("noise_levels", global_defaults["noise_levels"]), float)
        else:
            sensor_counts = [0]
            sensor_modes = ["none"]
            noise_levels = [0.0]

        for pde in pdes:
            for baseline in candidate_baselines:
                for sensor_mode in sensor_modes:
                    reason = compatibility_reason(baseline, pde, task, "" if sensor_mode == "none" else sensor_mode, task_group)
                    if reason:
                        skipped_row = {
                            "matrix_name": matrix_name,
                            "task_group": task_group,
                            "task": task,
                            "pde": pde,
                            "baseline": baseline,
                            "sensor_mode": sensor_mode,
                            "reason": reason,
                            "would_have_expanded": len(seeds) * len(sensor_counts) * len(noise_levels),
                        }
                        _merge_skip(skipped, skipped_row)
                        if include_skipped:
                            rows.append(_skipped_matrix_row(skipped_row, global_defaults, output_root))
                        continue
                    for seed in seeds:
                        for num_sensors in sensor_counts:
                            for noise_level in noise_levels:
                                row = _make_run_row(
                                    cfg=cfg,
                                    group_cfg=group_cfg,
                                    defaults=global_defaults,
                                    output_root=output_root,
                                    matrix_name=matrix_name,
                                    task_group=task_group,
                                    task=task,
                                    pde=str(pde),
                                    baseline=str(baseline),
                                    seed=int(seed),
                                    num_sensors=int(num_sensors),
                                    sensor_mode=str(sensor_mode),
                                    noise_level=float(noise_level),
                                )
                                rows.append(row)

    skipped_rows = list(skipped.values())
    summary = _summary(rows, skipped_rows, matrix_name)
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


def _global_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "seeds": _env_list("SEEDS", cfg.get("seeds", [1, 2, 3]), int),
        "train_size": _env_int("TRAIN_SIZE", int(cfg.get("train_size", 50000))),
        "val_size": _env_int("VAL_SIZE", int(cfg.get("val_size", 0))),
        "test_size": _env_int("TEST_SIZE", int(cfg.get("test_size", 1000))),
        "train_shards": _env_int("TRAIN_SHARDS", int(cfg.get("train_shards", 5))),
        "sensor_counts": _env_list("SENSOR_COUNTS", cfg.get("sensor_counts", [50, 100, 250, 500, 1000]), int),
        "sensor_modes": _env_list("SENSOR_MODES", cfg.get("sensor_modes", ["random", "fixed", "grid"]), str),
        "noise_levels": _env_list("NOISE_LEVELS", cfg.get("noise_levels", [0.0, 0.01, 0.05]), float),
        "data_loading_mode": os.environ.get("DATA_LOADING_MODE", str(cfg.get("data_loading_mode", "lazy"))),
        "device": os.environ.get("DEVICE", str(cfg.get("device", "cuda"))),
        "config": str(cfg.get("config", "baselines/configs/paper.yaml")),
        "scalar_param_mode": os.environ.get("SCALAR_PARAM_MODE", ""),
    }


def _make_run_row(
    cfg: dict[str, Any],
    group_cfg: dict[str, Any],
    defaults: dict[str, Any],
    output_root: Path,
    matrix_name: str,
    task_group: str,
    task: str,
    pde: str,
    baseline: str,
    seed: int,
    num_sensors: int,
    sensor_mode: str,
    noise_level: float,
) -> dict[str, Any]:
    resources = _resource_config(cfg, baseline)
    train_size = _env_int("TRAIN_SIZE", int(group_cfg.get("train_size", defaults["train_size"])))
    val_size = _env_int("VAL_SIZE", int(group_cfg.get("val_size", defaults["val_size"])))
    test_size = _env_int("TEST_SIZE", int(group_cfg.get("test_size", defaults["test_size"])))
    train_shards = _env_int("TRAIN_SHARDS", int(group_cfg.get("train_shards", defaults["train_shards"])))
    batch_size = _env_int("BATCH_SIZE", int(group_cfg.get("batch_size", resources.get("batch_size", 16))))
    epochs = _env_int("EPOCHS", int(group_cfg.get("epochs", resources.get("epochs", 1 if baseline in PER_INSTANCE_BASELINES else 200))))
    load_full_trajectory = bool(group_cfg.get("load_full_trajectory", resources.get("load_full_trajectory", task_group == "time_varying")))
    if task_group == "time_varying":
        load_full_trajectory = True
    scalar_param_mode = _scalar_param_mode(defaults, pde, task)
    row: dict[str, Any] = {
        "run_id": "",
        "run_name": "",
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
        "load_full_trajectory": load_full_trajectory,
        "batch_size": batch_size,
        "epochs": epochs,
        "steps": _method_steps(baseline, resources),
        "refine_steps": _method_refine_steps(baseline, resources),
        "particles": _method_particles(baseline, resources),
        "device": defaults["device"],
        "config": str(group_cfg.get("config", defaults["config"])),
        "output_dir": "",
        "log_dir": "",
        "status_file": "",
        "skip_reason": "",
    }
    row["run_id"] = _run_id(row)
    row["run_name"] = _run_name(row)
    row["output_dir"] = str(_run_output_dir(output_root, row))
    row["log_dir"] = str(_run_log_dir(output_root, row))
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
    if row["task"].startswith("sparse"):
        return (
            f"{row['task_group']}/{row['baseline']}/{row['pde']}/seed={row['seed']}/"
            f"sensors={row['num_sensors']}/{row['sensor_mode']}/noise={row['noise_level']}"
        )
    return f"{row['task_group']}/{row['baseline']}/{row['pde']}/seed={row['seed']}"


def _run_output_dir(output_root: Path, row: dict[str, Any]) -> Path:
    parts = [
        "runs",
        f"task_group={_safe_name(row['task_group'])}",
        f"pde={_safe_name(row['pde'])}",
        f"baseline={_safe_name(row['baseline'])}",
        f"task={_safe_name(row['task'])}",
        f"seed={row['seed']}",
        f"sensors={row['num_sensors']}",
        f"mode={_safe_name(row['sensor_mode'])}",
        f"noise={_safe_float(row['noise_level'])}",
        f"scalar={_safe_name(row['scalar_param_mode'])}",
        f"run={row['run_id']}",
    ]
    return output_root.joinpath(*parts)


def _run_log_dir(output_root: Path, row: dict[str, Any]) -> Path:
    return output_root / "logs" / f"task_group={_safe_name(row['task_group'])}" / f"run={row['run_id']}"


def _skipped_matrix_row(skip: dict[str, Any], defaults: dict[str, Any], output_root: Path) -> dict[str, Any]:
    row = {field: "" for field in MATRIX_FIELDS}
    row.update(
        {
            "run_id": _safe_name(f"skipped_{skip['task_group']}_{skip['baseline']}_{skip['pde']}_{skip['sensor_mode']}"),
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
            "load_full_trajectory": False,
            "batch_size": 0,
            "epochs": 0,
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
    key = (row["matrix_name"], row["task_group"], row["task"], row["pde"], row["baseline"], row["sensor_mode"], row["reason"])
    if key not in skipped:
        skipped[key] = dict(row)
    else:
        skipped[key]["would_have_expanded"] = int(skipped[key]["would_have_expanded"]) + int(row["would_have_expanded"])


def _summary(rows: list[dict[str, Any]], skipped_rows: list[dict[str, Any]], matrix_name: str) -> dict[str, Any]:
    by_group = Counter(row["task_group"] for row in rows if not row.get("skip_reason"))
    by_group_baseline = Counter((row["task_group"], row["baseline"]) for row in rows if not row.get("skip_reason"))
    skipped_by_group = Counter()
    skipped_total_expanded = 0
    for row in skipped_rows:
        skipped_by_group[row["task_group"]] += int(row.get("would_have_expanded", 1))
        skipped_total_expanded += int(row.get("would_have_expanded", 1))
    return {
        "matrix_name": matrix_name,
        "run_count": len([row for row in rows if not row.get("skip_reason")]),
        "skipped_combo_count": len(skipped_rows),
        "skipped_expanded_count": skipped_total_expanded,
        "by_task_group": dict(sorted(by_group.items())),
        "skipped_by_task_group": dict(sorted(skipped_by_group.items())),
        "by_task_group_baseline": {
            f"{group}/{baseline}": count for (group, baseline), count in sorted(by_group_baseline.items())
        },
    }


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


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in {None, ""} else int(default)


def _env_list(name: str, default: Any, caster) -> list[Any]:
    raw = os.environ.get(name)
    if raw not in {None, ""}:
        return [caster(item) for item in shlex.split(raw)]
    if isinstance(default, str):
        return [caster(item) for item in shlex.split(default)]
    return [caster(item) for item in list(default)]


def _safe_name(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "="} else "_" for ch in text)


def _safe_float(value: Any) -> str:
    return _safe_name(f"{float(value):g}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    matrix_name = args.matrix_name or str(cfg.get("name") or Path(args.config).stem)
    rows, skipped, summary = build_matrix(cfg, args.output_root, matrix_name, include_skipped=args.include_skipped)
    write_outputs(rows, skipped, summary, args.output_root, matrix_name)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
