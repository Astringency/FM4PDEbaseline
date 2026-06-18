from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from baselines.capabilities import paper_table_eligible as capability_paper_table_eligible
from baselines.capabilities import resolve_capability
from baselines.common.data_adapter import PDEBatch, PDEBatchDataset, build_default_registry, pde_collate, slice_pde_batch
from baselines.common.metrics import (
    append_result_csv,
    append_result_jsonl,
    mae,
    mse,
    num_parameters,
    obs_mse,
    physics_loss_metric,
)
from baselines.methods.deeponet import DeepONetBaseline
from baselines.methods.fno import FNOBaseline
from baselines.methods.ifno import IFNOBaseline
from baselines.methods.official import OfficialImportError
from baselines.methods.official import (
    get_ifno_official_aligned_status,
    get_pc_bnn_official_aligned_status,
    get_vivid_official_aligned_status,
)
from baselines.methods.pc_bnn import PCBNNBaseline
from baselines.methods.pde_opt import PDEOptBaseline
from baselines.methods.pinn_sparse import PINNSparseBaseline
from baselines.methods.recfno import RecFNOBaseline
from baselines.methods.senseiver import SenseiverBaseline
from baselines.methods.var4d import Var4DBaseline
from baselines.methods.vivid import VIVIDBaseline
from baselines.methods.voronoicnn import VoronoiCNNBaseline
from baselines.experiment_matrix import capability_skip_row


BASELINES = {
    "fno": FNOBaseline,
    "deeponet": DeepONetBaseline,
    "ifno": IFNOBaseline,
    "recfno": RecFNOBaseline,
    "senseiver": SenseiverBaseline,
    "voronoicnn": VoronoiCNNBaseline,
    "pinn_sparse": PINNSparseBaseline,
    "pc_bnn": PCBNNBaseline,
    "pde_opt": PDEOptBaseline,
    "var4d": Var4DBaseline,
    "vivid": VIVIDBaseline,
}

PER_INSTANCE_BASELINES = {"pinn_sparse", "pc_bnn", "pde_opt", "var4d", "vivid"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("FM4PDE baseline runner")
    parser.add_argument("--baseline", required=True, choices=sorted(BASELINES))
    parser.add_argument("--pde", required=True)
    parser.add_argument("--task", default="forward")
    parser.add_argument("--data-root", default="/home/tat512/C01Python/PDEdata")
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiment-mode", choices=["smoke", "debug", "paper"], default="debug")
    parser.add_argument("--num-sensors", type=int, default=500)
    parser.add_argument("--sensor-mode", choices=["random", "fixed", "grid", "time_varying"], default="random")
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--val-size", type=int, default=0)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--train-shards", type=int, default=5)
    parser.add_argument("--test-split", choices=["test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/baselines")
    parser.add_argument("--experiment-kind", default="")
    parser.add_argument("--ablation-factor", default="")
    parser.add_argument("--task-group", default="")
    parser.add_argument("--run-id", default="", help="Stable external run identifier used in output filenames.")
    parser.add_argument("--run-name", default="", help="Human-readable external run name stored in metadata.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true", help="Use deterministic synthetic data for smoke/debug tests.")
    parser.add_argument("--allow-synthetic-fallback", action="store_true", help="Fall back to synthetic data when requested real files are missing.")
    parser.add_argument("--synthetic-resolution", type=int, default=32)
    parser.add_argument("--prefer-test", action="store_true", help="Compatibility/debug option. Never use for paper training.")
    parser.add_argument("--scalar-param-mode", choices=["metadata", "materialize", "global"], default="metadata")
    parser.add_argument("--data-loading-mode", choices=["eager", "lazy"], default=None)
    parser.add_argument("--load-full-trajectory", action="store_true", help="Load full time trajectories when available instead of endpoint-only task tensors.")
    parser.add_argument("--physics-metric-mode", choices=["per_sample", "per_batch"], default=None)
    parser.add_argument("--strict-size", action="store_true", help="Fail if requested split size exceeds available samples.")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--steps", type=int, default=None, help="Override per-instance optimization steps in the method config.")
    parser.add_argument("--refine-steps", type=int, default=None, help="Override VIVID refinement steps in the method config.")
    parser.add_argument("--particles", type=int, default=None, help="Override PC-BNN particle count in the method config.")
    parser.add_argument("--implementation-mode", default=None, help="Override method implementation_mode.")
    parser.add_argument("--official-backend", default=None, help="Override method official_backend.")
    parser.add_argument("--method-override", action="append", default=[], help="Override a method config key as key=value. Can be repeated.")
    return parser.parse_args(argv)


def load_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_method_config(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged: dict[str, Any] = dict(cfg.get("method", {}))
    merged.update(cfg.get("method_by_baseline", {}).get(args.baseline, {}) or {})
    merged.update(cfg.get("method_by_pde", {}).get(args.pde, {}) or {})
    merged.update(cfg.get("method_by_baseline_and_pde", {}).get(args.baseline, {}).get(args.pde, {}) or {})
    if args.epochs is not None:
        merged["epochs"] = args.epochs
    else:
        merged.setdefault("epochs", int(cfg.get("epochs", 1)))
    if args.lr is not None:
        merged["lr"] = args.lr
    else:
        merged.setdefault("lr", float(cfg.get("learning_rate", 1e-3)))
    if args.steps is not None:
        merged["steps"] = int(args.steps)
    if args.refine_steps is not None:
        merged["refine_steps"] = int(args.refine_steps)
    if args.particles is not None:
        merged["particles"] = int(args.particles)
    if args.implementation_mode is not None:
        merged["implementation_mode"] = str(args.implementation_mode)
    if args.official_backend is not None:
        merged["official_backend"] = str(args.official_backend)
    for override in args.method_override or []:
        key, sep, value = str(override).partition("=")
        if not sep or not key:
            raise ValueError(f"--method-override must be key=value, got {override!r}")
        merged[key] = _parse_override_value(value)
    merged["device"] = args.device
    merged.setdefault("seed", int(args.seed))
    if args.dry_run:
        merged["max_steps"] = min(int(merged.get("max_steps", 1) or 1), 1)
        merged["max_val_steps"] = min(int(merged.get("max_val_steps", 1) or 1), 1)
        merged.setdefault("steps", 1)
        merged.setdefault("refine_steps", 1)
        merged["steps"] = min(int(merged.get("steps", 1)), 1)
        merged["refine_steps"] = min(int(merged.get("refine_steps", 1)), 1)
        merged["epochs"] = min(int(merged.get("epochs", 1)), 1)
    return merged


def build_data_spec(batch: PDEBatch) -> dict[str, Any]:
    branch_numel = int(batch.obs_values[0].numel()) if batch.obs_values is not None else int(batch.input_fields[0].numel())
    return {
        "pde": batch.pde_name,
        "task": batch.task,
        "input_shape": tuple(batch.input_fields.shape),
        "target_shape": tuple(batch.target_fields.shape),
        "input_channels": int(batch.input_fields.shape[1]),
        "target_channels": int(batch.target_fields.shape[1]),
        "input_channel_names": list(batch.input_channel_names),
        "target_channel_names": list(batch.target_channel_names),
        "input_numel": int(batch.input_fields[0].numel()),
        "target_numel": int(batch.target_fields[0].numel()),
        "branch_numel": branch_numel,
        "metadata": {k: v for k, v in batch.metadata.items() if not isinstance(v, torch.Tensor)},
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_mode(args)
    torch.manual_seed(args.seed)
    cfg = load_yaml(args.config)
    args.data_loading_mode = _resolve_data_loading_mode(args)
    args.physics_metric_mode = _resolve_physics_metric_mode(args, cfg)
    method_cfg = build_method_config(cfg, args)
    capability = resolve_capability(
        args.baseline,
        args.pde,
        args.task,
        args.sensor_mode if args.task.startswith("sparse") else "",
        args.task_group,
        load_full_trajectory=bool(args.load_full_trajectory or args.baseline in {"var4d", "vivid"}),
        train_inverse_operator=bool(method_cfg.get("train_inverse_operator", False)),
        uses_official_inverse_observation_operator=_uses_official_inverse_observation_operator(args.baseline, method_cfg),
    )
    capability_info = capability.to_row()
    if args.experiment_mode == "paper" and capability.support_status == "unsupported":
        _skip_paper_run(args, capability, method_cfg, capability.reason, adapter_status="capability_unsupported")
        return
    if (
        args.experiment_mode == "paper"
        and capability.implementation_required == "adapted_allowed"
        and _official_request_mode(method_cfg) in {"official", "official_or_skip", "official_architecture", "official_aligned"}
    ):
        _skip_paper_run(
            args,
            capability,
            method_cfg,
            f"{capability.reason}; adapted-only capability skipped in official/official_or_skip paper mode",
            adapter_status="adapted_only_skipped",
        )
        return
    preflight_skip = _preflight_backend_availability(args, capability, method_cfg)
    if preflight_skip is not None:
        _skip_paper_run(
            args,
            capability,
            method_cfg,
            preflight_skip["reason"],
            backend_info=preflight_skip.get("backend_info"),
            adapter_status=preflight_skip.get("adapter_status", "official_backend_unavailable"),
        )
        return

    train_size, val_size, test_size = _effective_sizes(args)
    registry = build_default_registry()
    use_sensors = args.task.startswith("sparse")

    val_dataset, split_info = _make_val_dataset_if_requested(registry, args, val_size, train_size, use_sensors)
    effective_train_size = int(split_info["effective_train_size"])
    is_per_instance = args.baseline in PER_INSTANCE_BASELINES
    if args.baseline == "vivid" and (
        bool(method_cfg.get("train_inverse_operator", False)) or _uses_official_inverse_observation_operator(args.baseline, method_cfg)
    ):
        is_per_instance = False
    spec_size = max(1, min(int(args.batch_size), 4, max(effective_train_size, 1)))
    if is_per_instance:
        spec_dataset = _make_split_dataset(
            registry,
            args,
            "train",
            spec_size,
            use_sensors,
            synthetic_seed=args.seed * 1000 + 11,
        )
        train_dataset_for_fit = spec_dataset
    else:
        train_dataset_for_fit = _make_split_dataset(
            registry,
            args,
            "train",
            effective_train_size,
            use_sensors,
            synthetic_seed=args.seed * 1000 + 11,
        )
        spec_dataset = train_dataset_for_fit
        if split_info["val_split_source"] == "deterministic_train_subset" and len(train_dataset_for_fit) < effective_train_size:
            raise ValueError(
                f"Cannot reserve validation tail: requested effective train size {effective_train_size}, "
                f"but only loaded {len(train_dataset_for_fit)} training samples."
            )
    test_dataset = _make_split_dataset(
        registry,
        args,
        "test",
        test_size,
        use_sensors,
        synthetic_seed=args.seed * 1000 + 23,
    )

    train_loader = DataLoader(train_dataset_for_fit, batch_size=args.batch_size, shuffle=not is_per_instance, collate_fn=pde_collate)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pde_collate)
        if val_dataset is not None
        else None
    )
    spec_loader = DataLoader(spec_dataset, batch_size=min(args.batch_size, len(spec_dataset)), shuffle=False, collate_fn=pde_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pde_collate)
    data_spec = build_data_spec(next(iter(spec_loader)))
    split_info.update(
        {
            "train_size_loaded_for_fit": 0 if is_per_instance else len(train_dataset_for_fit),
            "train_size_loaded_for_spec": len(spec_dataset),
            "train_size_requested": int(train_size),
            "train_size_loaded_in_memory": int(getattr(train_dataset_for_fit, "loaded_in_memory_samples", len(train_dataset_for_fit))),
            "per_instance_baseline": bool(is_per_instance),
        }
    )

    try:
        model = BASELINES[args.baseline]().build(method_cfg, data_spec).to(args.device)
    except OfficialImportError as exc:
        if args.experiment_mode == "paper" and _official_request_mode(method_cfg) == "official_or_skip":
            _skip_paper_run(
                args,
                capability,
                method_cfg,
                f"official backend unavailable: {exc}",
                adapter_status="official_import_failed",
            )
            return
        raise
    backend_info = _backend_info(model, method_cfg)
    eligibility = _paper_table_eligibility(capability, backend_info)
    capability_info["paper_table_eligible"] = eligibility
    if capability.support_status != "unsupported":
        capability_info["unsupported_reason"] = ""
    method_budget_fields = _method_budget_fields(method_cfg, args.baseline)
    request_mode = _official_request_mode(method_cfg)
    if args.experiment_mode == "paper" and request_mode in {"official", "official_or_skip", "official_architecture", "official_aligned"} and backend_info["fallback_used"]:
        reason = (
            f"{args.baseline} requested {request_mode} backend for paper mode but used fallback backend "
            f"{backend_info['backend_used']!r}: {backend_info.get('backend_warning', '')}"
        )
        if request_mode == "official_or_skip":
            _skip_paper_run(args, capability, method_cfg, reason, backend_info=backend_info)
            return
        raise RuntimeError(
            f"{args.baseline} requested official backend for paper mode but used fallback backend "
            f"{backend_info['backend_used']!r}. Install/enable the official dependency or use implementation_mode: official_or_skip."
        )

    train_start = time.perf_counter()
    train_history: dict[str, Any] = {}
    if not is_per_instance:
        train_history = model.fit(train_loader, val_loader)
    else:
        train_history = model.fit(train_loader, val_loader)
    train_time = time.perf_counter() - train_start

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = _run_file_prefix(args)
    config_snapshot = _write_config_snapshot(out_dir, args, cfg, method_cfg, data_spec, backend_info, method_budget_fields, capability_info)
    train_history_path = out_dir / f"{run_prefix}_train_history.json"
    train_history_path.write_text(json.dumps(_json_safe(train_history), indent=2), encoding="utf-8")
    checkpoint_path = ""
    if args.save_checkpoint:
        ckpt = out_dir / f"{run_prefix}.pt"
        model.save(ckpt)
        checkpoint_path = str(ckpt)

    raw_rows, eval_totals = _evaluate_full_test_loader(
        model=model,
        loader=test_loader,
        args=args,
        out_dir=out_dir,
        config_snapshot=str(config_snapshot),
        checkpoint_path=checkpoint_path,
        train_time=train_time,
        train_history=train_history,
        backend_info=backend_info,
        capability_info=capability_info,
        train_dataset=train_dataset_for_fit,
        spec_dataset=spec_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        split_info=split_info,
        method_budget_fields=method_budget_fields,
    )
    summary = _summarize_run(
        raw_rows,
        eval_totals,
        args,
        train_dataset_for_fit,
        spec_dataset,
        val_dataset,
        test_dataset,
        backend_info,
        capability_info,
        split_info,
        method_budget_fields,
    )
    summary.update(
        {
            "train_time": train_time,
            "num_params": int(model.parameter_count() if hasattr(model, "parameter_count") else num_parameters(model)),
            "config_path": str(config_snapshot),
            "checkpoint_path": checkpoint_path,
            "train_history_path": str(train_history_path),
            "commit_hash": _commit_hash(),
            "dry_run": bool(args.dry_run),
            "synthetic_data": bool(args.synthetic_data),
            "experiment_mode": args.experiment_mode,
            "run_id": args.run_id,
            "run_name": args.run_name,
            "train_history": json.dumps(_json_safe(train_history)),
        }
    )
    append_result_jsonl(out_dir / "results_summary.jsonl", _json_safe(summary))
    append_result_csv(out_dir / "results_summary.csv", _json_safe(summary))
    latest_path = out_dir / (f"{run_prefix}_results_summary_latest.csv" if args.run_id else "results_summary_latest.csv")
    _write_latest_csv(latest_path, _json_safe(summary))
    (out_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    if args.run_id:
        (out_dir / f"{run_prefix}_summary.json").write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


def _validate_mode(args: argparse.Namespace) -> None:
    if args.experiment_mode == "paper" and (args.dry_run or args.synthetic_data or args.allow_synthetic_fallback or args.prefer_test):
        raise ValueError("paper mode cannot use --dry-run, --synthetic-data, --allow-synthetic-fallback, or --prefer-test")
    if args.dry_run and args.experiment_mode == "paper":
        raise ValueError("--dry-run is restricted to smoke/debug modes")
    if args.synthetic_data and args.experiment_mode == "paper":
        raise ValueError("--synthetic-data is restricted to smoke/debug modes")


def _resolve_physics_metric_mode(args: argparse.Namespace, cfg: dict[str, Any]) -> str:
    if args.physics_metric_mode:
        return str(args.physics_metric_mode)
    configured = cfg.get("physics_metric_mode")
    if configured:
        mode = str(configured)
        if mode not in {"per_sample", "per_batch"}:
            raise ValueError(f"physics_metric_mode must be per_sample or per_batch, got {mode!r}")
        return mode
    return "per_sample"


def _resolve_data_loading_mode(args: argparse.Namespace) -> str:
    if args.data_loading_mode:
        return str(args.data_loading_mode)
    return "lazy" if args.experiment_mode == "paper" else "eager"


def _parse_override_value(value: str) -> Any:
    text = str(value)
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _method_budget_fields(method_cfg: dict[str, Any], baseline: str) -> dict[str, Any]:
    steps = int(method_cfg.get("steps", 0) or 0)
    refine_steps = int(method_cfg.get("refine_steps", 0) or 0)
    particles = int(method_cfg.get("particles", 0) or 0)
    labels = []
    if refine_steps > 0 and baseline == "vivid":
        labels.append(f"refine_steps={refine_steps}")
    if steps > 0:
        labels.append(f"steps={steps}")
    if particles > 0 and baseline == "pc_bnn":
        labels.append(f"particles={particles}")
    if refine_steps > 0 and baseline != "vivid":
        labels.append(f"refine_steps={refine_steps}")
    if particles > 0 and baseline != "pc_bnn":
        labels.append(f"particles={particles}")
    return {
        "steps": steps,
        "refine_steps": refine_steps,
        "particles": particles,
        "method_budget_label": ",".join(labels),
    }


def _experiment_fields(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment_kind": args.experiment_kind,
        "ablation_factor": args.ablation_factor,
        "task_group": args.task_group,
    }


def _split_load_full_trajectory(args: argparse.Namespace, split: str) -> bool:
    if args.load_full_trajectory:
        return True
    if args.baseline in {"var4d", "vivid"}:
        return True
    return False


def _effective_sizes(args: argparse.Namespace) -> tuple[int, int, int]:
    train_size = int(args.train_size)
    val_size = int(args.val_size)
    test_size = int(args.test_size)
    if args.dry_run:
        train_size = min(train_size, 16)
        val_size = min(val_size, 4)
        test_size = min(test_size, 8)
    return train_size, val_size, test_size


def _make_val_dataset_if_requested(
    registry,
    args: argparse.Namespace,
    val_size: int,
    train_size: int,
    use_sensors: bool,
) -> tuple[PDEBatchDataset | None, dict[str, Any]]:
    split_info: dict[str, Any] = {
        "train_requested_size": int(train_size),
        "effective_train_size": int(train_size),
        "val_requested_size": int(val_size),
        "val_split_source": "none",
        "val_from_train_offset": None,
    }
    if val_size <= 0:
        return None, split_info
    if args.synthetic_data:
        val_dataset = _make_split_dataset(
            registry,
            args,
            "val",
            val_size,
            use_sensors,
            synthetic_seed=args.seed * 1000 + 17,
        )
        split_info["val_split_source"] = "synthetic_independent_val"
        return val_dataset, split_info

    try:
        val_dataset = _make_split_dataset(
            registry,
            args,
            "val",
            val_size,
            use_sensors,
            synthetic_seed=args.seed * 1000 + 17,
        )
        split_info["val_split_source"] = str(val_dataset.batch.metadata.get("split_source", "independent_val"))
        return val_dataset, split_info
    except FileNotFoundError:
        if train_size <= val_size:
            raise ValueError(f"Cannot reserve val_size={val_size} from train_size={train_size}; reduce VAL_SIZE or provide an independent val file.")
        effective_train_size = int(train_size - val_size)
        split_info["effective_train_size"] = effective_train_size
        split_info["val_split_source"] = "deterministic_train_subset"
        split_info["val_from_train_offset"] = effective_train_size
        val_dataset = _make_split_dataset(
            registry,
            args,
            "val",
            val_size,
            use_sensors,
            synthetic_seed=args.seed * 1000 + 17,
            val_from_train_offset=effective_train_size,
            strict_size_override=True,
        )
        return val_dataset, split_info


def _make_split_dataset(
    registry,
    args: argparse.Namespace,
    split: str,
    size: int,
    use_sensors: bool,
    synthetic_seed: int,
    sample_offset: int = 0,
    val_from_train_offset: int | None = None,
    strict_size_override: bool | None = None,
) -> PDEBatchDataset:
    if args.synthetic_data:
        raw = registry.synthetic_raw(
            args.pde,
            n=size,
            resolution=args.synthetic_resolution,
            split=split,
            seed=synthetic_seed,
            scalar_param_mode=args.scalar_param_mode,
        )
        batch = registry.make_task(
            raw,
            args.pde,
            args.task,
            num_sensors=args.num_sensors if use_sensors else None,
            sensor_mode=args.sensor_mode,
            noise_level=args.noise_level,
            seed=args.seed,
            experiment_mode=args.experiment_mode,
        )
        return PDEBatchDataset(batch)
    return registry.make_dataset(
        args.pde,
        args.data_root,
        args.task,
        split=split,
        max_samples=size,
        train_shards=args.train_shards,
        sample_offset=sample_offset,
        val_from_train_offset=val_from_train_offset,
        num_sensors=args.num_sensors if use_sensors else None,
        sensor_mode=args.sensor_mode,
        noise_level=args.noise_level,
        seed=args.seed,
        prefer_test=args.prefer_test and split == "test",
        synthetic_if_missing=args.allow_synthetic_fallback,
        synthetic_resolution=args.synthetic_resolution,
        synthetic_seed=synthetic_seed,
        scalar_param_mode=args.scalar_param_mode,
        data_loading_mode=args.data_loading_mode,
        load_full_trajectory=_split_load_full_trajectory(args, split),
        experiment_mode=args.experiment_mode,
        strict_size=args.strict_size if strict_size_override is None else bool(strict_size_override),
    )


def _evaluate_full_test_loader(
    model,
    loader,
    args,
    out_dir: Path,
    config_snapshot: str,
    checkpoint_path: str,
    train_time: float,
    train_history: dict[str, Any],
    backend_info: dict[str, Any],
    capability_info: dict[str, Any],
    train_dataset: PDEBatchDataset,
    spec_dataset: PDEBatchDataset,
    val_dataset: PDEBatchDataset | None,
    test_dataset: PDEBatchDataset,
    split_info: dict[str, Any],
    method_budget_fields: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    inference_time_total = 0.0
    inference_optimization_time_total = 0.0
    raw_path = out_dir / "results_raw.jsonl"
    grad_enabled = args.baseline in {"pinn_sparse", "pc_bnn", "pde_opt", "var4d", "vivid"}
    for batch_index, batch in enumerate(loader):
        start = time.perf_counter()
        eval_batch = _to_device_batch_for_eval(batch, args.device)
        with torch.set_grad_enabled(grad_enabled):
            pred = model.predict(eval_batch)
        elapsed = time.perf_counter() - start
        _copy_eval_metadata(batch, eval_batch)
        pred_cpu = pred.detach().cpu()
        target = batch.target_fields.detach().cpu()
        inf_opt = float(batch.metadata.get("inference_optimization_time", 0.0) or 0.0)
        batch_n = int(target.shape[0])
        rel_values = _relative_l2_values(pred_cpu, target)
        input_or_coeff_values = _relative_l2_input_or_coeff_values(args.task, pred_cpu, target)
        mse_values = _mse_values(pred_cpu, target)
        mae_values = _mae_values(pred_cpu, target)
        metric_payload = _batch_metric_payload(pred_cpu, target, batch, args)
        row = {
            "run_id": args.run_id,
            "run_name": args.run_name,
            **_experiment_fields(args),
            **method_budget_fields,
            "pde": args.pde,
            "task": args.task,
            "baseline": args.baseline,
            "seed": args.seed,
            "batch_index": batch_index,
            "sample_count": batch_n,
            "split": "test",
            "train_size": int(split_info["effective_train_size"]),
            "train_requested_size": int(split_info["train_requested_size"]),
            "effective_train_size": int(split_info["effective_train_size"]),
            "train_size_loaded_for_fit": int(split_info["train_size_loaded_for_fit"]),
            "train_size_loaded_for_spec": int(split_info["train_size_loaded_for_spec"]),
            "val_size": len(val_dataset) if val_dataset is not None else 0,
            "val_requested_size": int(split_info["val_requested_size"]),
            "val_split_source": split_info["val_split_source"],
            "val_from_train_offset": split_info["val_from_train_offset"],
            "test_size": len(test_dataset),
            "train_shards": args.train_shards,
            "data_loading_mode": args.data_loading_mode,
            "load_full_trajectory": bool(args.load_full_trajectory),
            "loaded_full_trajectory": bool(_batch_loaded_full_trajectory(batch)),
            "train_size_requested": int(split_info["train_size_requested"]),
            "train_size_loaded_in_memory": int(split_info["train_size_loaded_in_memory"]),
            "data_root": args.data_root,
            "file_paths_summary": json.dumps(_file_summary(train_dataset, val_dataset, test_dataset), sort_keys=True),
            "scalar_param_mode": args.scalar_param_mode,
            "scalar_param_mode_requested": args.scalar_param_mode,
            "scalar_param_mode_effective": args.scalar_param_mode,
            "scalar_params_used_as_input": _scalar_params_used_as_input(batch),
            "pde_params_available": json.dumps(sorted(batch.pde_params), sort_keys=True),
            "input_channel_names": json.dumps(batch.input_channel_names),
            "target_channel_names": json.dumps(batch.target_channel_names),
            "num_sensors": args.num_sensors if args.task.startswith("sparse") else 0,
            "requested_sensor_mode": args.sensor_mode if args.task.startswith("sparse") else "none",
            "effective_sensor_mode": str(batch.metadata.get("effective_sensor_mode", args.sensor_mode if args.task.startswith("sparse") else "none")),
            "sensor_mode": str(batch.metadata.get("effective_sensor_mode", args.sensor_mode if args.task.startswith("sparse") else "none")),
            "time_varying_sensor_valid": bool(batch.metadata.get("time_varying_sensor_valid", True)),
            "noise_level": args.noise_level,
            "mask_id": batch.metadata.get("mask_id", ""),
            "backend_used": backend_info["backend_used"],
            "official_backend": backend_info["official_backend"],
            "fallback_used": backend_info["fallback_used"],
            "backend_warning": backend_info["backend_warning"],
            **_implementation_fields(backend_info),
            **_capability_fields(capability_info),
            **_native_data_interface_fields(batch, pred_cpu),
            "metric_granularity": args.physics_metric_mode,
            "relative_l2_solution": _mean_list(rel_values),
            "relative_l2_solution_values": json.dumps(rel_values),
            "relative_l2_input_or_coeff": _mean_list(input_or_coeff_values),
            "relative_l2_input_or_coeff_values": json.dumps(input_or_coeff_values),
            "mse": _mean_list(mse_values),
            "mse_values": json.dumps(mse_values),
            "mae": _mean_list(mae_values),
            "mae_values": json.dumps(mae_values),
            "obs_mse": metric_payload["obs_mse"],
            "obs_mse_clean": metric_payload["obs_mse_clean"],
            "obs_mse_noisy": metric_payload["obs_mse_noisy"],
            "pde_residual": metric_payload["pde_residual"],
            "bc_residual": metric_payload["bc_residual"],
            "ic_residual": metric_payload["ic_residual"],
            "physics_loss": metric_payload["physics_loss"],
            "residual_mode": metric_payload["residual_mode"],
            "residual_mode_counts": json.dumps(metric_payload["residual_mode_counts"], sort_keys=True),
            "assimilation_mode": str(batch.metadata.get("assimilation_mode", "")),
            "assimilation_mode_counts": json.dumps(_assimilation_mode_counts(batch), sort_keys=True),
            "inverse_observation_operator_used": bool(batch.metadata.get("inverse_observation_operator_used", False)),
            "posterior_particles": int(batch.metadata.get("posterior_particles", 0) or 0),
            "train_time": train_time,
            "inference_time": elapsed,
            "inference_optimization_time": inf_opt,
            "num_params": int(model.parameter_count() if hasattr(model, "parameter_count") else num_parameters(model)),
            "config_path": config_snapshot,
            "checkpoint_path": checkpoint_path,
            "commit_hash": _commit_hash(),
            "dry_run": bool(args.dry_run),
            "synthetic_data": bool(args.synthetic_data),
            "experiment_mode": args.experiment_mode,
            "global_sample_ids": json.dumps(batch.global_sample_ids),
            "input_shape": json.dumps(list(batch.input_fields.shape)),
            "target_shape": json.dumps(list(batch.target_fields.shape)),
            "pred_shape": json.dumps(list(pred_cpu.shape)),
            "train_history": json.dumps(_json_safe(train_history)),
        }
        for key in ("obs_mse", "obs_mse_clean", "obs_mse_noisy", "pde_residual", "bc_residual", "ic_residual", "physics_loss"):
            values_key = f"{key}_values"
            if values_key in metric_payload:
                row[values_key] = json.dumps(metric_payload[values_key])
        if tuple(pred_cpu.shape) != tuple(target.shape):
            raise RuntimeError(f"Prediction shape {tuple(pred_cpu.shape)} != target shape {tuple(target.shape)}")
        rows.append(row)
        append_result_jsonl(raw_path, _json_safe(row))
        inference_time_total += elapsed
        inference_optimization_time_total += inf_opt
    return rows, {
        "inference_time_total": inference_time_total,
        "inference_optimization_time_total": inference_optimization_time_total,
    }


def _batch_metric_payload(pred: torch.Tensor, target: torch.Tensor, batch: PDEBatch, args: argparse.Namespace) -> dict[str, Any]:
    if args.physics_metric_mode == "per_batch":
        metric_meta = {
            "input_fields": batch.input_fields,
            "full_tensor": batch.full_tensor,
            "task": batch.task,
            **batch.metadata,
        }
        physics_metrics = physics_loss_metric(pred, args.pde, dict(metric_meta))
        mode = str(physics_metrics["mode"])
        clean = _obs_mse_clean(pred, target, batch)
        noisy = _obs_mse_noisy(pred, batch)
        return {
            "obs_mse": clean,
            "obs_mse_clean": clean,
            "obs_mse_noisy": noisy,
            "pde_residual": _tensor_float(physics_metrics["interior"]),
            "bc_residual": _tensor_float(physics_metrics["bc"]),
            "ic_residual": _tensor_float(physics_metrics["ic"]),
            "physics_loss": _tensor_float(physics_metrics["total"]),
            "residual_mode": mode,
            "residual_mode_counts": {mode: int(target.shape[0])},
        }

    values: dict[str, list[float]] = {
        "obs_mse": [],
        "obs_mse_clean": [],
        "obs_mse_noisy": [],
        "pde_residual": [],
        "bc_residual": [],
        "ic_residual": [],
        "physics_loss": [],
    }
    residual_counts: Counter[str] = Counter()
    for item in range(int(target.shape[0])):
        item_batch = slice_pde_batch(batch, item)
        pred_i = pred[item : item + 1]
        target_i = target[item : item + 1]
        meta_i = {
            "input_fields": item_batch.input_fields,
            "full_tensor": item_batch.full_tensor,
            "task": item_batch.task,
            **item_batch.metadata,
        }
        physics_metrics = physics_loss_metric(pred_i, args.pde, dict(meta_i))
        clean = _obs_mse_clean(pred_i, target_i, item_batch)
        noisy = _obs_mse_noisy(pred_i, item_batch)
        values["obs_mse"].append(clean)
        values["obs_mse_clean"].append(clean)
        values["obs_mse_noisy"].append(noisy)
        values["pde_residual"].append(_tensor_float(physics_metrics["interior"]))
        values["bc_residual"].append(_tensor_float(physics_metrics["bc"]))
        values["ic_residual"].append(_tensor_float(physics_metrics["ic"]))
        values["physics_loss"].append(_tensor_float(physics_metrics["total"]))
        residual_counts[str(physics_metrics["mode"])] += 1
    payload: dict[str, Any] = {
        "residual_mode": _mode_label(residual_counts),
        "residual_mode_counts": dict(residual_counts),
    }
    for key, metric_values in values.items():
        payload[key] = _mean_list(metric_values)
        payload[f"{key}_values"] = metric_values
    return payload


def _mode_label(counts: Counter[str]) -> str:
    if not counts:
        return ""
    if len(counts) == 1:
        return next(iter(counts))
    return "mixed"


def _summarize_run(
    rows: list[dict[str, Any]],
    eval_totals: dict[str, float],
    args: argparse.Namespace,
    train_dataset: PDEBatchDataset,
    spec_dataset: PDEBatchDataset,
    val_dataset: PDEBatchDataset | None,
    test_dataset: PDEBatchDataset,
    backend_info: dict[str, Any],
    capability_info: dict[str, Any],
    split_info: dict[str, Any],
    method_budget_fields: dict[str, Any],
) -> dict[str, Any]:
    metric_keys = [
        "relative_l2_solution",
        "relative_l2_input_or_coeff",
        "mse",
        "mae",
        "obs_mse",
        "obs_mse_clean",
        "obs_mse_noisy",
        "pde_residual",
        "bc_residual",
        "ic_residual",
        "physics_loss",
    ]
    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "run_name": args.run_name,
        **_experiment_fields(args),
        **method_budget_fields,
        "pde": args.pde,
        "task": args.task,
        "baseline": args.baseline,
        "seed": args.seed,
        "split": "test",
        "train_size": int(split_info["effective_train_size"]),
        "train_requested_size": int(split_info["train_requested_size"]),
        "effective_train_size": int(split_info["effective_train_size"]),
        "train_size_loaded_for_fit": int(split_info["train_size_loaded_for_fit"]),
        "train_size_loaded_for_spec": int(split_info["train_size_loaded_for_spec"]),
        "val_size": len(val_dataset) if val_dataset is not None else 0,
        "val_requested_size": int(split_info["val_requested_size"]),
        "val_split_source": split_info["val_split_source"],
        "val_from_train_offset": split_info["val_from_train_offset"],
        "test_size": len(test_dataset),
        "train_shards": args.train_shards,
        "data_loading_mode": args.data_loading_mode,
        "load_full_trajectory": bool(args.load_full_trajectory),
        "loaded_full_trajectory": bool(getattr(test_dataset, "loaded_full_trajectory", _batch_loaded_full_trajectory(test_dataset.batch))),
        "train_size_requested": int(split_info["train_size_requested"]),
        "train_size_loaded_in_memory": int(split_info["train_size_loaded_in_memory"]),
        "data_root": args.data_root,
        "file_paths_summary": json.dumps(_file_summary(train_dataset, val_dataset, test_dataset), sort_keys=True),
        "scalar_param_mode": args.scalar_param_mode,
        "scalar_param_mode_requested": args.scalar_param_mode,
        "scalar_param_mode_effective": args.scalar_param_mode,
        "scalar_params_used_as_input": _scalar_params_used_as_input(test_dataset.batch),
        "pde_params_available": json.dumps(sorted(test_dataset.batch.pde_params), sort_keys=True),
        "input_channel_names": json.dumps(test_dataset.batch.input_channel_names),
        "target_channel_names": json.dumps(test_dataset.batch.target_channel_names),
        "num_sensors": args.num_sensors if args.task.startswith("sparse") else 0,
        "requested_sensor_mode": args.sensor_mode if args.task.startswith("sparse") else "none",
        "effective_sensor_mode": str(test_dataset.batch.metadata.get("effective_sensor_mode", args.sensor_mode if args.task.startswith("sparse") else "none")),
        "sensor_mode": str(test_dataset.batch.metadata.get("effective_sensor_mode", args.sensor_mode if args.task.startswith("sparse") else "none")),
        "time_varying_sensor_valid": bool(test_dataset.batch.metadata.get("time_varying_sensor_valid", True)),
        "noise_level": args.noise_level,
        "mask_id": test_dataset.batch.metadata.get("mask_id", ""),
        "backend_used": backend_info["backend_used"],
        "official_backend": backend_info["official_backend"],
        "fallback_used": backend_info["fallback_used"],
        "backend_warning": backend_info["backend_warning"],
        **_implementation_fields(backend_info),
        **_capability_fields(capability_info),
        **_native_data_interface_fields(test_dataset.batch, None),
        "metric_granularity": args.physics_metric_mode,
        "batch_count": len(rows),
        "residual_mode_counts": json.dumps(dict(_residual_mode_counter(rows)), sort_keys=True),
        "assimilation_mode_counts": json.dumps(dict(_assimilation_mode_counter(rows)), sort_keys=True),
        "inverse_observation_operator_used": any(bool(row.get("inverse_observation_operator_used", False)) for row in rows),
        "posterior_particles": max((int(row.get("posterior_particles", 0) or 0) for row in rows), default=0),
        "inference_time_total": eval_totals["inference_time_total"],
        "inference_time_per_sample": eval_totals["inference_time_total"] / max(len(test_dataset), 1),
        "inference_optimization_time_total": eval_totals["inference_optimization_time_total"],
        "inference_optimization_time_per_sample": eval_totals["inference_optimization_time_total"] / max(len(test_dataset), 1),
    }
    for key in metric_keys:
        values = []
        for row in rows:
            values_key = f"{key}_values"
            if values_key in row:
                raw_values = json.loads(row[values_key])
                values.extend(raw_values)
            elif row.get("metric_granularity") == "per_batch":
                values.append(row[key])
            else:
                values.extend([row[key]] * int(row.get("sample_count", 1) or 1))
        stats = _metric_stats(values)
        for suffix, value in stats.items():
            summary[f"{key}_{suffix}"] = value
    return summary


def _residual_mode_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if "residual_mode_counts" in row:
            try:
                parsed = json.loads(row["residual_mode_counts"]) if isinstance(row["residual_mode_counts"], str) else row["residual_mode_counts"]
                counts.update({str(k): int(v) for k, v in parsed.items()})
                continue
            except Exception:
                pass
        counts[str(row.get("residual_mode", ""))] += int(row.get("sample_count", 1) or 1)
    return counts


def _assimilation_mode_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = row.get("assimilation_mode_counts")
        if value:
            try:
                parsed = json.loads(value) if isinstance(value, str) else value
                counts.update({str(k): int(v) for k, v in parsed.items()})
                continue
            except Exception:
                pass
        mode = str(row.get("assimilation_mode", ""))
        if mode:
            counts[mode] += int(row.get("sample_count", 1) or 1)
    return counts


def _assimilation_mode_counts(batch: PDEBatch) -> dict[str, int]:
    mode = str(batch.metadata.get("assimilation_mode", ""))
    return {mode: int(batch.target_fields.shape[0])} if mode else {}


def _to_device_batch_for_eval(batch: PDEBatch, device: str) -> PDEBatch:
    from baselines.methods.base import _to_device_batch

    return _to_device_batch(batch, torch.device(device))


def _copy_eval_metadata(dst: PDEBatch, src: PDEBatch) -> None:
    for key in (
        "inference_optimization_time",
        "assimilation_mode",
        "inverse_observation_operator_used",
        "posterior_particles",
        "official_alignment_level",
    ):
        if key in src.metadata:
            dst.metadata[key] = src.metadata[key]


def _batch_loaded_full_trajectory(batch: PDEBatch) -> bool:
    return bool(batch.full_tensor.ndim == 5 or isinstance(batch.metadata.get("full_trajectory"), torch.Tensor))


def _scalar_params_used_as_input(batch: PDEBatch) -> bool:
    keys = set(batch.pde_params)
    if not keys:
        return False
    input_names = set(batch.input_channel_names)
    return any(key in input_names for key in keys)


def _scalar_params_used_as_input_from_spec(data_spec: dict[str, Any]) -> bool:
    metadata = data_spec.get("metadata", {}) if isinstance(data_spec, dict) else {}
    params = metadata.get("pde_params_available", [])
    names = data_spec.get("input_channel_names", []) if isinstance(data_spec, dict) else []
    return bool(set(params).intersection(set(names)))


def _relative_l2_values(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> list[float]:
    pred, target = _align(pred, target)
    diff = torch.linalg.vector_norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    denom = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(eps)
    return [float(x) for x in (diff / denom).detach().cpu()]


def _relative_l2_input_or_coeff_values(task: str, pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    if task in {"inverse", "sparse_inverse"}:
        return _relative_l2_values(pred, target)
    return [float("nan")] * int(target.shape[0])


def _obs_mse_clean(pred: torch.Tensor, target: torch.Tensor, batch: PDEBatch) -> float:
    if batch.task == "sparse_inverse":
        return float("nan")
    return float(obs_mse(pred, target, batch.mask).detach().cpu())


def _obs_mse_noisy(pred: torch.Tensor, batch: PDEBatch) -> float:
    if batch.task == "sparse_inverse" or batch.mask is None or batch.obs_values is None:
        return float("nan")
    try:
        c = min(pred.shape[1], batch.obs_values.shape[-1])
        spatial_mask = batch.mask[0].bool().reshape(-1).to(pred.device)
        flat_idx = spatial_mask.nonzero(as_tuple=False).squeeze(-1)
        pred_obs = pred.reshape(pred.shape[0], pred.shape[1], -1).permute(0, 2, 1)[:, flat_idx, :c]
        obs = batch.obs_values.to(pred.device, pred.dtype)[..., :c]
        return float((pred_obs - obs).pow(2).mean().detach().cpu())
    except Exception:
        return float("nan")


def _mse_values(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    pred, target = _align(pred, target)
    return [float(x) for x in (pred - target).pow(2).reshape(pred.shape[0], -1).mean(dim=1).detach().cpu()]


def _mae_values(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    pred, target = _align(pred, target)
    return [float(x) for x in (pred - target).abs().reshape(pred.shape[0], -1).mean(dim=1).detach().cpu()]


def _align(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.shape == target.shape:
        return pred, target
    if pred.ndim == target.ndim and pred.shape[0] == target.shape[0]:
        slices = [slice(None)]
        for a, b in zip(pred.shape[1:], target.shape[1:]):
            slices.append(slice(0, min(a, b)))
        return pred[tuple(slices)], target[tuple(slices)]
    raise ValueError(f"Cannot align prediction shape {tuple(pred.shape)} with target shape {tuple(target.shape)}")


def _metric_stats(values: list[float]) -> dict[str, float | int]:
    cleaned = [float(v) for v in values if not _is_nan(v)]
    nan_count = len(values) - len(cleaned)
    n = len(cleaned)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "sem": float("nan"), "ci95": float("nan"), "n": 0, "nan_count": nan_count}
    mean = sum(cleaned) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in cleaned) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    sem = std / math.sqrt(n) if n > 0 else float("nan")
    return {"mean": mean, "std": std, "sem": sem, "ci95": 1.96 * sem, "n": n, "nan_count": nan_count}


def _mean_list(values: list[float]) -> float:
    finite = [v for v in values if not _is_nan(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except Exception:
        return False


def _tensor_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _uses_official_inverse_observation_operator(baseline: str, method_cfg: dict[str, Any]) -> bool:
    if baseline != "vivid":
        return bool(method_cfg.get("uses_official_inverse_observation_operator", False))
    mode = _official_request_mode(method_cfg)
    return bool(method_cfg.get("uses_official_inverse_observation_operator", False)) or mode in {
        "official",
        "official_or_skip",
        "official_aligned",
        "auto",
    }


def _preflight_backend_availability(
    args: argparse.Namespace,
    capability,
    method_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    if args.experiment_mode != "paper" or capability.implementation_required != "official":
        return None
    mode = _official_request_mode(method_cfg)
    if mode not in {"official", "official_or_skip", "official_architecture", "official_aligned"}:
        return None
    try:
        if args.baseline == "ifno" and capability.task_family in {"full_forward", "full_inverse"}:
            get_ifno_official_aligned_status()
            return None
        if args.baseline == "vivid" and capability.task_family == "time_varying_da":
            get_vivid_official_aligned_status()
            return None
        if args.baseline == "pc_bnn" and capability.task_family == "sparse_reconstruction":
            if args.pde.lower() != "shallow_water":
                raise OfficialImportError("official-aligned PC-BNN is only enabled for 2D three-channel shallow-water fields")
            get_pc_bnn_official_aligned_status()
            return None
    except OfficialImportError as exc:
        reason = f"official backend unavailable before dataset loading: {exc}"
        backend_info = _default_skip_backend_info(method_cfg, reason, adapter_status="official_backend_unavailable")
        backend_info["fallback_used"] = mode == "official_or_skip"
        if mode == "official_or_skip":
            return {"reason": reason, "backend_info": backend_info, "adapter_status": "official_backend_unavailable"}
        raise RuntimeError(reason) from exc
    return None


def _backend_info(model, method_cfg: dict[str, Any]) -> dict[str, Any]:
    official_backend = str(getattr(model, "official_backend", "local"))
    backend_used = str(getattr(model, "backend_used", "") or official_backend or "local")
    requested = str(method_cfg.get("official_backend", "auto")).lower()
    implementation_requested = str(getattr(model, "implementation_mode_requested", method_cfg.get("implementation_mode", requested)) or "")
    fallback_used = bool(getattr(model, "fallback_used", False))
    if requested in {"local", "none"}:
        fallback_used = False
    warning = str(getattr(model, "backend_warning", "") or "")
    if fallback_used and requested in {"auto", "official"} and not warning:
        warning = f"using fallback backend {backend_used}"
    return {
        "backend_used": backend_used,
        "official_backend": official_backend,
        "fallback_used": fallback_used,
        "backend_warning": warning,
        "implementation_mode_requested": implementation_requested,
        "implementation_mode_effective": str(getattr(model, "implementation_mode_effective", "adapted")),
        "implementation_source": str(getattr(model, "implementation_source", backend_used)),
        "official_repo": str(getattr(model, "official_repo", "")),
        "official_commit_or_version": str(getattr(model, "official_commit_or_version", "")),
        "official_import_path": str(getattr(model, "official_import_path", "")),
        "official_import_success": bool(getattr(model, "official_import_success", False)),
        "official_reimplementation_success": bool(getattr(model, "official_reimplementation_success", False)),
        "official_alignment_level": str(getattr(model, "official_alignment_level", "")),
        "official_alignment_notes": str(getattr(model, "official_alignment_notes", "")),
        "adapter_status": str(getattr(model, "adapter_status", "")),
    }


def _default_skip_backend_info(method_cfg: dict[str, Any], reason: str, adapter_status: str = "skipped") -> dict[str, Any]:
    requested = str(method_cfg.get("implementation_mode", method_cfg.get("official_backend", "auto")) or "auto").lower()
    return {
        "backend_used": "skipped",
        "official_backend": str(method_cfg.get("official_backend", "")),
        "fallback_used": False,
        "backend_warning": reason,
        "implementation_mode_requested": requested,
        "implementation_mode_effective": "skipped",
        "implementation_source": "skipped",
        "official_repo": "",
        "official_commit_or_version": "",
        "official_import_path": "",
        "official_import_success": False,
        "official_reimplementation_success": False,
        "official_alignment_level": "",
        "official_alignment_notes": "",
        "adapter_status": adapter_status,
    }


def _skip_paper_run(
    args: argparse.Namespace,
    capability,
    method_cfg: dict[str, Any],
    reason: str,
    *,
    backend_info: dict[str, Any] | None = None,
    adapter_status: str = "skipped",
) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend_info = dict(backend_info or _default_skip_backend_info(method_cfg, reason, adapter_status=adapter_status))
    backend_info["fallback_used"] = bool(backend_info.get("fallback_used", False))
    backend_info["backend_warning"] = str(backend_info.get("backend_warning") or reason)
    if adapter_status != "skipped" and not backend_info.get("adapter_status"):
        backend_info["adapter_status"] = adapter_status
    row = capability_skip_row(capability, task_group=args.task_group)
    row.update(
        {
            "reason": reason,
            "unsupported_reason": reason,
            "paper_table_eligible": False,
            "backend_used": backend_info.get("backend_used", "skipped"),
            "official_backend": backend_info.get("official_backend", ""),
            "fallback_used": bool(backend_info.get("fallback_used", False)),
            "backend_warning": backend_info.get("backend_warning", reason),
            **_implementation_fields(backend_info),
        }
    )
    append_result_jsonl(out_dir / "skipped_combinations.jsonl", _json_safe(row))
    print(json.dumps(_json_safe({"status": "skipped", **row}), indent=2, sort_keys=True))
    return row


def _official_request_mode(method_cfg: dict[str, Any]) -> str:
    implementation_mode = str(method_cfg.get("implementation_mode", "") or "").lower()
    if implementation_mode in {"official", "official_or_skip", "official_architecture", "official_aligned", "adapted", "canonical_math"}:
        return implementation_mode
    backend = str(method_cfg.get("official_backend", "auto") or "auto").lower()
    if backend == "official":
        return "official"
    return implementation_mode or backend


def _requested_official(method_cfg: dict[str, Any]) -> bool:
    return _official_request_mode(method_cfg) in {"official", "official_or_skip", "official_architecture", "official_aligned"}


def _paper_table_eligibility(capability, backend_info: dict[str, Any]) -> bool:
    return capability_paper_table_eligible(
        capability,
        str(backend_info.get("implementation_mode_effective", "")),
        backend_info=backend_info,
    )


def _implementation_fields(backend_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "implementation_mode_requested": backend_info.get("implementation_mode_requested", ""),
        "implementation_mode_effective": backend_info.get("implementation_mode_effective", ""),
        "implementation_source": backend_info.get("implementation_source", ""),
        "official_repo": backend_info.get("official_repo", ""),
        "official_commit_or_version": backend_info.get("official_commit_or_version", ""),
        "official_import_path": backend_info.get("official_import_path", ""),
        "official_import_success": bool(backend_info.get("official_import_success", False)),
        "official_reimplementation_success": bool(backend_info.get("official_reimplementation_success", False)),
        "official_alignment_level": backend_info.get("official_alignment_level", ""),
        "official_alignment_notes": backend_info.get("official_alignment_notes", ""),
        "adapter_status": backend_info.get("adapter_status", ""),
    }


def _capability_fields(capability_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "capability_status": capability_info.get("support_status", ""),
        "support_status": capability_info.get("support_status", ""),
        "implementation_required": capability_info.get("implementation_required", ""),
        "task_family": capability_info.get("task_family", ""),
        "unsupported_reason": capability_info.get("unsupported_reason", ""),
        "capability_reason": capability_info.get("reason", ""),
        "citation_key": capability_info.get("citation_key", ""),
        "source_key": capability_info.get("source_key", ""),
        "notes_for_paper": capability_info.get("notes_for_paper", ""),
        "official_architecture_allowed": bool(capability_info.get("official_architecture_allowed", False)),
        "official_aligned_allowed": bool(capability_info.get("official_aligned_allowed", False)),
        "eligible_implementation_modes": json.dumps(list(capability_info.get("eligible_implementation_modes", []))),
        "paper_table_eligible": bool(capability_info.get("paper_table_eligible", False)),
    }


def _native_data_interface_fields(batch: PDEBatch, pred: torch.Tensor | None) -> dict[str, Any]:
    original = batch.metadata.get("original_input_fields")
    observed_names = batch.metadata.get("observation_source_channel_names", batch.input_channel_names)
    if isinstance(observed_names, (tuple, list)):
        observation_field_name = ",".join(str(x) for x in observed_names)
    else:
        observation_field_name = str(observed_names or "")
    predicted_field_name = ",".join(str(x) for x in batch.target_channel_names)
    raw_shape = list(original.shape) if isinstance(original, torch.Tensor) else list(batch.input_fields.shape)
    row = {
        "raw_input_shape": json.dumps(raw_shape),
        "official_input_shape": json.dumps(list(batch.input_fields.shape)),
        "native_input_shape": json.dumps(list(batch.input_fields.shape)),
        "target_shape_native": json.dumps(list(batch.target_fields.shape)),
        "observation_field_name": observation_field_name,
        "predicted_field_name": predicted_field_name,
        "scalar_pde_params_part_of_input": _scalar_params_used_as_input(batch),
        "full_trajectory_loaded": _batch_loaded_full_trajectory(batch),
        "sensors_static_random_grid_time_varying": str(batch.metadata.get("effective_sensor_mode", "none")),
    }
    if pred is not None:
        row["predicted_shape_native"] = json.dumps(list(pred.shape))
    return row


def _file_summary(train_dataset: PDEBatchDataset, val_dataset: PDEBatchDataset | None, test_dataset: PDEBatchDataset) -> dict[str, Any]:
    def one(ds: Any | None) -> dict[str, Any]:
        if ds is None:
            return {"count": 0, "first": []}
        paths = list(getattr(ds, "file_paths", []) or ds.batch.file_paths)
        return {"count": len(paths), "first": paths[:3]}

    return {"train": one(train_dataset), "val": one(val_dataset), "test": one(test_dataset)}


def _run_file_prefix(args: argparse.Namespace) -> str:
    if args.run_id:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in args.run_id)
    return f"{args.baseline}_{args.pde}_{args.task}_seed{args.seed}"


def _write_config_snapshot(
    out_dir: Path,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    method_cfg: dict[str, Any],
    data_spec: dict[str, Any],
    backend_info: dict[str, Any],
    method_budget_fields: dict[str, Any],
    capability_info: dict[str, Any],
) -> Path:
    prefix = _run_file_prefix(args)
    path = out_dir / f"{prefix}_config.json"
    payload = {
        "args": vars(args),
        "config": cfg,
        "method": method_cfg,
        "data_spec": _json_safe(data_spec),
        "backend": backend_info,
        "capability": capability_info,
        "scalar_param_mode_requested": args.scalar_param_mode,
        "scalar_param_mode_effective": args.scalar_param_mode,
        "scalar_params_used_as_input": _scalar_params_used_as_input_from_spec(data_spec),
        "data_loading_mode": args.data_loading_mode,
        "load_full_trajectory": bool(args.load_full_trajectory),
        "run_id": args.run_id,
        "run_name": args.run_name,
        **_experiment_fields(args),
        **method_budget_fields,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.config:
        src = Path(args.config)
        if src.exists():
            shutil.copy2(src, out_dir / f"{prefix}_{src.name}")
    return path


def _write_latest_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return float(obj.detach().cpu())
        return {"tensor_shape": list(obj.shape)}
    if isinstance(obj, Path):
        return str(obj)
    try:
        if isinstance(obj, float) and math.isnan(obj):
            return float("nan")
    except Exception:
        pass
    return obj


if __name__ == "__main__":
    with warnings.catch_warnings(record=False):
        main()
