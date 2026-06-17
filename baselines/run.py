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

from baselines.common.data_adapter import PDEBatch, PDEBatchDataset, build_default_registry, pde_collate
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
from baselines.methods.pc_bnn import PCBNNBaseline
from baselines.methods.pde_opt import PDEOptBaseline
from baselines.methods.pinn_sparse import PINNSparseBaseline
from baselines.methods.recfno import RecFNOBaseline
from baselines.methods.senseiver import SenseiverBaseline
from baselines.methods.var4d import Var4DBaseline
from baselines.methods.vivid import VIVIDBaseline
from baselines.methods.voronoicnn import VoronoiCNNBaseline


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

PER_INSTANCE_BASELINES = {"pinn_sparse", "pc_bnn", "pde_opt", "var4d"}


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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true", help="Use deterministic synthetic data for smoke/debug tests.")
    parser.add_argument("--allow-synthetic-fallback", action="store_true", help="Fall back to synthetic data when requested real files are missing.")
    parser.add_argument("--synthetic-resolution", type=int, default=32)
    parser.add_argument("--prefer-test", action="store_true", help="Compatibility/debug option. Never use for paper training.")
    parser.add_argument("--scalar-param-mode", choices=["metadata", "materialize", "global"], default="metadata")
    parser.add_argument("--strict-size", action="store_true", help="Fail if requested split size exceeds available samples.")
    parser.add_argument("--save-checkpoint", action="store_true")
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
    merged["device"] = args.device
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
    method_cfg = build_method_config(cfg, args)

    train_size, val_size, test_size = _effective_sizes(args)
    registry = build_default_registry()
    use_sensors = args.task.startswith("sparse")

    train_dataset = _make_split_dataset(
        registry,
        args,
        "train",
        train_size,
        use_sensors,
        synthetic_seed=args.seed * 1000 + 11,
    )
    val_dataset = None
    if val_size > 0:
        val_dataset = _make_split_dataset(
            registry,
            args,
            "val",
            val_size,
            use_sensors,
            synthetic_seed=args.seed * 1000 + 17,
            sample_offset=train_size,
            val_from_train_offset=train_size,
        )
    test_dataset = _make_split_dataset(
        registry,
        args,
        "test",
        test_size,
        use_sensors,
        synthetic_seed=args.seed * 1000 + 23,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.baseline not in PER_INSTANCE_BASELINES, collate_fn=pde_collate)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pde_collate)
        if val_dataset is not None
        else None
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pde_collate)
    first_train_batch = next(iter(train_loader))
    data_spec = build_data_spec(first_train_batch)

    model = BASELINES[args.baseline]().build(method_cfg, data_spec).to(args.device)
    backend_info = _backend_info(model, method_cfg)
    if args.experiment_mode == "paper" and _requested_official(method_cfg) and backend_info["fallback_used"]:
        raise RuntimeError(
            f"{args.baseline} requested official backend for paper mode but used fallback backend "
            f"{backend_info['backend_used']!r}. Install/enable the official dependency or set official_backend:auto explicitly."
        )

    train_start = time.perf_counter()
    train_history: dict[str, Any] = {}
    if args.baseline not in PER_INSTANCE_BASELINES:
        train_history = model.fit(train_loader, val_loader)
    else:
        train_history = model.fit(train_loader, val_loader)
    train_time = time.perf_counter() - train_start

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = _write_config_snapshot(out_dir, args, cfg, method_cfg, data_spec, backend_info)
    train_history_path = out_dir / f"{args.baseline}_{args.pde}_{args.task}_seed{args.seed}_train_history.json"
    train_history_path.write_text(json.dumps(_json_safe(train_history), indent=2), encoding="utf-8")
    checkpoint_path = ""
    if args.save_checkpoint:
        ckpt = out_dir / f"{args.baseline}_{args.pde}_{args.task}_seed{args.seed}.pt"
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
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )
    summary = _summarize_run(raw_rows, eval_totals, args, train_dataset, val_dataset, test_dataset, backend_info)
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
            "train_history": json.dumps(_json_safe(train_history)),
        }
    )
    append_result_jsonl(out_dir / "results_summary.jsonl", _json_safe(summary))
    append_result_csv(out_dir / "results_summary.csv", _json_safe(summary))
    _write_latest_csv(out_dir / "results_summary_latest.csv", _json_safe(summary))

    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


def _validate_mode(args: argparse.Namespace) -> None:
    if args.experiment_mode == "paper" and (args.dry_run or args.synthetic_data or args.allow_synthetic_fallback or args.prefer_test):
        raise ValueError("paper mode cannot use --dry-run, --synthetic-data, --allow-synthetic-fallback, or --prefer-test")
    if args.dry_run and args.experiment_mode == "paper":
        raise ValueError("--dry-run is restricted to smoke/debug modes")
    if args.synthetic_data and args.experiment_mode == "paper":
        raise ValueError("--synthetic-data is restricted to smoke/debug modes")


def _effective_sizes(args: argparse.Namespace) -> tuple[int, int, int]:
    train_size = int(args.train_size)
    val_size = int(args.val_size)
    test_size = int(args.test_size)
    if args.dry_run:
        train_size = min(train_size, 16)
        val_size = min(val_size, 4)
        test_size = min(test_size, 8)
    return train_size, val_size, test_size


def _make_split_dataset(
    registry,
    args: argparse.Namespace,
    split: str,
    size: int,
    use_sensors: bool,
    synthetic_seed: int,
    sample_offset: int = 0,
    val_from_train_offset: int | None = None,
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
        strict_size=args.strict_size,
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
    train_dataset: PDEBatchDataset,
    val_dataset: PDEBatchDataset | None,
    test_dataset: PDEBatchDataset,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    inference_time_total = 0.0
    inference_optimization_time_total = 0.0
    raw_path = out_dir / "results_raw.jsonl"
    grad_enabled = args.baseline in {"pinn_sparse", "pc_bnn", "pde_opt", "var4d", "vivid"}
    for batch_index, batch in enumerate(loader):
        start = time.perf_counter()
        with torch.set_grad_enabled(grad_enabled):
            pred = model.predict(_to_device_batch_for_eval(batch, args.device))
        elapsed = time.perf_counter() - start
        pred_cpu = pred.detach().cpu()
        target = batch.target_fields.detach().cpu()
        metric_meta = {
            "input_fields": batch.input_fields,
            "full_tensor": batch.full_tensor,
            "task": batch.task,
            **batch.metadata,
        }
        physics_metrics = physics_loss_metric(pred_cpu, args.pde, dict(metric_meta))
        inf_opt = float(batch.metadata.get("inference_optimization_time", 0.0) or 0.0)
        batch_n = int(target.shape[0])
        rel_values = _relative_l2_values(pred_cpu, target)
        mse_values = _mse_values(pred_cpu, target)
        mae_values = _mae_values(pred_cpu, target)
        row = {
            "pde": args.pde,
            "task": args.task,
            "baseline": args.baseline,
            "seed": args.seed,
            "batch_index": batch_index,
            "sample_count": batch_n,
            "split": "test",
            "train_size": len(train_dataset),
            "val_size": len(val_dataset) if val_dataset is not None else 0,
            "test_size": len(test_dataset),
            "train_shards": args.train_shards,
            "data_root": args.data_root,
            "file_paths_summary": json.dumps(_file_summary(train_dataset, val_dataset, test_dataset), sort_keys=True),
            "scalar_param_mode": args.scalar_param_mode,
            "pde_params_available": json.dumps(sorted(batch.pde_params), sort_keys=True),
            "input_channel_names": json.dumps(batch.input_channel_names),
            "target_channel_names": json.dumps(batch.target_channel_names),
            "num_sensors": args.num_sensors if args.task.startswith("sparse") else 0,
            "sensor_mode": args.sensor_mode if args.task.startswith("sparse") else "none",
            "noise_level": args.noise_level,
            "mask_id": batch.metadata.get("mask_id", ""),
            "backend_used": backend_info["backend_used"],
            "official_backend": backend_info["official_backend"],
            "fallback_used": backend_info["fallback_used"],
            "backend_warning": backend_info["backend_warning"],
            "relative_l2_solution": _mean_list(rel_values),
            "relative_l2_solution_values": json.dumps(rel_values),
            "relative_l2_input_or_coeff": float("nan"),
            "mse": _mean_list(mse_values),
            "mse_values": json.dumps(mse_values),
            "mae": _mean_list(mae_values),
            "mae_values": json.dumps(mae_values),
            "obs_mse": float(obs_mse(pred_cpu, target, batch.mask).detach().cpu()),
            "pde_residual": _tensor_float(physics_metrics["interior"]),
            "bc_residual": _tensor_float(physics_metrics["bc"]),
            "ic_residual": _tensor_float(physics_metrics["ic"]),
            "physics_loss": _tensor_float(physics_metrics["total"]),
            "residual_mode": str(physics_metrics["mode"]),
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


def _summarize_run(
    rows: list[dict[str, Any]],
    eval_totals: dict[str, float],
    args: argparse.Namespace,
    train_dataset: PDEBatchDataset,
    val_dataset: PDEBatchDataset | None,
    test_dataset: PDEBatchDataset,
    backend_info: dict[str, Any],
) -> dict[str, Any]:
    metric_keys = [
        "relative_l2_solution",
        "relative_l2_input_or_coeff",
        "mse",
        "mae",
        "obs_mse",
        "pde_residual",
        "bc_residual",
        "ic_residual",
        "physics_loss",
    ]
    summary: dict[str, Any] = {
        "pde": args.pde,
        "task": args.task,
        "baseline": args.baseline,
        "seed": args.seed,
        "split": "test",
        "train_size": len(train_dataset),
        "val_size": len(val_dataset) if val_dataset is not None else 0,
        "test_size": len(test_dataset),
        "train_shards": args.train_shards,
        "data_root": args.data_root,
        "file_paths_summary": json.dumps(_file_summary(train_dataset, val_dataset, test_dataset), sort_keys=True),
        "scalar_param_mode": args.scalar_param_mode,
        "pde_params_available": json.dumps(sorted(test_dataset.batch.pde_params), sort_keys=True),
        "input_channel_names": json.dumps(test_dataset.batch.input_channel_names),
        "target_channel_names": json.dumps(test_dataset.batch.target_channel_names),
        "num_sensors": args.num_sensors if args.task.startswith("sparse") else 0,
        "sensor_mode": args.sensor_mode if args.task.startswith("sparse") else "none",
        "noise_level": args.noise_level,
        "mask_id": test_dataset.batch.metadata.get("mask_id", ""),
        "backend_used": backend_info["backend_used"],
        "official_backend": backend_info["official_backend"],
        "fallback_used": backend_info["fallback_used"],
        "backend_warning": backend_info["backend_warning"],
        "residual_mode_counts": json.dumps(dict(_residual_mode_counter(rows)), sort_keys=True),
        "inference_time_total": eval_totals["inference_time_total"],
        "inference_time_per_sample": eval_totals["inference_time_total"] / max(len(test_dataset), 1),
        "inference_optimization_time_total": eval_totals["inference_optimization_time_total"],
        "inference_optimization_time_per_sample": eval_totals["inference_optimization_time_total"] / max(len(test_dataset), 1),
    }
    for key in metric_keys:
        values = []
        for row in rows:
            if key in {"relative_l2_solution", "mse", "mae"}:
                raw_values = json.loads(row[f"{key}_values"])
                values.extend(raw_values)
            else:
                values.extend([row[key]] * int(row.get("sample_count", 1) or 1))
        stats = _metric_stats(values)
        for suffix, value in stats.items():
            summary[f"{key}_{suffix}"] = value
    return summary


def _residual_mode_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get("residual_mode", ""))] += int(row.get("sample_count", 1) or 1)
    return counts


def _to_device_batch_for_eval(batch: PDEBatch, device: str) -> PDEBatch:
    from baselines.methods.base import _to_device_batch

    return _to_device_batch(batch, torch.device(device))


def _relative_l2_values(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> list[float]:
    pred, target = _align(pred, target)
    diff = torch.linalg.vector_norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    denom = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(eps)
    return [float(x) for x in (diff / denom).detach().cpu()]


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


def _backend_info(model, method_cfg: dict[str, Any]) -> dict[str, Any]:
    official_backend = str(getattr(model, "official_backend", "local"))
    backend_used = str(getattr(model, "backend_used", "") or official_backend or "local")
    if backend_used == "local" and official_backend != "local":
        backend_used = official_backend
    requested = str(method_cfg.get("official_backend", "auto")).lower()
    fallback_used = bool(getattr(model, "fallback_used", False) or backend_used == "local" or official_backend == "local")
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
    }


def _requested_official(method_cfg: dict[str, Any]) -> bool:
    return str(method_cfg.get("official_backend", "auto")).lower() == "official"


def _file_summary(train_dataset: PDEBatchDataset, val_dataset: PDEBatchDataset | None, test_dataset: PDEBatchDataset) -> dict[str, Any]:
    def one(ds: PDEBatchDataset | None) -> dict[str, Any]:
        if ds is None:
            return {"count": 0, "first": []}
        paths = list(ds.batch.file_paths)
        return {"count": len(paths), "first": paths[:3]}

    return {"train": one(train_dataset), "val": one(val_dataset), "test": one(test_dataset)}


def _write_config_snapshot(out_dir: Path, args: argparse.Namespace, cfg: dict[str, Any], method_cfg: dict[str, Any], data_spec: dict[str, Any], backend_info: dict[str, Any]) -> Path:
    path = out_dir / f"{args.baseline}_{args.pde}_{args.task}_seed{args.seed}_config.json"
    payload = {
        "args": vars(args),
        "config": cfg,
        "method": method_cfg,
        "data_spec": _json_safe(data_spec),
        "backend": backend_info,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.config:
        src = Path(args.config)
        if src.exists():
            shutil.copy2(src, out_dir / f"{args.baseline}_{args.pde}_{args.task}_seed{args.seed}_{src.name}")
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
