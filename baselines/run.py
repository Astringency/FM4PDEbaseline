from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from baselines.common.data_adapter import build_default_registry, pde_collate
from baselines.common.metrics import (
    append_result_csv,
    append_result_jsonl,
    mae,
    mse,
    num_parameters,
    obs_mse,
    physics_loss_metric,
    relative_l2,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FM4PDE baseline runner")
    parser.add_argument("--baseline", required=True, choices=sorted(BASELINES))
    parser.add_argument("--pde", required=True)
    parser.add_argument("--task", default="forward")
    parser.add_argument("--data-root", default="/home/tat512/C01Python/PDEdata")
    parser.add_argument("--config", default=None)
    parser.add_argument("--num-sensors", type=int, default=500)
    parser.add_argument("--sensor-mode", choices=["random", "fixed", "grid", "time_varying"], default="random")
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--test-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/baselines")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true", help="Use deterministic synthetic data for smoke tests.")
    parser.add_argument("--synthetic-resolution", type=int, default=32)
    parser.add_argument("--prefer-test", action="store_true", help="Load small test files instead of train files when available.")
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser.parse_args()


def load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_data_spec(batch) -> dict:
    branch_numel = int(batch.obs_values[0].numel()) if batch.obs_values is not None else int(batch.input_fields[0].numel())
    return {
        "pde": batch.pde_name,
        "task": batch.task,
        "input_shape": tuple(batch.input_fields.shape),
        "target_shape": tuple(batch.target_fields.shape),
        "input_channels": int(batch.input_fields.shape[1]),
        "target_channels": int(batch.target_fields.shape[1]),
        "input_numel": int(batch.input_fields[0].numel()),
        "target_numel": int(batch.target_fields[0].numel()),
        "branch_numel": branch_numel,
        "metadata": {k: v for k, v in batch.metadata.items() if not isinstance(v, torch.Tensor)},
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = load_yaml(args.config)
    method_cfg = dict(cfg.get("method", {}))
    method_cfg.update(
        {
            "epochs": args.epochs,
            "lr": args.lr,
            "device": args.device,
            "max_steps": 1 if args.dry_run else None,
        }
    )
    if args.dry_run:
        method_cfg.setdefault("steps", 1)
        method_cfg.setdefault("refine_steps", 1)
        args.train_size = min(args.train_size, 4)
        args.test_size = min(args.test_size, 2)

    registry = build_default_registry()
    use_sensors = args.task.startswith("sparse") or args.num_sensors > 0
    max_samples = args.train_size if not args.dry_run else min(args.train_size, 4)
    if args.synthetic_data:
        raw = registry.synthetic_raw(args.pde, n=max_samples, resolution=args.synthetic_resolution)
        batch_all = registry.make_task(
            raw,
            args.pde,
            args.task,
            num_sensors=args.num_sensors if use_sensors else None,
            sensor_mode=args.sensor_mode,
            noise_level=args.noise_level,
            seed=args.seed,
        )
        dataset = _batch_dataset(registry, batch_all)
    else:
        dataset = registry.make_dataset(
            args.pde,
            args.data_root,
            args.task,
            split="test" if args.prefer_test or args.dry_run else "train",
            max_samples=max_samples,
            num_sensors=args.num_sensors if use_sensors else None,
            sensor_mode=args.sensor_mode,
            noise_level=args.noise_level,
            seed=args.seed,
            prefer_test=args.prefer_test or args.dry_run,
            synthetic_if_missing=True,
            synthetic_resolution=args.synthetic_resolution,
        )
        batch_all = dataset.batch

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pde_collate)
    first_batch = next(iter(loader))
    data_spec = build_data_spec(first_batch)
    model = BASELINES[args.baseline]().build(method_cfg, data_spec).to(args.device)

    train_start = time.perf_counter()
    train_history = {}
    if not args.dry_run and args.baseline not in {"pinn_sparse", "pc_bnn", "pde_opt", "var4d"}:
        train_history = model.fit(loader)
    train_time = time.perf_counter() - train_start

    infer_start = time.perf_counter()
    model.eval()
    with torch.set_grad_enabled(args.baseline in {"pinn_sparse", "pc_bnn", "pde_opt", "var4d", "vivid"}):
        pred = model.predict(first_batch)
    inference_time = time.perf_counter() - infer_start

    metric_meta = {
        "input_fields": first_batch.input_fields,
        "full_tensor": first_batch.full_tensor,
        "task": first_batch.task,
        **first_batch.metadata,
    }
    rel = relative_l2(pred.detach(), first_batch.target_fields)
    physics_metrics = physics_loss_metric(pred.detach(), args.pde, dict(metric_meta))
    inference_optimization_time = float(first_batch.metadata.get("inference_optimization_time", 0.0) or 0.0)
    row = {
        "pde": args.pde,
        "task": args.task,
        "baseline": args.baseline,
        "seed": args.seed,
        "train_size": args.train_size,
        "num_sensors": args.num_sensors if use_sensors else 0,
        "sensor_mode": args.sensor_mode if use_sensors else "none",
        "noise_level": args.noise_level,
        "relative_l2_input_or_coeff": float("nan"),
        "relative_l2_solution": float(rel.detach().cpu()),
        "mse": float(mse(pred.detach(), first_batch.target_fields).detach().cpu()),
        "mae": float(mae(pred.detach(), first_batch.target_fields).detach().cpu()),
        "obs_mse": float(obs_mse(pred.detach(), first_batch.target_fields, first_batch.mask).detach().cpu()),
        "pde_residual": float(physics_metrics["interior"].detach().cpu()),
        "bc_residual": float(physics_metrics["bc"].detach().cpu()),
        "ic_residual": float(physics_metrics["ic"].detach().cpu()),
        "physics_loss": float(physics_metrics["total"].detach().cpu()),
        "residual_mode": str(physics_metrics["mode"]),
        "train_time": train_time,
        "inference_time": inference_time,
        "inference_optimization_time": inference_optimization_time,
        "num_params": int(model.parameter_count() if hasattr(model, "parameter_count") else num_parameters(model)),
        "config_path": args.config or "",
        "checkpoint_path": "",
        "commit_hash": _commit_hash(),
        "mask_id": first_batch.metadata.get("mask_id", ""),
        "dry_run": bool(args.dry_run),
        "input_shape": list(first_batch.input_fields.shape),
        "target_shape": list(first_batch.target_fields.shape),
        "pred_shape": list(pred.shape),
        "train_history": json.dumps(train_history),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = out_dir / f"{args.baseline}_{args.pde}_{args.task}_seed{args.seed}_config.json"
    config_snapshot.write_text(json.dumps({"args": vars(args), "method": method_cfg, "data_spec": _json_safe(data_spec)}, indent=2), encoding="utf-8")
    row["config_path"] = str(config_snapshot)
    if args.save_checkpoint:
        ckpt = out_dir / f"{args.baseline}_{args.pde}_{args.task}_seed{args.seed}.pt"
        model.save(ckpt)
        row["checkpoint_path"] = str(ckpt)
    append_result_jsonl(out_dir / "results.jsonl", row)
    append_result_csv(out_dir / "results.csv", row)

    print(json.dumps(row, indent=2, sort_keys=True))
    if tuple(pred.shape) != tuple(first_batch.target_fields.shape):
        raise RuntimeError(f"Prediction shape {tuple(pred.shape)} != target shape {tuple(first_batch.target_fields.shape)}")


def _batch_dataset(registry, batch):
    from baselines.common.data_adapter import PDEBatchDataset

    return PDEBatchDataset(batch)


def _commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return {"tensor_shape": list(obj.shape)}
    return obj


if __name__ == "__main__":
    main()
