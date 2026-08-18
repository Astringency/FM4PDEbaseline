from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
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
from baselines.common.sample_artifacts import EvaluationArtifactWriter
from baselines.methods.deeponet import DeepONetBaseline
from baselines.methods.fno import FNOBaseline
from baselines.methods.ifno import IFNOBaseline
from baselines.methods.official import OfficialImportError
from baselines.methods.official import (
    get_ifno_official_aligned_status,
    get_ifno_official_status,
    get_pc_bnn_net_class,
    get_pc_bnn_official_aligned_status,
    get_vivid_official_aligned_status,
    get_vivid_official_status,
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
from scripts.experiments.provenance import (
    DEFAULT_TASK_PROTOCOL_VERSION,
    MATRIX_SCHEMA_VERSION,
    reject_historical_experiment_path,
    requires_full_data_manifest,
    repository_revision,
    run_fingerprint,
    sha256_file,
    validate_full_data_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


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


def _run_stage(stage: str, state: str, **fields: Any) -> None:
    extra = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None and value != "")
    suffix = f" {extra}" if extra else ""
    print(f"[run stage] {stage} {state}{suffix}", file=sys.stderr, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("FM4PDE baseline runner")
    parser.add_argument("--baseline", required=True, choices=sorted(BASELINES))
    parser.add_argument("--pde", required=True)
    parser.add_argument("--task", default="forward")
    parser.add_argument("--data-root", default="/home/tat512/C01Python/PDEdata")
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiment-mode", choices=["smoke", "debug", "paper"], default="debug")
    parser.add_argument("--num-sensors", type=int, default=500)
    parser.add_argument(
        "--sensor-mode",
        choices=["random", "random_per_sample", "fixed", "grid", "time_varying", "time_slices_per_sample"],
        default="random_per_sample",
    )
    parser.add_argument("--sensor-budget-mode", choices=["per_time", "total"], default=None)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--val-size", type=int, default=0)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--train-shards", type=int, default=5)
    parser.add_argument("--test-split", choices=["test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sensor-seed", type=int, default=None, help="Independent base seed for sensor layouts; defaults to --seed.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/baselines")
    parser.add_argument("--experiment-kind", default="")
    parser.add_argument("--ablation-factor", default="")
    parser.add_argument("--task-group", default="")
    parser.add_argument("--run-id", default="", help="Stable external run identifier used in output filenames.")
    parser.add_argument("--run-name", default="", help="Human-readable external run name stored in metadata.")
    parser.add_argument("--run-fingerprint", default="", help="Content-addressed experiment fingerprint supplied by the matrix builder.")
    parser.add_argument("--matrix-schema-version", type=int, default=MATRIX_SCHEMA_VERSION)
    parser.add_argument("--config-content-sha256", default="", help="SHA-256 of the resolved experiment config contract.")
    parser.add_argument(
        "--experiment-config-sha256",
        default="",
        help="SHA-256 of the matrix-design YAML audited by the full data manifest.",
    )
    parser.add_argument(
        "--data-manifest-sha256",
        default="",
        help="SHA-256 of the passing full data_protocol_report.json bound to this experiment.",
    )
    parser.add_argument(
        "--data-manifest-path",
        default="",
        help="Auditable path to the bound full data_protocol_report.json.",
    )
    parser.add_argument(
        "--commit-hash",
        default="",
        help="Exact repository revision (HEAD plus dirty-tree digest) supplied by the experiment matrix.",
    )
    parser.add_argument("--summary-schema-version", type=int, default=2)
    parser.add_argument("--execution-mode", choices=["train", "eval_only"], default=None)
    parser.add_argument("--task-protocol-version", default="2")
    parser.add_argument("--sensor-protocol-version", default="2")
    parser.add_argument("--comparison-track", choices=["unified_adapted", "official_native"], default="unified_adapted")
    parser.add_argument("--source-train-run-id", default="", help="Training run identifier expected for eval-only checkpoints.")
    parser.add_argument(
        "--source-train-run-fingerprint",
        default="",
        help="Training fingerprint expected inside an eval-only checkpoint; distinct from the evaluation run fingerprint.",
    )
    parser.add_argument("--source-train-seed", type=int, default=None, help="Model-training seed expected in an eval-only checkpoint.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true", help="Use deterministic synthetic data for smoke/debug tests.")
    parser.add_argument("--allow-synthetic-fallback", action="store_true", help="Fall back to synthetic data when requested real files are missing.")
    parser.add_argument("--synthetic-resolution", type=int, default=32)
    parser.add_argument("--prefer-test", action="store_true", help="Compatibility/debug option. Never use for paper training.")
    parser.add_argument("--scalar-param-mode", choices=["metadata", "materialize", "global"], default="metadata")
    parser.add_argument("--data-loading-mode", choices=["eager", "lazy"], default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=None)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=None)
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--max-loaded-dataset-gb", type=float, default=None)
    parser.add_argument("--load-full-trajectory", action="store_true", help="Load full time trajectories when available instead of endpoint-only task tensors.")
    parser.add_argument("--physics-metric-mode", choices=["per_sample", "per_batch"], default=None)
    parser.add_argument("--strict-size", action="store_true", help="Fail if requested split size exceeds available samples.")
    parser.add_argument(
        "--save-checkpoint",
        dest="save_checkpoint",
        action="store_true",
        default=True,
        help="Save a reusable model checkpoint after training. Enabled by default.",
    )
    parser.add_argument(
        "--no-save-checkpoint",
        dest="save_checkpoint",
        action="store_false",
        help="Disable checkpoint saving for runs where only metrics are needed.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override per-instance optimization steps in the method config.")
    parser.add_argument("--refine-steps", type=int, default=None, help="Override VIVID refinement steps in the method config.")
    parser.add_argument("--particles", type=int, default=None, help="Override PC-BNN particle count in the method config.")
    parser.add_argument("--implementation-mode", default=None, help="Override method implementation_mode.")
    parser.add_argument("--official-backend", default=None, help="Override method official_backend.")
    parser.add_argument("--method-override", action="append", default=[], help="Override a method config key as key=value. Can be repeated.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; load checkpoint and run test evaluation only.")
    parser.add_argument("--checkpoint", default="", help="Path to checkpoint .pt file for --eval-only mode.")
    parser.add_argument("--checkpoint-sha256", default="", help="Expected SHA-256 of --checkpoint for a strict eval-only run.")
    parser.add_argument(
        "--save-sample-artifacts",
        dest="save_sample_artifacts",
        action="store_true",
        default=True,
        help="Save one reloadable .pt artifact for every evaluated test sample (default: enabled).",
    )
    parser.add_argument(
        "--no-save-sample-artifacts",
        dest="save_sample_artifacts",
        action="store_false",
        help="Disable per-sample artifacts in non-paper diagnostic runs.",
    )
    parser.add_argument(
        "--plot-sample-pdf",
        dest="plot_sample_pdf",
        action="store_true",
        default=True,
        help="Render evaluated samples into a multipage PDF (default: enabled).",
    )
    parser.add_argument(
        "--no-plot-sample-pdf",
        dest="plot_sample_pdf",
        action="store_false",
        help="Disable PDF rendering in non-paper diagnostic runs.",
    )
    return parser.parse_args(argv)


def load_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_baseline_checkpoint(
    path: str | Path,
    *,
    baseline: str | None = None,
    map_location: str | torch.device = "cpu",
    expected: dict[str, Any] | None = None,
    require_provenance: bool = False,
) -> torch.nn.Module:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    _validate_checkpoint_payload(payload, expected or {}, require_provenance=require_provenance)
    baseline_name = str(baseline or payload.get("baseline") or "")
    if not baseline_name:
        raise ValueError("Checkpoint does not record a baseline name; pass baseline='...' explicitly.")
    if baseline_name not in BASELINES:
        raise ValueError(f"Unknown baseline in checkpoint: {baseline_name!r}")
    model = BASELINES[baseline_name]().build(payload.get("config", {}), payload.get("data_spec", {}))
    model.load_payload(payload)
    return model.to(torch.device(map_location))


def _validate_checkpoint_payload(payload: dict[str, Any], expected: dict[str, Any], *, require_provenance: bool) -> None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        if require_provenance:
            raise ValueError("Checkpoint has no provenance contract and cannot be used for a strict eval-only run")
        provenance = {}
    data_spec = payload.get("data_spec") if isinstance(payload.get("data_spec"), dict) else {}
    actual = {
        "baseline": payload.get("baseline"),
        "pde": provenance.get("pde", data_spec.get("pde")),
        "task": provenance.get("task", data_spec.get("task")),
        "run_fingerprint": provenance.get("run_fingerprint"),
        "config_content_sha256": provenance.get("config_content_sha256"),
        "experiment_config_sha256": provenance.get("experiment_config_sha256"),
        "data_manifest_sha256": provenance.get("data_manifest_sha256"),
        "data_manifest_path": provenance.get("data_manifest_path"),
        "task_protocol_version": provenance.get("task_protocol_version"),
        "sensor_protocol_version": provenance.get("sensor_protocol_version"),
        "commit_hash": provenance.get("commit_hash"),
        "source_train_run_id": provenance.get("run_id"),
        "seed": provenance.get("seed"),
        "comparison_track": provenance.get("comparison_track"),
        "sensor_mode": provenance.get("sensor_mode"),
        "num_sensors": provenance.get("num_sensors"),
        "requested_train_size": provenance.get("requested_train_size"),
        "val_size": provenance.get("val_size"),
        "train_shards": provenance.get("train_shards"),
        "batch_size": provenance.get("batch_size"),
        "epochs": provenance.get("epochs"),
    }
    mismatches = []
    for key, wanted in expected.items():
        if wanted in {None, ""}:
            continue
        got = actual.get(key)
        if str(got) != str(wanted):
            mismatches.append(f"{key}: checkpoint={got!r}, expected={wanted!r}")
    if mismatches:
        raise ValueError("Checkpoint provenance mismatch: " + "; ".join(mismatches))


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
    reject_historical_experiment_path(args.output_dir, field="--output-dir")
    _validate_mode(args)
    if args.experiment_mode == "paper" and requires_full_data_manifest(
        args.task_protocol_version
    ):
        args.strict_size = True
    actual_revision = repository_revision(ROOT)
    if args.commit_hash and args.commit_hash != actual_revision:
        raise ValueError(
            "--commit-hash does not match the current repository revision: "
            f"supplied={args.commit_hash!r}, actual={actual_revision!r}; regenerate the matrix from this code tree"
        )
    args.commit_hash = actual_revision
    if args.sensor_seed is None:
        args.sensor_seed = int(args.seed)
    if args.source_train_seed is None:
        args.source_train_seed = int(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_yaml(args.config)
    _record_requested_runtime_args(args, cfg)
    args.data_loading_mode = _resolve_data_loading_mode(args)
    args.effective_data_loading_mode = "eager"
    _resolve_dataloader_args(args, cfg)
    args.physics_metric_mode = _resolve_physics_metric_mode(args, cfg)
    args.sensor_budget_mode = _resolve_sensor_budget_mode(args, cfg)
    method_cfg = build_method_config(cfg, args)
    args.epochs_effective = int(method_cfg.get("epochs", 0) or 0)
    derived_execution_mode = "eval_only" if args.eval_only else "train"
    if args.execution_mode is not None and args.execution_mode != derived_execution_mode:
        raise ValueError(
            f"--execution-mode={args.execution_mode} conflicts with "
            f"{'--eval-only' if args.eval_only else 'a training invocation'}"
        )
    args.execution_mode = derived_execution_mode
    observed_config_sha256 = _config_content_sha256(args.config, cfg)
    if (
        args.config_content_sha256
        and args.config_content_sha256 != observed_config_sha256
    ):
        raise ValueError(
            "--config-content-sha256 does not match the loaded configuration: "
            f"supplied={args.config_content_sha256}, observed={observed_config_sha256}"
        )
    args.config_content_sha256 = observed_config_sha256
    _validate_data_manifest_binding(args)
    _validate_and_bind_run_fingerprint(args, method_cfg)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = _run_file_prefix(args)
    method_cfg["run_output_dir"] = str(out_dir)
    method_cfg["run_prefix"] = run_prefix
    method_cfg["train_history_json_path"] = str(out_dir / f"{run_prefix}_train_history.json")
    method_cfg["train_history_jsonl_path"] = str(out_dir / f"{run_prefix}_train_history.jsonl")
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
        and args.comparison_track == "official_native"
        and not bool(getattr(capability, "official_native_eligible", False))
    ):
        _skip_paper_run(
            args,
            capability,
            method_cfg,
            "No end-to-end official-native verification recipe is registered for this combination",
            adapter_status="official_native_not_verified",
        )
        return
    if (
        args.experiment_mode == "paper"
        and args.comparison_track == "official_native"
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

    _run_stage("build_dataset", "start", baseline=args.baseline, pde=args.pde, task=args.task)
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
    print_dataset_memory_summary("train", train_dataset_for_fit)
    print_dataset_memory_summary("val", val_dataset)
    print_dataset_memory_summary("test", test_dataset)
    memory_fields = dataset_memory_result_fields(train_dataset_for_fit, val_dataset, test_dataset)
    _check_loaded_dataset_memory_limit(args, memory_fields)

    train_loader = build_pde_dataloader(train_dataset_for_fit, args, shuffle=not is_per_instance)
    val_loader = build_pde_dataloader(val_dataset, args, shuffle=False) if val_dataset is not None else None
    spec_loader = build_pde_dataloader(spec_dataset, args, shuffle=False, batch_size=min(args.batch_size, len(spec_dataset)))
    test_loader = build_pde_dataloader(test_dataset, args, shuffle=False)
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
    _run_stage(
        "build_dataset",
        "done",
        train_size=len(train_dataset_for_fit),
        val_size=0 if val_dataset is None else len(val_dataset),
        test_size=len(test_dataset),
    )

    _run_stage("build_model", "start", baseline=args.baseline)
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
    _run_stage("build_model", "done", baseline=args.baseline, backend=backend_info.get("backend_used", ""))
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

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval-only requires --checkpoint")
        if args.experiment_mode == "paper" and (not args.source_train_run_id or not args.source_train_run_fingerprint):
            raise ValueError(
                "paper eval-only requires --source-train-run-id and --source-train-run-fingerprint so the checkpoint "
                "can be tied to its training run"
            )
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model = load_baseline_checkpoint(
            ckpt_path,
            map_location=args.device,
            expected={
                "baseline": args.baseline,
                "pde": args.pde,
                "task": args.task,
                "run_fingerprint": args.source_train_run_fingerprint,
                "config_content_sha256": args.config_content_sha256,
                "experiment_config_sha256": args.experiment_config_sha256,
                "data_manifest_sha256": args.data_manifest_sha256,
                "data_manifest_path": args.data_manifest_path,
                "task_protocol_version": args.task_protocol_version,
                "sensor_protocol_version": args.sensor_protocol_version,
                "commit_hash": args.commit_hash,
                "source_train_run_id": args.source_train_run_id,
                "seed": args.source_train_seed,
                "comparison_track": args.comparison_track,
                "sensor_mode": args.sensor_mode,
                "num_sensors": args.num_sensors if use_sensors else 0,
                "requested_train_size": split_info["train_requested_size"],
                "val_size": split_info["val_requested_size"],
                "train_shards": args.train_shards,
                "batch_size": args.batch_size,
                "epochs": method_cfg.get("epochs", 0),
            },
            require_provenance=True,
        )
        backend_info = _backend_info(model, method_cfg)
        capability_info["paper_table_eligible"] = _paper_table_eligibility(capability, backend_info)
        train_time = 0.0
        train_history = {}
        normalization_stats_path = ""
        normalization_fields = {}
        checkpoint_provenance = dict(getattr(model, "provenance", {}) or {})
        config_snapshot = str(checkpoint_provenance.get("config_path", ""))
        config_hash = str(checkpoint_provenance.get("config_hash", ""))
        checkpoint_path = str(ckpt_path)
        train_history_path = out_dir / f"{run_prefix}_train_history.json"
    else:
        _run_stage("fit", "start", baseline=args.baseline, pde=args.pde, task=args.task)
        _synchronize_device(args.device)
        train_start = time.perf_counter()
        train_history = {}
        if not is_per_instance:
            train_history = model.fit(train_loader, val_loader)
        else:
            train_history = model.fit(train_loader, val_loader)
        _synchronize_device(args.device)
        train_time = time.perf_counter() - train_start
        split_info["fit_setup_time"] = float(train_time)
        if is_per_instance:
            # Per-instance optimization happens inside predict() on each test
            # sample. The microsecond setup call is not amortized training.
            train_time = 0.0
        _run_stage("fit", "done", baseline=args.baseline, elapsed_sec=f"{train_time:.3f}")

        normalization_stats_path = _write_normalization_stats(out_dir, run_prefix, model)
        normalization_fields = _normalization_fields(model, normalization_stats_path)
        config_snapshot = _write_config_snapshot(out_dir, args, cfg, method_cfg, data_spec, backend_info, method_budget_fields, capability_info, normalization_fields, memory_fields)
        config_hash = _file_sha1(config_snapshot)
        model.provenance = {
            "run_id": args.run_id,
            "run_fingerprint": args.run_fingerprint,
            "baseline": args.baseline,
            "pde": args.pde,
            "task": args.task,
            "seed": int(args.seed),
            "config_content_sha256": args.config_content_sha256,
            "experiment_config_sha256": args.experiment_config_sha256,
            "data_manifest_sha256": args.data_manifest_sha256,
            "data_manifest_path": args.data_manifest_path,
            "config_hash": config_hash,
            "config_path": str(config_snapshot),
            "task_protocol_version": str(args.task_protocol_version),
            "sensor_protocol_version": str(args.sensor_protocol_version),
            "comparison_track": str(args.comparison_track),
            "execution_mode": "train",
            "commit_hash": str(getattr(args, "commit_hash", "")),
            "requested_train_size": int(split_info["train_requested_size"]),
            "effective_train_size": 0 if is_per_instance else int(split_info["effective_train_size"]),
            "val_size": int(split_info["val_requested_size"]),
            "train_shards": int(args.train_shards),
            "batch_size": int(args.batch_size),
            "epochs": int(method_cfg.get("epochs", 0)),
            "sensor_mode": str(args.sensor_mode),
            "sensor_seed": int(args.sensor_seed),
            "num_sensors": int(args.num_sensors if use_sensors else 0),
        }
        train_history_path = out_dir / f"{run_prefix}_train_history.json"
        train_history_payload = _json_safe(train_history)
        if isinstance(train_history_payload, dict):
            train_history_payload.setdefault("completed_epochs", len(train_history_payload.get("train_loss", [])))
        train_history_path.write_text(json.dumps(train_history_payload, indent=2), encoding="utf-8")
        checkpoint_path = ""
        if args.save_checkpoint:
            ckpt = out_dir / f"{run_prefix}.pt"
            model.save(ckpt)
            checkpoint_path = str(ckpt)

    _run_stage("eval", "start", baseline=args.baseline, pde=args.pde, task=args.task)
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
        normalization_fields=normalization_fields,
        memory_fields=memory_fields,
        config_hash=config_hash,
    )
    _run_stage("eval", "done", baseline=args.baseline)
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
        normalization_fields,
        memory_fields,
        config_hash,
    )
    summary.update(
        {
            "train_time": train_time,
            "num_params": int(model.parameter_count() if hasattr(model, "parameter_count") else num_parameters(model)),
            "num_params_storage": int(
                model.parameter_storage_count()
                if hasattr(model, "parameter_storage_count")
                else sum(p.numel() for p in model.parameters() if p.requires_grad)
            ),
            "parameter_count_convention": "real_scalar_dof_complex_counts_as_two",
            "config_path": str(config_snapshot),
            "config_hash": config_hash,
            "checkpoint_path": checkpoint_path,
            "train_history_path": str(train_history_path),
            "commit_hash": args.commit_hash,
            "dry_run": bool(args.dry_run),
            "synthetic_data": bool(args.synthetic_data),
            "experiment_mode": args.experiment_mode,
            "run_id": args.run_id,
            "run_name": args.run_name,
            "effective_data_loading_mode": getattr(args, "effective_data_loading_mode", "eager"),
            **_dataloader_fields(args),
            **memory_fields,
            "epoch_log_path": str(out_dir / f"{run_prefix}_train_history.jsonl"),
            "train_history_jsonl_path": str(out_dir / f"{run_prefix}_train_history.jsonl"),
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
    _run_stage("write_summary", "done", output_dir=out_dir)

    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


def _validate_mode(args: argparse.Namespace) -> None:
    if args.experiment_mode == "paper" and (args.dry_run or args.synthetic_data or args.allow_synthetic_fallback or args.prefer_test):
        raise ValueError("paper mode cannot use --dry-run, --synthetic-data, --allow-synthetic-fallback, or --prefer-test")
    if args.dry_run and args.experiment_mode == "paper":
        raise ValueError("--dry-run is restricted to smoke/debug modes")
    if args.synthetic_data and args.experiment_mode == "paper":
        raise ValueError("--synthetic-data is restricted to smoke/debug modes")
    if args.experiment_mode == "paper" and not bool(args.save_sample_artifacts):
        raise ValueError("paper mode requires --save-sample-artifacts")
    if args.experiment_mode == "paper" and not bool(args.plot_sample_pdf):
        raise ValueError("paper mode requires --plot-sample-pdf")


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
    mode = str(args.data_loading_mode or "eager")
    if mode not in {"eager", "lazy"}:
        raise ValueError(f"data_loading_mode must be eager or lazy, got {mode!r}")
    if mode == "lazy":
        message = (
            "data_loading_mode='lazy' is disabled because per-sample file reads make paper baselines I/O-bound. "
            "Regenerate the matrix/config with data_loading_mode='eager'."
        )
        if args.experiment_mode == "paper":
            raise ValueError(message)
        warnings.warn(message + " Forcing eager loading.", RuntimeWarning, stacklevel=2)
    return "eager"


def _record_requested_runtime_args(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    """Preserve matrix-level requests before resolving device/runtime effects."""
    args.data_loading_mode_requested = str(
        args.data_loading_mode
        if args.data_loading_mode is not None
        else cfg.get("data_loading_mode", "eager")
    )
    args.num_workers_requested = int(
        args.num_workers if args.num_workers is not None else cfg.get("num_workers", 0)
    )
    args.pin_memory_requested = bool(
        args.pin_memory if args.pin_memory is not None else cfg.get("pin_memory", True)
    )
    args.persistent_workers_requested = bool(
        args.persistent_workers
        if args.persistent_workers is not None
        else cfg.get("persistent_workers", False)
    )
    requested_prefetch = (
        args.prefetch_factor
        if args.prefetch_factor is not None
        else cfg.get("prefetch_factor", None)
    )
    args.prefetch_factor_requested = int(requested_prefetch or 0)


def _resolve_dataloader_args(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    args.num_workers = int(args.num_workers if args.num_workers is not None else cfg.get("num_workers", 0))
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.pin_memory is None:
        args.pin_memory = bool(cfg.get("pin_memory", True))
    if args.persistent_workers is None:
        args.persistent_workers = bool(cfg.get("persistent_workers", False))
    if args.prefetch_factor is None:
        value = cfg.get("prefetch_factor", None)
        args.prefetch_factor = None if value is None else int(value)
    elif args.prefetch_factor is not None:
        args.prefetch_factor = int(args.prefetch_factor)
    if args.prefetch_factor is not None and args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be > 0 when set")
    if args.num_workers == 0:
        args.persistent_workers = False
    if args.max_loaded_dataset_gb is None and cfg.get("max_loaded_dataset_gb", None) is not None:
        args.max_loaded_dataset_gb = float(cfg["max_loaded_dataset_gb"])
    if args.max_loaded_dataset_gb is not None and float(args.max_loaded_dataset_gb) < 0:
        raise ValueError("--max-loaded-dataset-gb must be >= 0 when set")


def build_pde_dataloader(
    dataset,
    args: argparse.Namespace,
    shuffle: bool,
    batch_size: int | None = None,
) -> DataLoader:
    num_workers = int(getattr(args, "num_workers", 0) or 0)
    device = str(getattr(args, "device", "cpu"))
    pin_requested = bool(getattr(args, "pin_memory", False))
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size if batch_size is not None else args.batch_size),
        "shuffle": bool(shuffle),
        "drop_last": False,
        "collate_fn": pde_collate,
        "num_workers": num_workers,
        "pin_memory": bool(pin_requested and device.startswith("cuda")),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(getattr(args, "persistent_workers", False))
        prefetch = getattr(args, "prefetch_factor", None)
        if prefetch is not None:
            kwargs["prefetch_factor"] = int(prefetch)
    return DataLoader(dataset, **kwargs)


def _dataloader_fields(args: argparse.Namespace) -> dict[str, Any]:
    num_workers = int(getattr(args, "num_workers", 0) or 0)
    pin_memory = bool(getattr(args, "pin_memory", False) and str(getattr(args, "device", "cpu")).startswith("cuda"))
    return {
        "dataloader_num_workers": num_workers,
        "dataloader_pin_memory": pin_memory,
        "dataloader_persistent_workers": bool(getattr(args, "persistent_workers", False) and num_workers > 0),
        "dataloader_prefetch_factor": int(args.prefetch_factor) if num_workers > 0 and getattr(args, "prefetch_factor", None) is not None else 0,
    }


def _requested_design_fields(args: argparse.Namespace) -> dict[str, Any]:
    """Requested values bound by the matrix fingerprint (not device effects)."""
    return {
        "test_requested_size": int(getattr(args, "test_size", 0) or 0),
        "batch_size": int(getattr(args, "batch_size", 0) or 0),
        "epochs": int(
            getattr(args, "epochs_effective", getattr(args, "epochs", 0) or 0)
        ),
        "device": str(getattr(args, "device", "cpu")),
        "sensor_budget_mode_requested": (
            str(getattr(args, "sensor_budget_mode", "per_time"))
            if str(getattr(args, "task", "")).startswith("sparse")
            else "none"
        ),
        "data_loading_mode_requested": str(
            getattr(
                args,
                "data_loading_mode_requested",
                getattr(args, "data_loading_mode", "eager"),
            )
        ),
        "num_workers": int(
            getattr(args, "num_workers_requested", getattr(args, "num_workers", 0) or 0)
        ),
        "pin_memory_requested": bool(
            getattr(args, "pin_memory_requested", getattr(args, "pin_memory", False))
        ),
        "persistent_workers_requested": bool(
            getattr(
                args,
                "persistent_workers_requested",
                getattr(args, "persistent_workers", False),
            )
        ),
        "prefetch_factor_requested": int(
            getattr(
                args,
                "prefetch_factor_requested",
                getattr(args, "prefetch_factor", 0) or 0,
            )
        ),
    }


def tensor_nbytes(tensor: Any) -> int:
    if not isinstance(tensor, torch.Tensor):
        return 0
    return int(tensor.numel() * tensor.element_size())


def _unique_tensor_storage_nbytes(
    value: Any,
    *,
    seen_storages: set[tuple[str, int, int]],
    seen_containers: set[int],
) -> int:
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        storage_nbytes = int(storage.nbytes())
        key = (str(value.device), int(storage.data_ptr()), storage_nbytes)
        if key in seen_storages:
            return 0
        seen_storages.add(key)
        return storage_nbytes
    if isinstance(value, dict):
        container_id = id(value)
        if container_id in seen_containers:
            return 0
        seen_containers.add(container_id)
        return sum(
            _unique_tensor_storage_nbytes(
                item,
                seen_storages=seen_storages,
                seen_containers=seen_containers,
            )
            for item in value.values()
        )
    if isinstance(value, (list, tuple, set)):
        container_id = id(value)
        if container_id in seen_containers:
            return 0
        seen_containers.add(container_id)
        return sum(
            _unique_tensor_storage_nbytes(
                item,
                seen_storages=seen_storages,
                seen_containers=seen_containers,
            )
            for item in value
        )
    return 0


def dataset_tensor_memory_summary(dataset: PDEBatchDataset | None) -> dict[str, Any]:
    if dataset is None:
        return {
            "samples": 0,
            "full_tensor_gb": 0.0,
            "input_fields_gb": 0.0,
            "target_fields_gb": 0.0,
            "auxiliary_tensor_memory_gb": 0.0,
            "tensor_memory_gb": 0.0,
            "loaded_full_trajectory": False,
            "data_loading_mode": "",
        }
    batch = dataset.batch
    full_gb = tensor_nbytes(batch.full_tensor) / 1e9
    input_gb = tensor_nbytes(batch.input_fields) / 1e9
    target_gb = tensor_nbytes(batch.target_fields) / 1e9
    primary_storage_bytes = _unique_tensor_storage_nbytes(
        (batch.full_tensor, batch.input_fields, batch.target_fields),
        seen_storages=set(),
        seen_containers=set(),
    )
    total_storage_bytes = _unique_tensor_storage_nbytes(
        vars(batch),
        seen_storages=set(),
        seen_containers=set(),
    )
    auxiliary_gb = max(total_storage_bytes - primary_storage_bytes, 0) / 1e9
    return {
        "samples": len(dataset),
        "full_tensor_gb": full_gb,
        "input_fields_gb": input_gb,
        "target_fields_gb": target_gb,
        "auxiliary_tensor_memory_gb": auxiliary_gb,
        "tensor_memory_gb": total_storage_bytes / 1e9,
        "loaded_full_trajectory": bool(getattr(dataset, "loaded_full_trajectory", _batch_loaded_full_trajectory(batch))),
        "data_loading_mode": str(getattr(dataset, "data_loading_mode", batch.metadata.get("data_loading_mode", ""))),
    }


def print_dataset_memory_summary(name: str, dataset: PDEBatchDataset | None) -> None:
    summary = dataset_tensor_memory_summary(dataset)
    print(
        f"[dataset memory] split={name} samples={summary['samples']} "
        f"full_tensor={summary['full_tensor_gb']:.6g}GB "
        f"input_fields={summary['input_fields_gb']:.6g}GB "
        f"target_fields={summary['target_fields_gb']:.6g}GB "
        f"auxiliary_tensors={summary['auxiliary_tensor_memory_gb']:.6g}GB "
        f"loaded_full_trajectory={summary['loaded_full_trajectory']} "
        f"data_loading_mode={summary['data_loading_mode']}",
        file=sys.stderr,
        flush=True,
    )


def dataset_memory_result_fields(
    train_dataset: PDEBatchDataset,
    val_dataset: PDEBatchDataset | None,
    test_dataset: PDEBatchDataset,
) -> dict[str, float]:
    train = dataset_tensor_memory_summary(train_dataset)
    val = dataset_tensor_memory_summary(val_dataset)
    test = dataset_tensor_memory_summary(test_dataset)
    return {
        "train_tensor_memory_gb": float(train["tensor_memory_gb"]),
        "val_tensor_memory_gb": float(val["tensor_memory_gb"]),
        "test_tensor_memory_gb": float(test["tensor_memory_gb"]),
        "train_full_tensor_memory_gb": float(train["full_tensor_gb"]),
        "train_input_tensor_memory_gb": float(train["input_fields_gb"]),
        "train_target_tensor_memory_gb": float(train["target_fields_gb"]),
        "train_auxiliary_tensor_memory_gb": float(train["auxiliary_tensor_memory_gb"]),
    }


def _check_loaded_dataset_memory_limit(args: argparse.Namespace, fields: dict[str, float]) -> None:
    total_gb = float(fields["train_tensor_memory_gb"] + fields["val_tensor_memory_gb"] + fields["test_tensor_memory_gb"])
    limit = getattr(args, "max_loaded_dataset_gb", None)
    if limit is not None and total_gb > float(limit):
        raise RuntimeError(f"Loaded dataset tensor memory {total_gb:.6g}GB exceeds max_loaded_dataset_gb={float(limit):.6g}GB")
    if limit is None:
        print(f"[dataset memory] total_tensor_memory={total_gb:.6g}GB max_loaded_dataset_gb=None warning=limit_disabled", file=sys.stderr, flush=True)


def _resolve_sensor_budget_mode(args: argparse.Namespace, cfg: dict[str, Any]) -> str:
    mode = args.sensor_budget_mode or cfg.get("sensor_budget_mode", "per_time")
    mode = str(mode or "per_time")
    if mode not in {"per_time", "total"}:
        raise ValueError(f"sensor_budget_mode must be per_time or total, got {mode!r}")
    return mode


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
        "effective_train_size": int(train_size - val_size) if val_size > 0 else int(train_size),
        "val_requested_size": int(val_size),
        "val_split_source": "none",
        "val_from_train_offset": None,
    }
    if val_size <= 0:
        return None, split_info
    if train_size <= val_size:
        raise ValueError(
            f"Cannot reserve val_size={val_size} inside train_size={train_size}; increase TRAIN_SIZE or reduce VAL_SIZE."
        )
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
            sensor_budget_mode=args.sensor_budget_mode,
            noise_level=args.noise_level,
            seed=args.sensor_seed,
            experiment_mode=args.experiment_mode,
            build_voronoi_grid=args.baseline in {"recfno", "voronoicnn"},
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
        sensor_budget_mode=args.sensor_budget_mode,
        noise_level=args.noise_level,
        seed=args.sensor_seed,
        prefer_test=args.prefer_test and split == "test",
        synthetic_if_missing=args.allow_synthetic_fallback,
        synthetic_resolution=args.synthetic_resolution,
        synthetic_seed=synthetic_seed,
        scalar_param_mode=args.scalar_param_mode,
        data_loading_mode=args.data_loading_mode,
        load_full_trajectory=_split_load_full_trajectory(args, split),
        experiment_mode=args.experiment_mode,
        build_voronoi_grid=args.baseline in {"recfno", "voronoicnn"},
        strict_size=(
            args.strict_size
            if strict_size_override is None
            else bool(strict_size_override)
        ),
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
    normalization_fields: dict[str, Any],
    memory_fields: dict[str, float],
    config_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    sensor_seed = int(getattr(args, "sensor_seed", args.seed))
    source_train_seed = int(getattr(args, "source_train_seed", args.seed))
    rows: list[dict[str, Any]] = []
    inference_time_total = 0.0
    inference_optimization_time_total = 0.0
    raw_path = out_dir / "results_raw.jsonl"
    grad_enabled = args.baseline in {"pinn_sparse", "pc_bnn", "pde_opt", "var4d", "vivid"}
    per_instance = bool(split_info.get("per_instance_baseline", False))
    reported_train_size = 0 if per_instance else int(split_info["effective_train_size"])
    checkpoint_sha256 = _file_sha256(Path(checkpoint_path)) if checkpoint_path else ""
    checkpoint_provenance = dict(getattr(model, "provenance", {}) or {})
    artifact_writer: EvaluationArtifactWriter | None = None
    if bool(getattr(args, "save_sample_artifacts", True)):
        artifact_name = f"{_run_file_prefix(args)}_samples" if str(getattr(args, "run_id", "")) else "samples"
        artifact_writer = EvaluationArtifactWriter(
            out_dir / artifact_name,
            run_metadata={
                "run_id": str(getattr(args, "run_id", "")),
                "run_fingerprint": str(getattr(args, "run_fingerprint", "")),
                "baseline": str(args.baseline),
                "pde": str(args.pde),
                "task": str(args.task),
                "seed": int(args.seed),
                "sensor_seed": sensor_seed,
            },
            pdf_path=(out_dir / f"{artifact_name}.pdf") if bool(getattr(args, "plot_sample_pdf", True)) else None,
        )
        artifact_writer.__enter__()
    provenance_fields = {
        "matrix_schema_version": int(getattr(args, "matrix_schema_version", MATRIX_SCHEMA_VERSION)),
        "summary_schema_version": int(getattr(args, "summary_schema_version", 2)),
        "status": "success",
        "execution_mode": str(getattr(args, "execution_mode", "eval_only" if getattr(args, "eval_only", False) else "train")),
        "eval_only": bool(getattr(args, "eval_only", False)),
        "run_fingerprint": str(getattr(args, "run_fingerprint", "")),
        "config_content_sha256": str(getattr(args, "config_content_sha256", "")),
        "experiment_config_sha256": str(
            getattr(args, "experiment_config_sha256", "")
        ),
        "data_manifest_sha256": str(getattr(args, "data_manifest_sha256", "")),
        "data_manifest_path": str(getattr(args, "data_manifest_path", "")),
        "task_protocol_version": str(getattr(args, "task_protocol_version", "2")),
        "sensor_protocol_version": str(getattr(args, "sensor_protocol_version", "2")),
        "comparison_track": str(getattr(args, "comparison_track", "unified_adapted")),
        "source_train_run_id": str(
            checkpoint_provenance.get("run_id", args.run_id if not getattr(args, "eval_only", False) else "")
        ),
        "source_train_run_fingerprint": str(checkpoint_provenance.get("run_fingerprint", "")),
        "source_train_seed": int(
            checkpoint_provenance.get(
                "seed",
                source_train_seed if getattr(args, "eval_only", False) else args.seed,
            )
        ),
        "checkpoint_sha256": checkpoint_sha256,
    }
    for batch_index, batch in enumerate(loader):
        _synchronize_device(args.device)
        start = time.perf_counter()
        eval_batch = _to_device_batch_for_eval(_make_inference_batch(batch), args.device)
        with torch.set_grad_enabled(grad_enabled):
            pred = model.predict_physical(eval_batch) if hasattr(model, "predict_physical") else model.predict(eval_batch)
        _synchronize_device(args.device)
        elapsed = time.perf_counter() - start
        _copy_eval_metadata(batch, eval_batch)
        pred_cpu = pred.detach().cpu()
        target = batch.target_fields.detach().cpu()
        pred_cpu, target = _align(pred_cpu, target)
        inf_opt = float(batch.metadata.get("inference_optimization_time", 0.0) or 0.0)
        batch_n = int(target.shape[0])
        if args.task in {"sparse_solution", "sparse_reconstruction"} and batch.metadata.get("joint_reconstruction"):
            rel_values, input_or_coeff_values = _joint_reconstruction_relative_l2_values(
                pred_cpu, target, batch.metadata
            )
        else:
            rel_values = (
                [float("nan")] * int(target.shape[0])
                if args.task in {"inverse", "sparse_inverse"}
                else _relative_l2_values(pred_cpu, target)
            )
            input_or_coeff_values = _relative_l2_input_or_coeff_values(args.task, pred_cpu, target)
        mse_values = _mse_values(pred_cpu, target)
        mae_values = _mae_values(pred_cpu, target)
        metric_payload = _batch_metric_payload(pred_cpu, target, batch, args)
        row = {
            "run_id": args.run_id,
            "run_name": args.run_name,
            **provenance_fields,
            **_experiment_fields(args),
            **method_budget_fields,
            "pde": args.pde,
            "task": args.task,
            "baseline": args.baseline,
            "seed": args.seed,
            "sensor_seed": sensor_seed,
            "batch_index": batch_index,
            "sample_count": batch_n,
            "split": "test",
            "train_size": reported_train_size,
            "train_requested_size": int(split_info["train_requested_size"]),
            "effective_train_size": reported_train_size,
            "train_size_loaded_for_fit": int(split_info["train_size_loaded_for_fit"]),
            "train_size_loaded_for_spec": int(split_info["train_size_loaded_for_spec"]),
            "val_size": len(val_dataset) if val_dataset is not None else 0,
            "val_requested_size": int(split_info["val_requested_size"]),
            "val_split_source": split_info["val_split_source"],
            "val_from_train_offset": split_info["val_from_train_offset"],
            "test_size": len(test_dataset),
            "train_shards": args.train_shards,
            **_requested_design_fields(args),
            "data_loading_mode": args.data_loading_mode,
            "effective_data_loading_mode": getattr(args, "effective_data_loading_mode", "eager"),
            **_dataloader_fields(args),
            **memory_fields,
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
            "sensor_budget_mode": str(batch.metadata.get("sensor_budget_mode", args.sensor_budget_mode if args.task.startswith("sparse") else "none")),
            "num_sensors_per_time": json.dumps(batch.metadata.get("num_sensors_per_time", [] if args.task.startswith("sparse") else [])),
            "num_observations_total": int(batch.metadata.get("num_observations_total", 0) or 0),
            "requested_sensor_mode": args.sensor_mode if args.task.startswith("sparse") else "none",
            "effective_sensor_mode": str(batch.metadata.get("effective_sensor_mode", args.sensor_mode if args.task.startswith("sparse") else "none")),
            "sensor_mode": str(batch.metadata.get("effective_sensor_mode", args.sensor_mode if args.task.startswith("sparse") else "none")),
            "time_varying_sensor_valid": bool(batch.metadata.get("time_varying_sensor_valid", True)),
            "noise_level": args.noise_level,
            "mask_id": batch.metadata.get("mask_id", ""),
            "mask_ids_unique_count": len(set(batch.metadata.get("mask_ids", []) or [])),
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
            "learned_state_injection_mode": str(batch.metadata.get("learned_state_injection_mode", "")),
            "learned_state_shape": json.dumps(list(batch.metadata.get("learned_state_shape", ())) if batch.metadata.get("learned_state_shape") else []),
            "optimized_state_shape": json.dumps(list(batch.metadata.get("optimized_state_shape", ())) if batch.metadata.get("optimized_state_shape") else []),
            "posterior_particles": int(batch.metadata.get("posterior_particles", 0) or 0),
            "train_time": train_time,
            "fit_setup_time": float(split_info.get("fit_setup_time", 0.0) or 0.0),
            "test_time_optimization": per_instance,
            "amortized_training": not per_instance,
            "inference_time": elapsed,
            "inference_optimization_time": inf_opt,
            "optimization_steps_completed": int(batch.metadata.get("optimization_steps_completed", 0) or 0),
            "optimization_early_stopped": bool(batch.metadata.get("optimization_early_stopped", False)),
            "num_params": int(model.parameter_count() if hasattr(model, "parameter_count") else num_parameters(model)),
            "num_params_storage": int(
                model.parameter_storage_count()
                if hasattr(model, "parameter_storage_count")
                else sum(p.numel() for p in model.parameters() if p.requires_grad)
            ),
            "parameter_count_convention": "real_scalar_dof_complex_counts_as_two",
            "config_path": config_snapshot,
            "config_hash": config_hash,
            "checkpoint_path": checkpoint_path,
            "commit_hash": str(getattr(args, "commit_hash", "")),
            "dry_run": bool(args.dry_run),
            "synthetic_data": bool(args.synthetic_data),
            "experiment_mode": args.experiment_mode,
            "global_sample_ids": json.dumps(batch.global_sample_ids),
            "input_shape": json.dumps(list(batch.input_fields.shape)),
            "target_shape": json.dumps(list(batch.target_fields.shape)),
            "pred_shape": json.dumps(list(pred_cpu.shape)),
            "train_history": json.dumps(_json_safe(train_history)),
            **_train_history_fields(train_history),
            **normalization_fields,
        }
        for key in ("obs_mse", "obs_mse_clean", "obs_mse_noisy", "pde_residual", "bc_residual", "ic_residual", "physics_loss"):
            values_key = f"{key}_values"
            if values_key in metric_payload:
                row[values_key] = json.dumps(metric_payload[values_key])
        if tuple(pred_cpu.shape) != tuple(target.shape):
            raise RuntimeError(f"Prediction shape {tuple(pred_cpu.shape)} != target shape {tuple(target.shape)}")
        if artifact_writer is not None:
            sample_metrics: list[dict[str, Any]] = []
            for item in range(batch_n):
                values = {
                    "relative_l2_solution": rel_values[item],
                    "relative_l2_input_or_coeff": input_or_coeff_values[item],
                    "mse": mse_values[item],
                    "mae": mae_values[item],
                }
                for key in (
                    "obs_mse",
                    "obs_mse_clean",
                    "obs_mse_noisy",
                    "pde_residual",
                    "bc_residual",
                    "ic_residual",
                    "physics_loss",
                ):
                    per_sample_values = metric_payload.get(f"{key}_values")
                    values[key] = (
                        per_sample_values[item]
                        if isinstance(per_sample_values, list) and item < len(per_sample_values)
                        else metric_payload.get(key, float("nan"))
                    )
                sample_metrics.append(values)
            artifact_writer.write_batch(
                batch,
                pred_cpu,
                predictive_std=(
                    eval_batch.metadata.get("predictive_std")
                    if isinstance(eval_batch.metadata.get("predictive_std"), torch.Tensor)
                    else None
                ),
                posterior_samples=(
                    eval_batch.metadata.get("posterior_samples")
                    if isinstance(eval_batch.metadata.get("posterior_samples"), torch.Tensor)
                    else None
                ),
                metrics=sample_metrics,
                batch_index=batch_index,
            )
        rows.append(row)
        append_result_jsonl(raw_path, _json_safe(row))
        inference_time_total += elapsed
        inference_optimization_time_total += inf_opt
    artifact_fields: dict[str, Any] = {}
    if artifact_writer is not None:
        artifact_writer.__exit__(None, None, None)
        artifact_fields = artifact_writer.summary
    return rows, {
        "inference_time_total": inference_time_total,
        "inference_optimization_time_total": inference_optimization_time_total,
        **artifact_fields,
    }


def _batch_metric_payload(pred: torch.Tensor, target: torch.Tensor, batch: PDEBatch, args: argparse.Namespace) -> dict[str, Any]:
    if args.physics_metric_mode == "per_batch":
        physics_pred, physics_input = _joint_physics_views(pred, target, batch.metadata, batch.input_fields)
        metric_meta = {
            "input_fields": physics_input,
            "full_tensor": batch.full_tensor,
            "task": batch.task,
            **batch.metadata,
        }
        physics_metrics = physics_loss_metric(physics_pred, args.pde, dict(metric_meta))
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
        physics_pred_i, physics_input_i = _joint_physics_views(
            pred_i, target_i, item_batch.metadata, item_batch.input_fields
        )
        meta_i = {
            "input_fields": physics_input_i,
            "full_tensor": item_batch.full_tensor,
            "task": item_batch.task,
            **item_batch.metadata,
        }
        physics_metrics = physics_loss_metric(physics_pred_i, args.pde, dict(meta_i))
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
    eval_totals: dict[str, Any],
    args: argparse.Namespace,
    train_dataset: PDEBatchDataset,
    spec_dataset: PDEBatchDataset,
    val_dataset: PDEBatchDataset | None,
    test_dataset: PDEBatchDataset,
    backend_info: dict[str, Any],
    capability_info: dict[str, Any],
    split_info: dict[str, Any],
    method_budget_fields: dict[str, Any],
    normalization_fields: dict[str, Any],
    memory_fields: dict[str, float],
    config_hash: str,
) -> dict[str, Any]:
    sensor_seed = int(getattr(args, "sensor_seed", args.seed))
    source_train_seed = int(getattr(args, "source_train_seed", args.seed))
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
    per_instance = bool(split_info.get("per_instance_baseline", False))
    reported_train_size = 0 if per_instance else int(split_info["effective_train_size"])
    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "run_name": args.run_name,
        "matrix_schema_version": int(getattr(args, "matrix_schema_version", MATRIX_SCHEMA_VERSION)),
        "summary_schema_version": int(getattr(args, "summary_schema_version", 2)),
        "status": "success",
        "execution_mode": str(getattr(args, "execution_mode", "eval_only" if getattr(args, "eval_only", False) else "train")),
        "eval_only": bool(getattr(args, "eval_only", False)),
        "run_fingerprint": str(getattr(args, "run_fingerprint", "")),
        "config_content_sha256": str(getattr(args, "config_content_sha256", "")),
        "experiment_config_sha256": str(
            getattr(args, "experiment_config_sha256", "")
        ),
        "data_manifest_sha256": str(getattr(args, "data_manifest_sha256", "")),
        "data_manifest_path": str(getattr(args, "data_manifest_path", "")),
        "task_protocol_version": str(getattr(args, "task_protocol_version", "2")),
        "sensor_protocol_version": str(getattr(args, "sensor_protocol_version", "2")),
        "comparison_track": str(getattr(args, "comparison_track", "unified_adapted")),
        "source_train_run_id": str(rows[0].get("source_train_run_id", "") if rows else ""),
        "source_train_run_fingerprint": str(rows[0].get("source_train_run_fingerprint", "") if rows else ""),
        "source_train_seed": int(
            rows[0].get(
                "source_train_seed",
                source_train_seed if getattr(args, "eval_only", False) else args.seed,
            )
            if rows
            else (source_train_seed if getattr(args, "eval_only", False) else args.seed)
        ),
        "checkpoint_sha256": str(rows[0].get("checkpoint_sha256", "") if rows else ""),
        **_experiment_fields(args),
        **method_budget_fields,
        "pde": args.pde,
        "task": args.task,
        "baseline": args.baseline,
        "seed": args.seed,
        "sensor_seed": sensor_seed,
        "split": "test",
        "train_size": reported_train_size,
        "train_requested_size": int(split_info["train_requested_size"]),
        "effective_train_size": reported_train_size,
        "train_size_loaded_for_fit": int(split_info["train_size_loaded_for_fit"]),
        "train_size_loaded_for_spec": int(split_info["train_size_loaded_for_spec"]),
        "val_size": len(val_dataset) if val_dataset is not None else 0,
        "val_requested_size": int(split_info["val_requested_size"]),
        "val_split_source": split_info["val_split_source"],
        "val_from_train_offset": split_info["val_from_train_offset"],
        "test_size": len(test_dataset),
        "train_shards": args.train_shards,
        **_requested_design_fields(args),
        "data_loading_mode": args.data_loading_mode,
        "effective_data_loading_mode": getattr(args, "effective_data_loading_mode", "eager"),
        **_dataloader_fields(args),
        **memory_fields,
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
        "sensor_budget_mode": str(test_dataset.batch.metadata.get("sensor_budget_mode", args.sensor_budget_mode if args.task.startswith("sparse") else "none")),
        "num_sensors_per_time": json.dumps(test_dataset.batch.metadata.get("num_sensors_per_time", [] if args.task.startswith("sparse") else [])),
        "num_observations_total": int(test_dataset.batch.metadata.get("num_observations_total", 0) or 0),
        "requested_sensor_mode": args.sensor_mode if args.task.startswith("sparse") else "none",
        "effective_sensor_mode": str(test_dataset.batch.metadata.get("effective_sensor_mode", args.sensor_mode if args.task.startswith("sparse") else "none")),
        "sensor_mode": str(test_dataset.batch.metadata.get("effective_sensor_mode", args.sensor_mode if args.task.startswith("sparse") else "none")),
        "time_varying_sensor_valid": bool(test_dataset.batch.metadata.get("time_varying_sensor_valid", True)),
        "noise_level": args.noise_level,
        "mask_id": test_dataset.batch.metadata.get("mask_id", ""),
        "mask_ids_unique_count": len(set(test_dataset.batch.metadata.get("mask_ids", []) or [])),
        "split_mask_manifest": json.dumps(
            _split_mask_manifest(train_dataset, val_dataset, test_dataset), sort_keys=True
        ),
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
        "learned_state_injection_mode_counts": json.dumps(dict(Counter(str(row.get("learned_state_injection_mode", "")) for row in rows if row.get("learned_state_injection_mode"))), sort_keys=True),
        "posterior_particles": max((int(row.get("posterior_particles", 0) or 0) for row in rows), default=0),
        "inference_time_total": eval_totals["inference_time_total"],
        "inference_time_per_sample": eval_totals["inference_time_total"] / max(len(test_dataset), 1),
        "inference_optimization_time_total": eval_totals["inference_optimization_time_total"],
        "inference_optimization_time_per_sample": eval_totals["inference_optimization_time_total"] / max(len(test_dataset), 1),
        "optimization_steps_completed_total": int(sum(int(row.get("optimization_steps_completed", 0) or 0) for row in rows)),
        "optimization_early_stopped_batches": int(sum(bool(row.get("optimization_early_stopped", False)) for row in rows)),
        "fit_setup_time": float(split_info.get("fit_setup_time", 0.0) or 0.0),
        "test_time_optimization": per_instance,
        "amortized_training": not per_instance,
        "sample_artifact_count": int(eval_totals.get("sample_artifact_count", 0) or 0),
        "sample_artifact_schema_version": str(eval_totals.get("sample_artifact_schema_version", "")),
        "sample_artifact_dir": str(eval_totals.get("sample_artifact_dir", "")),
        "sample_manifest_path": str(eval_totals.get("sample_manifest_path", "")),
        "sample_pdf_path": str(eval_totals.get("sample_pdf_path", "")),
        "config_hash": config_hash,
        **_train_history_fields(rows[0].get("train_history", "{}") if rows else {}),
        **normalization_fields,
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


def _make_inference_batch(batch: PDEBatch) -> PDEBatch:
    """Create the model-facing view without labels or hidden full truth.

    Targets remain available to the evaluator outside this object. A zero tensor
    preserves the declared output shape for per-instance algorithms without
    exposing its values. Full/source/initial truth is similarly removed from
    the model-facing metadata.
    """
    private_keys = {
        "full_tensor",
        "full_trajectory",
        "original_input_fields",
        "observed_solution_fields",
        "observation_source_fields",
        "background_fields",
        "solution_fields",
        "source_fields",
        "coeff_fields",
        "initial_1d",
    }
    metadata = {key: value for key, value in batch.metadata.items() if key not in private_keys}
    metadata["inference_truth_hidden"] = True
    return PDEBatch(
        pde_name=batch.pde_name,
        task=batch.task,
        full_tensor=torch.zeros_like(batch.full_tensor),
        input_fields=batch.input_fields,
        target_fields=torch.zeros_like(batch.target_fields),
        coords=batch.coords,
        mask=batch.mask,
        obs_values=batch.obs_values,
        obs_coords=batch.obs_coords,
        channel_names=list(batch.channel_names),
        input_channel_names=list(batch.input_channel_names),
        target_channel_names=list(batch.target_channel_names),
        metadata=metadata,
        pde_params=dict(batch.pde_params),
        split=batch.split,
        sample_indices=batch.sample_indices,
        global_sample_ids=list(batch.global_sample_ids),
        file_paths=list(batch.file_paths),
    )


def _copy_eval_metadata(dst: PDEBatch, src: PDEBatch) -> None:
    for key in (
        "inference_optimization_time",
        "assimilation_mode",
        "inverse_observation_operator_used",
        "posterior_particles",
        "official_alignment_level",
        "learned_state_injection_mode",
        "learned_state_shape",
        "optimized_state_shape",
        "optimization_steps_completed",
        "optimization_early_stopped",
        "optimization_status_per_sample",
        "pc_bnn_joint_field_posterior",
        "pc_bnn_posterior_objective",
        "pc_bnn_task_adapter",
        "posterior_noise_precision",
    ):
        if key in src.metadata:
            dst.metadata[key] = src.metadata[key]


def _batch_loaded_full_trajectory(batch: PDEBatch) -> bool:
    return bool(
        batch.metadata.get("loaded_full_trajectory", False)
        or batch.full_tensor.ndim == 5
        or (
            batch.full_tensor.ndim == 4
            and str(batch.metadata.get("canonical_layout", "")) == "NCTX"
            and int(batch.full_tensor.shape[2]) > 1
        )
        or isinstance(batch.metadata.get("full_trajectory"), torch.Tensor)
    )


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


def _joint_reconstruction_relative_l2_values(
    pred: torch.Tensor,
    target: torch.Tensor,
    metadata: dict[str, Any],
) -> tuple[list[float], list[float]]:
    split_axis = metadata.get("joint_split_axis")
    input_extent = int(metadata.get("joint_input_extent", 0) or 0)
    if metadata.get("joint_reconstruction") and split_axis is not None and input_extent > 0:
        axis = int(split_axis)
        if axis < 0:
            axis += target.ndim
        if axis <= 0 or axis >= target.ndim or target.shape[axis] <= input_extent:
            raise ValueError(
                f"invalid joint reconstruction split: axis={split_axis}, input_extent={input_extent}, "
                f"target_shape={tuple(target.shape)}"
            )
        input_pred = pred.narrow(axis, 0, input_extent)
        input_target = target.narrow(axis, 0, input_extent)
        solution_extent = int(target.shape[axis] - input_extent)
        solution_pred = pred.narrow(axis, input_extent, solution_extent)
        solution_target = target.narrow(axis, input_extent, solution_extent)
        return (
            _relative_l2_values(solution_pred, solution_target),
            _relative_l2_values(input_pred, input_target),
        )
    input_channels = int(metadata.get("joint_input_channels", 0) or 0)
    if not metadata.get("joint_reconstruction") or input_channels <= 0 or target.shape[1] <= input_channels:
        return _relative_l2_values(pred, target), [float("nan")] * int(target.shape[0])
    return (
        _relative_l2_values(pred[:, input_channels:], target[:, input_channels:]),
        _relative_l2_values(pred[:, :input_channels], target[:, :input_channels]),
    )


def _joint_physics_views(
    pred: torch.Tensor,
    target: torch.Tensor,
    metadata: dict[str, Any],
    fallback_input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_channels = int(metadata.get("joint_input_channels", 0) or 0)
    if not metadata.get("joint_reconstruction") or input_channels <= 0 or pred.shape[1] <= input_channels:
        return pred, fallback_input
    return pred[:, input_channels:], target[:, :input_channels]


def _obs_mse_clean(pred: torch.Tensor, target: torch.Tensor, batch: PDEBatch) -> float:
    if batch.task in {"sparse_inverse", "sparse_forward"}:
        return float("nan")
    return float(obs_mse(pred, target, batch.mask).detach().cpu())


def _obs_mse_noisy(pred: torch.Tensor, batch: PDEBatch) -> float:
    if batch.task in {"sparse_inverse", "sparse_forward"} or batch.mask is None or batch.obs_values is None:
        return float("nan")
    try:
        c = min(pred.shape[1], batch.obs_values.shape[-1])
        masks = batch.mask
        if tuple(masks.shape) == tuple(pred.shape[1:]):
            masks = masks.unsqueeze(0).expand(pred.shape[0], *masks.shape)
        if tuple(masks.shape) != tuple(pred.shape):
            return float("nan")
        pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
        pred_obs = torch.stack(
            [
                pred_flat[i, :, masks[i, 0].bool().reshape(-1)].transpose(0, 1)
                for i in range(pred.shape[0])
            ],
            dim=0,
        )[..., :c]
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
    raise ValueError(
        f"Prediction and target shapes must exactly match; got {tuple(pred.shape)} and {tuple(target.shape)}. "
        "Silent cropping or broadcasting is forbidden."
    )


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


def _train_history_fields(train_history: Any) -> dict[str, Any]:
    if isinstance(train_history, str):
        try:
            train_history = json.loads(train_history)
        except Exception:
            train_history = {}
    if not isinstance(train_history, dict):
        train_history = {}
    return {
        "best_val_loss": train_history.get("best_val_loss"),
        "best_epoch": train_history.get("best_epoch"),
    }


def _write_normalization_stats(out_dir: Path, run_prefix: str, model) -> str:
    stats = getattr(model, "normalization_stats", None)
    if stats is None:
        return ""
    path = out_dir / f"{run_prefix}_normalization_stats.json"
    path.write_text(json.dumps(_json_safe(stats.json_summary()), indent=2, sort_keys=True), encoding="utf-8")
    model.normalization_stats_path = str(path)
    return str(path)


def _normalization_fields(model, stats_path: str = "") -> dict[str, Any]:
    stats = getattr(model, "normalization_stats", None)
    uses = bool(getattr(model, "uses_normalization", False) and stats is not None)
    fields: dict[str, Any] = {
        "normalize": uses,
        "uses_normalization": uses,
        "normalization_stats_path": stats_path or str(getattr(model, "normalization_stats_path", "")),
        "input_mean": "",
        "input_std": "",
        "target_mean": "",
        "target_std": "",
    }
    if stats is None:
        return fields
    summary = stats.json_summary()
    fields.update(
        {
            "input_mean": json.dumps(summary["input_mean"]),
            "input_std": json.dumps(summary["input_std"]),
            "target_mean": json.dumps(summary["target_mean"]),
            "target_std": json.dumps(summary["target_std"]),
        }
    )
    return fields


def _file_sha1(path: Path) -> str:
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()[:16]
    except Exception:
        return ""


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def _synchronize_device(device: str | torch.device) -> None:
    """Make wall-clock timings include queued CUDA work and host transfers."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


def _config_content_sha256(config_path: str | None, cfg: dict[str, Any]) -> str:
    if config_path:
        path = Path(config_path)
        if path.is_file():
            return _file_sha256(path)
    encoded = json.dumps(_json_safe(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_data_manifest_binding(args: argparse.Namespace) -> None:
    """Fail closed on missing, stale, or non-full formal data evidence."""
    manifest_sha256 = str(getattr(args, "data_manifest_sha256", "") or "").strip().lower()
    manifest_path_text = str(getattr(args, "data_manifest_path", "") or "").strip()
    experiment_config_sha256 = str(
        getattr(args, "experiment_config_sha256", "") or ""
    ).strip().lower()
    formal_paper_run = args.experiment_mode == "paper" and requires_full_data_manifest(
        args.task_protocol_version
    )
    if formal_paper_run and (
        not manifest_sha256
        or not manifest_path_text
        or not experiment_config_sha256
    ):
        raise ValueError(
            f"paper runs using task protocol {DEFAULT_TASK_PROTOCOL_VERSION} require "
            "--data-manifest-sha256, --data-manifest-path, and "
            "--experiment-config-sha256 from a passing full data-protocol report"
        )
    if experiment_config_sha256:
        _require_cli_sha256(
            experiment_config_sha256, "--experiment-config-sha256"
        )
    if manifest_sha256 and (
        len(manifest_sha256) != 64 or any(character not in "0123456789abcdef" for character in manifest_sha256)
    ):
        raise ValueError("--data-manifest-sha256 must be a 64-character hexadecimal SHA-256 digest")
    if manifest_path_text:
        if not manifest_sha256:
            raise ValueError("--data-manifest-path requires --data-manifest-sha256")
        manifest, manifest_path, _observed_sha256 = validate_full_data_manifest(
            manifest_path_text,
            expected_sha256=manifest_sha256,
            expected_experiment_config_sha256=experiment_config_sha256,
            expected_data_root=args.data_root,
            expected_verifier_sha256=sha256_file(
                ROOT / "scripts" / "verify_data_protocol.py"
            ),
            verify_source_signatures=True,
        )
        if str(args.pde) not in {str(value) for value in manifest.get("pdes", [])}:
            raise ValueError(
                f"data manifest does not cover requested PDE {args.pde!r}: {manifest_path}"
            )
        args.data_manifest_path = manifest_path
    args.data_manifest_sha256 = manifest_sha256
    args.experiment_config_sha256 = experiment_config_sha256


def _require_cli_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(
            f"{field} must be a 64-character hexadecimal SHA-256 digest"
        )


def _validate_and_bind_run_fingerprint(
    args: argparse.Namespace, method_cfg: dict[str, Any]
) -> None:
    sparse = str(args.task).startswith("sparse")
    if args.execution_mode == "eval_only":
        checkpoint_path = Path(str(args.checkpoint or ""))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"eval-only checkpoint not found for fingerprint validation: {checkpoint_path}"
            )
        observed_checkpoint_sha256 = sha256_file(checkpoint_path)
        supplied_checkpoint_sha256 = str(args.checkpoint_sha256 or "").lower()
        if supplied_checkpoint_sha256 and supplied_checkpoint_sha256 != observed_checkpoint_sha256:
            raise ValueError(
                "--checkpoint-sha256 does not match --checkpoint: "
                f"supplied={supplied_checkpoint_sha256}, observed={observed_checkpoint_sha256}"
            )
        args.checkpoint_sha256 = observed_checkpoint_sha256

    payload: dict[str, Any] = {
        "matrix_schema_version": int(args.matrix_schema_version),
        "summary_schema_version": int(args.summary_schema_version),
        "execution_mode": str(args.execution_mode),
        "comparison_track": str(args.comparison_track),
        "experiment_kind": str(args.experiment_kind),
        "ablation_factor": str(args.ablation_factor),
        "task_group": str(args.task_group),
        "baseline": args.baseline,
        "pde": args.pde,
        "task": args.task,
        "task_protocol_version": str(args.task_protocol_version),
        "sensor_protocol_version": str(args.sensor_protocol_version),
        "seed": int(args.seed),
        "sensor_seed": int(args.sensor_seed),
        "train_size": int(args.train_size),
        "val_size": int(args.val_size),
        "test_size": int(args.test_size),
        "train_shards": int(args.train_shards),
        "batch_size": int(args.batch_size),
        "epochs": int(method_cfg.get("epochs", 0) or 0),
        "device": str(args.device),
        "sensor_mode": str(args.sensor_mode) if sparse else "none",
        "sensor_budget_mode": str(args.sensor_budget_mode) if sparse else "none",
        "num_sensors": int(args.num_sensors) if sparse else 0,
        "noise_level": float(args.noise_level) if sparse else 0.0,
        "steps": int(method_cfg.get("steps", 0) or 0),
        "refine_steps": int(method_cfg.get("refine_steps", 0) or 0),
        "particles": int(method_cfg.get("particles", 0) or 0),
        "scalar_param_mode": str(args.scalar_param_mode),
        "data_loading_mode": str(args.data_loading_mode_requested),
        "num_workers": int(args.num_workers_requested),
        "pin_memory": bool(args.pin_memory_requested),
        "persistent_workers": bool(args.persistent_workers_requested),
        "prefetch_factor": int(args.prefetch_factor_requested),
        "load_full_trajectory": bool(args.load_full_trajectory),
        "config_content_sha256": args.config_content_sha256,
        "experiment_config_sha256": str(args.experiment_config_sha256),
        "data_manifest_sha256": str(args.data_manifest_sha256),
        "commit_hash": str(args.commit_hash),
    }
    if args.execution_mode == "eval_only":
        payload.update(
            {
                "source_train_run_id": str(args.source_train_run_id),
                "source_train_run_fingerprint": str(args.source_train_run_fingerprint),
                "source_train_seed": int(args.source_train_seed),
                "checkpoint_sha256": str(args.checkpoint_sha256),
            }
        )
    observed_fingerprint = run_fingerprint(payload)
    supplied_fingerprint = str(args.run_fingerprint or "")
    if supplied_fingerprint and supplied_fingerprint != observed_fingerprint:
        raise ValueError(
            "--run-fingerprint does not match the effective execution design: "
            f"supplied={supplied_fingerprint}, observed={observed_fingerprint}; "
            "launch through run_one.py or regenerate the matrix"
        )
    args.run_fingerprint = observed_fingerprint


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
            if mode == "official":
                get_ifno_official_status()
            elif mode == "official_aligned":
                get_ifno_official_aligned_status()
            else:
                try:
                    get_ifno_official_status()
                except OfficialImportError:
                    if not bool(getattr(capability, "official_aligned_allowed", False)):
                        raise
                    get_ifno_official_aligned_status()
            return None
        if args.baseline == "vivid" and capability.task_family == "time_varying_da":
            if mode == "official":
                get_vivid_official_status()
            elif mode == "official_aligned":
                get_vivid_official_aligned_status()
            else:
                try:
                    get_vivid_official_status()
                except OfficialImportError:
                    if not bool(getattr(capability, "official_aligned_allowed", False)):
                        raise
                    get_vivid_official_aligned_status()
            return None
        if args.baseline == "pc_bnn" and capability.task_family == "sparse_reconstruction":
            if args.pde.lower() != "shallow_water":
                raise OfficialImportError("official-aligned PC-BNN is only enabled for 2D three-channel shallow-water fields")
            if mode == "official":
                get_pc_bnn_net_class()
            elif mode == "official_aligned":
                get_pc_bnn_official_aligned_status()
            else:
                try:
                    get_pc_bnn_net_class()
                except OfficialImportError:
                    if not bool(getattr(capability, "official_aligned_allowed", False)):
                        raise
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
        "official_vendored_path": str(getattr(model, "official_vendored_path", "")),
        "official_local_modifications": str(getattr(model, "official_local_modifications", "")),
        "official_metadata_note": str(getattr(model, "official_metadata_note", "")),
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
        "official_vendored_path": "",
        "official_local_modifications": "",
        "official_metadata_note": "",
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
        "official_vendored_path": backend_info.get("official_vendored_path", ""),
        "official_local_modifications": backend_info.get("official_local_modifications", ""),
        "official_metadata_note": backend_info.get("official_metadata_note", ""),
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
        "unified_comparison_eligible": bool(capability_info.get("unified_comparison_eligible", False)),
        "official_native_eligible": bool(capability_info.get("official_native_eligible", False)),
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


def _split_mask_manifest(
    train_dataset: PDEBatchDataset,
    val_dataset: PDEBatchDataset | None,
    test_dataset: PDEBatchDataset,
) -> dict[str, Any]:
    def one(dataset: PDEBatchDataset | None) -> dict[str, Any]:
        if dataset is None:
            return {"mask_id": "", "sample_mask_count": 0, "unique_mask_count": 0, "mask_ids_sha256": ""}
        metadata = dataset.batch.metadata
        mask_ids = [str(value) for value in metadata.get("mask_ids", []) or [] if value]
        encoded = json.dumps(mask_ids, separators=(",", ":")).encode("utf-8")
        return {
            "mask_id": str(metadata.get("mask_id", "")),
            "sample_mask_count": len(mask_ids),
            "unique_mask_count": len(set(mask_ids)),
            "mask_ids_sha256": hashlib.sha256(encoded).hexdigest() if mask_ids else "",
        }

    return {"train_epoch0": one(train_dataset), "validation": one(val_dataset), "test": one(test_dataset)}


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
    normalization_fields: dict[str, Any],
    memory_fields: dict[str, float],
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
        "effective_data_loading_mode": getattr(args, "effective_data_loading_mode", "eager"),
        **_dataloader_fields(args),
        **memory_fields,
        "load_full_trajectory": bool(args.load_full_trajectory),
        "sensor_budget_mode": args.sensor_budget_mode,
        "normalization": normalization_fields,
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
