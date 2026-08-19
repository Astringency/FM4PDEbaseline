from __future__ import annotations

import copy
import json
import math
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.data_adapter import PDEBatch
from baselines.common.normalization import (
    NormalizationStats,
    denormalize_prediction,
    estimate_normalization_stats,
    normalize_batch_input_target,
)


class BaselineModel(nn.Module):
    name: str = "base"

    def __init__(self) -> None:
        super().__init__()
        self.config: dict[str, Any] = {}
        self.data_spec: dict[str, Any] = {}
        self.backend_used: str = "local"
        self.official_backend: str = "local"
        self.fallback_used: bool = False
        self.backend_warning: str = ""
        self.implementation_mode_requested: str = "adapted"
        self.implementation_mode_effective: str = "adapted"
        self.implementation_source: str = "local"
        self.official_repo: str = ""
        self.official_commit_or_version: str = ""
        self.official_import_path: str = ""
        self.official_vendored_path: str = ""
        self.official_local_modifications: str = ""
        self.official_metadata_note: str = ""
        self.official_import_success: bool = False
        self.official_reimplementation_success: bool = False
        self.official_alignment_level: str = "local"
        self.official_alignment_notes: str = ""
        self.adapter_status: str = "local_adapted"
        self.normalization_stats: NormalizationStats | None = None
        self.uses_normalization: bool = False
        self.normalization_stats_path: str = ""
        self.provenance: dict[str, Any] = {}

    def build(self, config, data_spec):
        self.config = dict(config or {})
        self.data_spec = dict(data_spec or {})
        self.implementation_mode_requested = _requested_implementation_mode(self.config)
        return self

    def set_backend(
        self,
        backend_used: str,
        official_backend: str | None = None,
        fallback_used: bool = False,
        warning: str = "",
        implementation_mode_effective: str | None = None,
        implementation_source: str | None = None,
        official_repo: str = "",
        official_commit_or_version: str = "",
        official_import_path: str = "",
        official_vendored_path: str = "",
        official_local_modifications: str = "",
        official_metadata_note: str = "",
        official_import_success: bool | None = None,
        official_reimplementation_success: bool | None = None,
        official_alignment_level: str | None = None,
        official_alignment_notes: str = "",
        adapter_status: str | None = None,
    ) -> None:
        self.backend_used = str(backend_used)
        self.official_backend = str(official_backend or backend_used)
        self.fallback_used = bool(fallback_used)
        self.backend_warning = str(warning or "")
        effective = implementation_mode_effective or _default_effective_mode(self.backend_used, self.fallback_used)
        self.implementation_mode_effective = str(effective)
        self.implementation_source = str(implementation_source or self.backend_used)
        self.official_repo = str(official_repo or "")
        self.official_commit_or_version = str(official_commit_or_version or "")
        self.official_import_path = str(official_import_path or "")
        self.official_vendored_path = str(official_vendored_path or official_import_path or "")
        self.official_local_modifications = str(official_local_modifications or "")
        self.official_metadata_note = str(official_metadata_note or "")
        if official_import_success is None:
            official_import_success = self.implementation_mode_effective == "official" and not self.fallback_used
        self.official_import_success = bool(official_import_success)
        if official_reimplementation_success is None:
            official_reimplementation_success = self.implementation_mode_effective in {"official_architecture", "official_aligned"} and not self.fallback_used
        self.official_reimplementation_success = bool(official_reimplementation_success)
        self.official_alignment_level = str(official_alignment_level or _default_alignment_level(self.implementation_mode_effective))
        self.official_alignment_notes = str(official_alignment_notes or "")
        self.adapter_status = str(adapter_status or _default_adapter_status(self.implementation_mode_effective, self.fallback_used))

    def mark_canonical_math(self, source: str, adapter_status: str = "canonical_math") -> None:
        self.set_backend(
            source,
            source,
            fallback_used=False,
            implementation_mode_effective="canonical_math",
            implementation_source=source,
            official_import_success=False,
            official_reimplementation_success=False,
            official_alignment_level="objective",
            adapter_status=adapter_status,
        )

    def fit(self, train_loader, val_loader=None):
        return {}

    def predict(self, batch: PDEBatch):
        raise NotImplementedError

    def supervised_training_pair(self, batch: PDEBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the prediction and target used by the supervised optimizer.

        Most baselines train on their complete output field. Models whose
        official training recipe samples outputs before an expensive decoder
        can override this seam without changing full-field inference.
        """
        return self.predict(batch), batch.target_fields

    def predict_physical(self, batch: PDEBatch):
        if self.uses_normalization and self.normalization_stats is not None:
            norm_batch = normalize_batch_input_target(batch, self.normalization_stats)
            pred = self.predict(norm_batch)
            return denormalize_prediction(pred, self.normalization_stats)
        return self.predict(batch)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 2,
                "baseline": self.name,
                "state_dict": self.state_dict(),
                "config": self.config,
                "data_spec": self.data_spec,
                "backend": self.backend_metadata(),
                "uses_normalization": self.uses_normalization,
                "normalization_stats": self.normalization_stats.state_dict() if self.normalization_stats is not None else None,
                "provenance": dict(self.provenance),
            },
            path,
        )

    def load(self, path, map_location="cpu"):
        payload = torch.load(path, map_location=map_location, weights_only=False)
        return self.load_payload(payload)

    def load_payload(self, payload: dict[str, Any]):
        restore_state_dict(self, payload["state_dict"])
        self.config = payload.get("config", {})
        self.data_spec = payload.get("data_spec", {})
        backend = payload.get("backend", {})
        for key, value in backend.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.uses_normalization = bool(payload.get("uses_normalization", False))
        self.normalization_stats = NormalizationStats.from_state_dict(payload.get("normalization_stats"))
        self.provenance = dict(payload.get("provenance", {}))
        return self

    def parameter_count(self) -> int:
        """Count trainable real scalar degrees of freedom.

        PyTorch stores one complex value as one element, but official
        NeuralOperator/iFNO parameter reports count its real and imaginary
        parts separately. This definition makes comparisons consistent.
        """
        return int(sum(p.numel() * (2 if p.is_complex() else 1) for p in self.parameters() if p.requires_grad))

    def parameter_storage_count(self) -> int:
        """Return raw PyTorch trainable ``numel`` for provenance/debugging."""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def backend_metadata(self) -> dict[str, Any]:
        return {
            "backend_used": self.backend_used,
            "official_backend": self.official_backend,
            "fallback_used": self.fallback_used,
            "backend_warning": self.backend_warning,
            "implementation_mode_requested": self.implementation_mode_requested,
            "implementation_mode_effective": self.implementation_mode_effective,
            "implementation_source": self.implementation_source,
            "official_repo": self.official_repo,
            "official_commit_or_version": self.official_commit_or_version,
            "official_import_path": self.official_import_path,
            "official_vendored_path": self.official_vendored_path,
            "official_local_modifications": self.official_local_modifications,
            "official_metadata_note": self.official_metadata_note,
            "official_import_success": self.official_import_success,
            "official_reimplementation_success": self.official_reimplementation_success,
            "official_alignment_level": self.official_alignment_level,
            "official_alignment_notes": self.official_alignment_notes,
            "adapter_status": self.adapter_status,
            "uses_normalization": self.uses_normalization,
            "normalization_stats_path": self.normalization_stats_path,
        }


def _requested_implementation_mode(config: dict[str, Any]) -> str:
    if "implementation_mode" in config:
        return str(config.get("implementation_mode") or "").lower()
    backend = str(config.get("official_backend", "auto")).lower()
    if backend in {"local", "none", "adapted"}:
        return "adapted"
    if backend == "official":
        return "official"
    if backend in {"neuraloperator", "deepxde", "recfno", "senseiver", "pc_bnn", "ifno"}:
        return "official"
    return backend or "auto"


def _default_effective_mode(backend_used: str, fallback_used: bool) -> str:
    backend = str(backend_used).lower()
    if backend in {"pde_opt", "var4d", "canonical_math"}:
        return "canonical_math"
    if backend in {"official_architecture", "official_aligned"}:
        return backend
    if fallback_used or backend in {"local", "none"}:
        return "adapted"
    return "official"


def _default_adapter_status(effective_mode: str, fallback_used: bool) -> str:
    if fallback_used:
        return "fallback_adapted"
    mode = str(effective_mode).lower()
    if mode == "official":
        return "official_code"
    if mode == "official_architecture":
        return "official_architecture_reimplementation"
    if mode == "official_aligned":
        return "official_aligned_reimplementation"
    if mode == "canonical_math":
        return "canonical_math"
    return "local_adapted"


def _default_alignment_level(effective_mode: str) -> str:
    mode = str(effective_mode).lower()
    if mode == "official":
        return "exact_code"
    if mode == "official_architecture":
        return "architecture"
    if mode == "official_aligned":
        return "objective"
    if mode == "canonical_math":
        return "objective"
    return "local"


class LossPlateauStopper:
    """Shared convergence rule for per-instance optimization baselines."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("early_stopping", False))
        self.patience = max(1, int(config.get("early_stopping_patience", 20)))
        self.min_delta = float(config.get("early_stopping_min_delta", 1e-4))
        self.min_steps = max(
            1,
            int(
                config.get(
                    "min_optimization_steps",
                    config.get("min_steps", config.get("min_epochs", 1)),
                )
            ),
        )
        self.best: float | None = None
        self.no_improve = 0
        self.completed_steps = 0
        self.early_stopped = False
        self.last_improved = False
        self.best_step: int | None = None

    def update(self, loss: torch.Tensor | float, *, completed_step: bool = True) -> bool:
        value = float(loss.detach().cpu()) if isinstance(loss, torch.Tensor) else float(loss)
        if completed_step:
            self.completed_steps += 1
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.no_improve = 0
            self.last_improved = True
            self.best_step = self.completed_steps
        else:
            self.no_improve += 1
            self.last_improved = False
        self.early_stopped = (
            self.enabled
            and self.completed_steps >= self.min_steps
            and self.no_improve >= self.patience
        )
        return self.early_stopped

    def status(self) -> dict[str, Any]:
        return {
            "completed_steps": int(self.completed_steps),
            "early_stopped": bool(self.early_stopped),
            "best_loss": self.best,
            "best_step": self.best_step,
            "min_delta": self.min_delta,
            "patience": self.patience,
        }


def run_per_instance_optimizer(optimizer, closure, steps: int, config: dict[str, Any]) -> dict[str, Any]:
    """Optimize one test instance and restore the best evaluated parameter state.

    Losses are observed after each optimizer update.  This matters for Adam and
    especially for PyTorch LBFGS, whose ``step`` return value is the first
    closure loss rather than the loss at the final parameters.
    """
    stopper = LossPlateauStopper(config)
    params = _optimizer_parameters(optimizer)
    current_loss = closure()
    stopper.update(current_loss, completed_step=False)
    best_params = _snapshot_parameters(params)
    is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
    if is_lbfgs:
        # One outer iteration must correspond to one recorded budget step.
        # Leaving max_iter=steps here would hide all internal iterations behind
        # one stale return value and make patience impossible to apply.
        for group in optimizer.param_groups:
            group["max_iter"] = 1
            group["max_eval"] = max(int(group.get("max_eval", 1)), 1)
    for _ in range(max(int(steps), 0)):
        if is_lbfgs:
            optimizer.step(closure)
        else:
            # ``current_loss`` was produced by closure and its gradients match
            # the current parameters.
            optimizer.step()
        current_loss = closure()
        if stopper.update(current_loss):
            if stopper.last_improved:
                best_params = _snapshot_parameters(params)
            break
        if stopper.last_improved:
            best_params = _snapshot_parameters(params)
    restored_best = bool(config.get("restore_best", True))
    if restored_best:
        _restore_parameters(params, best_params)
    optimizer.zero_grad(set_to_none=True)
    status = stopper.status()
    status["restored_best"] = restored_best
    return status


def _optimizer_parameters(optimizer) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in seen:
                seen.add(id(parameter))
                params.append(parameter)
    return params


def _snapshot_parameters(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in params]


def _restore_parameters(params: list[torch.nn.Parameter], snapshot: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for parameter, value in zip(params, snapshot):
            parameter.copy_(value)


def record_optimization_status(batch: PDEBatch, statuses: list[dict[str, Any]]) -> None:
    if not statuses:
        return
    batch.metadata["optimization_steps_completed"] = int(sum(int(s["completed_steps"]) for s in statuses))
    batch.metadata["optimization_early_stopped"] = bool(all(bool(s["early_stopped"]) for s in statuses))
    batch.metadata["optimization_status_per_sample"] = statuses


def snapshot_state_dict(model: nn.Module) -> OrderedDict[str, Any]:
    """Take a CPU snapshot of a possibly non-standard state_dict."""
    snapshot: OrderedDict[str, Any] = OrderedDict()
    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            snapshot[key] = value.detach().cpu().clone()
            continue
        try:
            snapshot[key] = copy.deepcopy(value)
        except Exception as exc:
            warnings.warn(
                f"Skipping non-tensor state_dict key {key!r} during best-state snapshot because deepcopy failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return snapshot


def restore_state_dict(model: nn.Module, state_dict: OrderedDict[str, Any] | dict[str, Any]) -> None:
    """Restore a snapshot, falling back only for state_dicts with non-tensor entries."""
    try:
        model.load_state_dict(state_dict)
        return
    except Exception as exc:
        non_tensor_keys = [key for key, value in state_dict.items() if not isinstance(value, torch.Tensor)]
        if not non_tensor_keys:
            raise
        tensor_state: OrderedDict[str, torch.Tensor] = OrderedDict(
            (key, value) for key, value in state_dict.items() if isinstance(value, torch.Tensor)
        )
        warnings.warn(
            "Strict best-state restore failed for a state_dict containing non-tensor entries; "
            f"retrying with tensor-only strict=False restore. Non-tensor keys skipped: {non_tensor_keys}. "
            f"Original error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        result = model.load_state_dict(tensor_state, strict=False)
        missing_keys = list(getattr(result, "missing_keys", []))
        if missing_keys:
            raise RuntimeError(
                "Tensor-only best-state restore left real model parameters or buffers missing: "
                f"{missing_keys}. Original strict restore error: {exc}"
            ) from exc


def run_supervised_fit(model: BaselineModel, train_loader, val_loader=None):
    device = torch.device(model.config.get("device", "cpu"))
    epochs = int(model.config.get("epochs", 1))
    lr = float(model.config.get("lr", 1e-3))
    max_steps = model.config.get("max_steps")
    max_val_steps = model.config.get("max_val_steps")
    log_interval = int(model.config.get("log_interval") or 0)
    normalize = bool(model.config.get("normalize", False))
    if normalize and model.normalization_stats is None:
        max_norm_batches = model.config.get("normalization_max_batches")
        scope = "entire train_loader" if max_norm_batches is None else f"max_batches={max_norm_batches}"
        print(
            f"[normalization] baseline={model.name} pde={model.data_spec.get('pde', '')} "
            f"task={model.data_spec.get('task', '')} start scope={scope}",
            file=sys.stderr,
            flush=True,
        )
        model.normalization_stats = estimate_normalization_stats(
            train_loader,
            max_batches=max_norm_batches,
            eps=float(model.config.get("normalization_eps", 1e-6)),
        )
        stats = model.normalization_stats.json_summary()
        print(
            f"[normalization] baseline={model.name} pde={model.data_spec.get('pde', '')} "
            f"task={model.data_spec.get('task', '')} done num_batches={stats['num_batches']} "
            f"num_samples={stats['num_samples']} input_mean={_compact_float_list(stats['input_mean'])} "
            f"input_std={_compact_float_list(stats['input_std'])} target_mean={_compact_float_list(stats['target_mean'])} "
            f"target_std={_compact_float_list(stats['target_std'])}",
            file=sys.stderr,
            flush=True,
        )
    model.uses_normalization = bool(normalize and model.normalization_stats is not None)
    model.to(device)
    opt = _build_optimizer(model, model.config, lr)
    scheduler = _build_lr_scheduler(opt, model.config, epochs)
    monitor_name = _scheduler_monitor_name(model.config, val_loader)
    early_stopping = bool(model.config.get("early_stopping", False))
    early_stopping_patience = int(model.config.get("early_stopping_patience", 20))
    early_stopping_min_delta = float(model.config.get("early_stopping_min_delta", 1e-4))
    min_epochs = int(model.config.get("min_epochs", 1))
    restore_best = bool(model.config.get("restore_best", True))
    grad_clip_norm = model.config.get("grad_clip_norm")
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)
    loss_name = str(model.config.get("training_loss", "mse")).lower()
    history = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": None,
        "best_val_loss": None,
        "early_stopped": False,
        "stop_epoch": None,
        "stop_reason": "",
        "monitor_name": monitor_name,
        "best_monitor_loss": None,
        "lr_history": [],
        "normalize": model.uses_normalization,
        "normalization_stats": model.normalization_stats.json_summary() if model.normalization_stats is not None else None,
        "training_loss": loss_name,
        "optimizer": str(model.config.get("optimizer", "adam")).lower(),
        "lr_scheduler": str(model.config.get("lr_scheduler", "none")).lower(),
    }
    best_state = None
    best_val = None
    best_train = None
    best_monitor = None
    no_improve_epochs = 0
    cumulative_train_time = 0.0
    for epoch in range(epochs):
        _set_dataset_epoch(train_loader, epoch)
        if val_loader is not None:
            _set_dataset_epoch(val_loader, 0)
        epoch_start = time.perf_counter()
        model.train()
        total = 0.0
        count = 0
        train_batches = 0
        train_samples = 0
        for step, batch in enumerate(train_loader):
            if max_steps is not None and step >= int(max_steps):
                break
            batch = _to_device_batch(batch, device)
            train_samples += int(batch.input_fields.shape[0])
            if model.uses_normalization and model.normalization_stats is not None:
                batch = normalize_batch_input_target(batch, model.normalization_stats)
            opt.zero_grad(set_to_none=True)
            pred, target = model.supervised_training_pair(batch)
            _require_exact_shape(pred, target, model.name, "train")
            optimization_loss, reported_loss = _supervised_losses(model, pred, target)
            _raise_if_nonfinite_loss(optimization_loss, model, epoch + 1, step + 1, "train_loss")
            optimization_loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            opt.step()
            batch_samples = int(target.shape[0])
            total += float(reported_loss.detach().cpu()) * batch_samples
            count += batch_samples
            train_batches += 1
            if log_interval > 0 and (step + 1) % log_interval == 0:
                running = total / max(count, 1)
                print(
                    f"[fit step] baseline={model.name} pde={model.data_spec.get('pde', '')} "
                    f"task={model.data_spec.get('task', '')} epoch={epoch + 1}/{epochs} "
                    f"step={step + 1}/{_safe_len(train_loader)} batch_loss={float(reported_loss.detach().cpu()):.6g} "
                    f"running_train_loss={running:.6g} device={device}{_cuda_mem_text(device)}",
                    file=sys.stderr,
                    flush=True,
                )
        train_loss = total / max(count, 1)
        _raise_if_nonfinite_scalar(train_loss, model, epoch + 1, count, "train_loss")
        history["train_loss"].append(train_loss)
        if val_loader is None and (best_train is None or train_loss < best_train):
            best_train = train_loss
        val_loss = None
        val_count = 0
        val_batches = 0
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    if max_val_steps is not None and step >= int(max_val_steps):
                        break
                    batch = _to_device_batch(batch, device)
                    if model.uses_normalization and model.normalization_stats is not None:
                        batch = normalize_batch_input_target(batch, model.normalization_stats)
                    pred = model.predict(batch)
                    _require_exact_shape(pred, batch.target_fields, model.name, "validation")
                    _optimization_loss, reported_loss = _supervised_losses(model, pred, batch.target_fields)
                    _raise_if_nonfinite_loss(reported_loss, model, epoch + 1, step + 1, "val_loss")
                    batch_samples = int(batch.target_fields.shape[0])
                    val_total += float(reported_loss.detach().cpu()) * batch_samples
                    val_count += batch_samples
                    val_batches += 1
            val_loss = val_total / max(val_count, 1)
            _raise_if_nonfinite_scalar(val_loss, model, epoch + 1, val_count, "val_loss")
            history["val_loss"].append(val_loss)
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                history["best_epoch"] = epoch + 1
                history["best_val_loss"] = val_loss
        monitor_loss = train_loss if monitor_name == "train_loss" else val_loss
        if monitor_loss is None:
            monitor_name = "train_loss"
            history["monitor_name"] = monitor_name
            monitor_loss = train_loss
        improved = best_monitor is None or float(monitor_loss) < float(best_monitor) - early_stopping_min_delta
        if improved:
            best_monitor = float(monitor_loss)
            history["best_monitor_loss"] = best_monitor
            history["best_epoch"] = epoch + 1
            best_state = snapshot_state_dict(model)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        old_lr = float(opt.param_groups[0]["lr"])
        _step_lr_scheduler(scheduler, monitor_loss)
        new_lr = float(opt.param_groups[0]["lr"])
        history["lr_history"].append(new_lr)
        if old_lr != new_lr:
            print(
                f"[lr scheduler] baseline={model.name} pde={model.data_spec.get('pde', '')} "
                f"task={model.data_spec.get('task', '')} epoch={epoch + 1}/{epochs} "
                f"old_lr={old_lr:.6g} new_lr={new_lr:.6g} monitor={monitor_name} monitor_loss={float(monitor_loss):.6g}",
                file=sys.stderr,
                flush=True,
            )
        epoch_time = time.perf_counter() - epoch_start
        cumulative_train_time += epoch_time
        samples_per_sec = train_samples / max(epoch_time, 1e-12)
        should_stop = early_stopping and epoch + 1 >= min_epochs and no_improve_epochs >= early_stopping_patience
        if should_stop:
            history["early_stopped"] = True
            history["stop_epoch"] = epoch + 1
            history["stop_reason"] = (
                f"no improvement in {monitor_name} for {no_improve_epochs} epochs "
                f"(patience={early_stopping_patience}, min_delta={early_stopping_min_delta})"
            )
        _write_incremental_history(model.config, history, epoch + 1)
        print(
            f"[fit epoch] baseline={model.name} pde={model.data_spec.get('pde', '')} "
            f"task={model.data_spec.get('task', '')} epoch={epoch + 1}/{epochs} "
            f"train_loss={history['train_loss'][-1]:.6g} val_loss={_fmt_optional(val_loss)} "
            f"best_val_loss={_fmt_optional(history['best_val_loss'])} best_epoch={history['best_epoch']} "
            f"monitor={monitor_name} monitor_loss={float(monitor_loss):.6g} "
            f"no_improve_epochs={no_improve_epochs} early_stopping_patience={early_stopping_patience} "
            f"epoch_time_sec={epoch_time:.3f} cumulative_train_time_sec={cumulative_train_time:.3f} "
            f"train_steps={train_batches} val_steps={val_batches} samples_per_sec={samples_per_sec:.3f} "
            f"lr={opt.param_groups[0]['lr']:.3e} device={device}{_cuda_mem_text(device)}",
            file=sys.stderr,
            flush=True,
        )
        if should_stop:
            break
    if not history["early_stopped"]:
        history["stop_epoch"] = len(history["train_loss"])
    if best_state is not None and restore_best:
        restore_state_dict(model, best_state)
    return history


def _set_dataset_epoch(loader, epoch: int) -> None:
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "set_epoch"):
        dataset.set_epoch(int(epoch))


def _require_exact_shape(pred: torch.Tensor, target: torch.Tensor, baseline: str, stage: str) -> None:
    if tuple(pred.shape) != tuple(target.shape):
        raise ValueError(
            f"{baseline} {stage} prediction shape {tuple(pred.shape)} must exactly match target shape "
            f"{tuple(target.shape)}; PyTorch broadcasting is forbidden"
        )


def _build_optimizer(model: nn.Module, config: dict[str, Any], lr: float) -> torch.optim.Optimizer:
    optimizer_name = str(config.get("optimizer", "adam")).lower()
    weight_decay = float(config.get("weight_decay", 0.0))
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer={optimizer_name!r}; supported values: adam, adamw")


def _build_lr_scheduler(opt: torch.optim.Optimizer, config: dict[str, Any], epochs: int):
    scheduler_name = str(config.get("lr_scheduler", "none")).lower()
    if scheduler_name in {"", "none", "off", "false"}:
        return None
    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(config.get("scheduler_factor", 0.5)),
            patience=int(config.get("scheduler_patience", 5)),
            threshold=float(config.get("scheduler_threshold", 1e-6)),
            min_lr=float(config.get("scheduler_min_lr", 1e-5)),
        )
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(int(config.get("cosine_t_max", epochs) or epochs), 1),
            eta_min=float(config.get("cosine_min_lr", 1e-5)),
        )
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=max(int(config.get("scheduler_step_size", 100)), 1),
            gamma=float(config.get("scheduler_gamma", 0.5)),
        )
    if scheduler_name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            opt,
            gamma=float(config.get("scheduler_gamma", 0.98)),
        )
    raise ValueError(
        f"Unsupported lr_scheduler={scheduler_name!r}; supported values: "
        "none, reduce_on_plateau, cosine, step, exponential"
    )


def _supervised_losses(
    model: BaselineModel,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the official/adapted optimization loss and a sample-mean report loss."""
    official_loss = getattr(model, "official_training_loss", None)
    if official_loss is not None:
        loss = official_loss(prediction, target)
        return loss, loss
    loss_name = str(model.config.get("training_loss", "mse")).lower()
    if loss_name == "mse":
        loss = F.mse_loss(prediction, target)
        return loss, loss
    if loss_name in {"l1", "mae"}:
        loss = F.l1_loss(prediction, target)
        return loss, loss
    if loss_name == "sum_mse":
        return F.mse_loss(prediction, target, reduction="sum"), F.mse_loss(prediction, target)
    if loss_name in {"relative_l2", "l2"}:
        per_sample = torch.linalg.vector_norm(
            (prediction - target).reshape(prediction.shape[0], -1), dim=1
        ) / torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(1e-12)
        loss = per_sample.mean()
        return loss, loss
    raise ValueError(
        f"Unsupported training_loss={loss_name!r}; supported values: mse, sum_mse, l1, relative_l2"
    )


def _scheduler_monitor_name(config: dict[str, Any], val_loader=None) -> str:
    monitor = str(config.get("scheduler_monitor", "val_loss")).lower()
    if monitor not in {"val_loss", "train_loss"}:
        raise ValueError(f"Unsupported scheduler_monitor={monitor!r}; supported values: val_loss, train_loss")
    if monitor == "val_loss" and val_loader is None:
        return "train_loss"
    return monitor


def _step_lr_scheduler(scheduler, monitor_loss: float | None) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(float(monitor_loss))
    else:
        scheduler.step()


def _raise_if_nonfinite_loss(loss: torch.Tensor, model: BaselineModel, epoch: int, step: int, loss_name: str) -> None:
    if not torch.isfinite(loss.detach()).all().item():
        _raise_nonfinite(model, epoch, step, loss_name, float(loss.detach().cpu()))


def _raise_if_nonfinite_scalar(value: float, model: BaselineModel, epoch: int, step: int, loss_name: str) -> None:
    if not math.isfinite(float(value)):
        _raise_nonfinite(model, epoch, step, loss_name, float(value))


def _raise_nonfinite(model: BaselineModel, epoch: int, step: int, loss_name: str, value: float) -> None:
    raise RuntimeError(
        f"Non-finite loss detected: baseline={model.name} pde={model.data_spec.get('pde', '')} "
        f"task={model.data_spec.get('task', '')} epoch={epoch} step={step} loss_name={loss_name} loss={value}"
    )


def _write_incremental_history(config: dict[str, Any], history: dict[str, Any], epoch: int) -> None:
    json_path = config.get("train_history_json_path")
    jsonl_path = config.get("train_history_jsonl_path")
    payload = _history_json_safe(history)
    payload["completed_epochs"] = epoch
    if json_path:
        path = Path(str(json_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if jsonl_path:
        path = Path(str(jsonl_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, **payload}) + "\n")


def _history_json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _history_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_history_json_safe(v) for v in value]
    return value


def _fmt_optional(value: Any) -> str:
    return "nan" if value is None else f"{float(value):.6g}"


def _safe_len(loader) -> int | str:
    try:
        return len(loader)
    except TypeError:
        return "?"


def _cuda_mem_text(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return ""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(idx) / (1024**3)
    reserved = torch.cuda.memory_reserved(idx) / (1024**3)
    return f" cuda_mem_alloc={alloc:.3f}GB cuda_mem_reserved={reserved:.3f}GB"


def _compact_float_list(values: list[float], limit: int = 4) -> str:
    shown = ",".join(f"{float(v):.4g}" for v in values[:limit])
    return f"[{shown}{',...' if len(values) > limit else ''}]"


def _to_device_batch(batch: PDEBatch, device: torch.device) -> PDEBatch:
    def move(x):
        return x.to(device) if isinstance(x, torch.Tensor) else x

    meta = _move_metadata(batch.metadata, device)
    return PDEBatch(
        pde_name=batch.pde_name,
        task=batch.task,
        full_tensor=batch.full_tensor.to(device),
        input_fields=batch.input_fields.to(device),
        target_fields=batch.target_fields.to(device),
        coords=move(batch.coords),
        mask=move(batch.mask),
        obs_values=move(batch.obs_values),
        obs_coords=move(batch.obs_coords),
        channel_names=batch.channel_names,
        input_channel_names=batch.input_channel_names,
        target_channel_names=batch.target_channel_names,
        metadata=meta,
        pde_params=_move_metadata(batch.pde_params, device),
        split=batch.split,
        sample_indices=batch.sample_indices.to(device) if isinstance(batch.sample_indices, torch.Tensor) else batch.sample_indices,
        global_sample_ids=list(batch.global_sample_ids),
        file_paths=list(batch.file_paths),
    )


def _move_metadata(metadata: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif isinstance(value, dict):
            out[key] = _move_metadata(value, device)
        else:
            out[key] = value
    return out
