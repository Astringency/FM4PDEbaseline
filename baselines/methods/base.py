from __future__ import annotations

import copy
import json
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
                "format_version": 1,
                "baseline": self.name,
                "state_dict": self.state_dict(),
                "config": self.config,
                "data_spec": self.data_spec,
                "backend": self.backend_metadata(),
                "uses_normalization": self.uses_normalization,
                "normalization_stats": self.normalization_stats.state_dict() if self.normalization_stats is not None else None,
            },
            path,
        )

    def load(self, path, map_location="cpu"):
        payload = torch.load(path, map_location=map_location)
        return self.load_payload(payload)

    def load_payload(self, payload: dict[str, Any]):
        self.load_state_dict(payload["state_dict"])
        self.config = payload.get("config", {})
        self.data_spec = payload.get("data_spec", {})
        backend = payload.get("backend", {})
        for key, value in backend.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.uses_normalization = bool(payload.get("uses_normalization", False))
        self.normalization_stats = NormalizationStats.from_state_dict(payload.get("normalization_stats"))
        return self

    def parameter_count(self) -> int:
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
    legacy = str(config.get("official_backend", "auto")).lower()
    if legacy in {"local", "none", "adapted"}:
        return "adapted"
    if legacy == "official":
        return "official"
    if legacy in {"neuraloperator", "deepxde", "recfno", "senseiver", "pc_bnn", "ifno"}:
        return "official"
    return legacy or "auto"


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
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": None,
        "best_val_loss": None,
        "normalize": model.uses_normalization,
        "normalization_stats": model.normalization_stats.json_summary() if model.normalization_stats is not None else None,
    }
    best_state = None
    best_val = None
    cumulative_train_time = 0.0
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        total = 0.0
        count = 0
        train_samples = 0
        for step, batch in enumerate(train_loader):
            if max_steps is not None and step >= int(max_steps):
                break
            batch = _to_device_batch(batch, device)
            train_samples += int(batch.input_fields.shape[0])
            if model.uses_normalization and model.normalization_stats is not None:
                batch = normalize_batch_input_target(batch, model.normalization_stats)
            target = batch.target_fields
            opt.zero_grad(set_to_none=True)
            pred = model.predict(batch)
            loss = loss_fn(pred, target)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            count += 1
            if log_interval > 0 and (step + 1) % log_interval == 0:
                running = total / max(count, 1)
                print(
                    f"[fit step] baseline={model.name} pde={model.data_spec.get('pde', '')} "
                    f"task={model.data_spec.get('task', '')} epoch={epoch + 1}/{epochs} "
                    f"step={step + 1}/{_safe_len(train_loader)} batch_loss={float(loss.detach().cpu()):.6g} "
                    f"running_train_loss={running:.6g} device={device}{_cuda_mem_text(device)}",
                    file=sys.stderr,
                    flush=True,
                )
        history["train_loss"].append(total / max(count, 1))
        val_loss = None
        val_count = 0
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
                    val_total += float(F.mse_loss(pred, batch.target_fields).detach().cpu())
                    val_count += 1
            val_loss = val_total / max(val_count, 1)
            history["val_loss"].append(val_loss)
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                history["best_epoch"] = epoch + 1
                history["best_val_loss"] = val_loss
                best_state = snapshot_state_dict(model)
        epoch_time = time.perf_counter() - epoch_start
        cumulative_train_time += epoch_time
        samples_per_sec = train_samples / max(epoch_time, 1e-12)
        _write_incremental_history(model.config, history, epoch + 1)
        print(
            f"[fit epoch] baseline={model.name} pde={model.data_spec.get('pde', '')} "
            f"task={model.data_spec.get('task', '')} epoch={epoch + 1}/{epochs} "
            f"train_loss={history['train_loss'][-1]:.6g} val_loss={_fmt_optional(val_loss)} "
            f"best_val_loss={_fmt_optional(history['best_val_loss'])} best_epoch={history['best_epoch']} "
            f"epoch_time_sec={epoch_time:.3f} cumulative_train_time_sec={cumulative_train_time:.3f} "
            f"train_steps={count} val_steps={val_count} samples_per_sec={samples_per_sec:.3f} "
            f"lr={opt.param_groups[0]['lr']:.3e} device={device}{_cuda_mem_text(device)}",
            file=sys.stderr,
            flush=True,
        )
    if best_state is not None:
        restore_state_dict(model, best_state)
    return history


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
