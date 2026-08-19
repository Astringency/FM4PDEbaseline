from __future__ import annotations

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.data_adapter import PDEBatch
from baselines.common.normalization import estimate_normalization_stats, normalize_batch_input_target

from .base import (
    BaselineModel,
    _build_lr_scheduler,
    _build_optimizer,
    _compact_float_list,
    _cuda_mem_text,
    _fmt_optional,
    _raise_if_nonfinite_loss,
    _raise_if_nonfinite_scalar,
    _require_exact_shape,
    _safe_len,
    _scheduler_monitor_name,
    _step_lr_scheduler,
    _set_dataset_epoch,
    _to_device_batch,
    _write_incremental_history,
    restore_state_dict,
    snapshot_state_dict,
)
from .ifno_official_aligned import OfficialAlignedIFNO2d
from .official import (
    OfficialImportError,
    get_ifno_official_aligned_status,
    get_ifno_official_status,
    official_source_info,
    requested_implementation_mode,
)
from .shared import SpectralConv2d, grid_channels


class _CouplingBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int) -> None:
        super().__init__()
        half = width // 2
        self.s12 = SpectralConv2d(half, half, modes1, modes2)
        self.w12 = nn.Conv2d(half, half, 1)
        self.s21 = SpectralConv2d(half, half, modes1, modes2)
        self.w21 = nn.Conv2d(half, half, 1)

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=1)
        if not inverse:
            scale2 = F.softplus(self.s12(x2) + self.w12(x2)) + 1e-3
            y1 = x1 * scale2
            scale1 = F.softplus(self.s21(y1) + self.w21(y1)) + 1e-3
            y2 = x2 * scale1
            return torch.cat([y1, y2], dim=1)
        scale1 = F.softplus(self.s21(x1) + self.w21(x1)) + 1e-3
        y2 = x2 / scale1
        scale2 = F.softplus(self.s12(y2) + self.w12(y2)) + 1e-3
        y1 = x1 / scale2
        return torch.cat([y1, y2], dim=1)


class IFNOBaseline(BaselineModel):
    name = "ifno"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        width = int(self.config.get("width", 32))
        if width % 2:
            width += 1
        modes1 = int(self.config.get("modes1", 12))
        modes2 = int(self.config.get("modes2", 12))
        layers = int(self.config.get("layers", 3))
        beta = float(self.config.get("beta", 2.0))
        padding = int(self.config.get("padding", 0))
        backend = str(self.config.get("official_backend", "auto")).lower()
        implementation_mode = requested_implementation_mode(self.config)
        self.input_channels = int(data_spec["input_channels"])
        self.target_channels = int(data_spec["target_channels"])
        requested_local = backend in {"local", "none"} or implementation_mode == "adapted"
        if implementation_mode == "official" and not requested_local:
            get_ifno_official_status()
            raise OfficialImportError("direct official iFNO model adapter is not implemented after import status validation")
        self.official_aligned = (
            not requested_local
            and implementation_mode in {"official_or_skip", "official_aligned", "official_architecture", "auto"}
            and backend in {"auto", "ifno", "official"}
        )
        if self.official_aligned:
            direct_import_success = False
            direct_import_warning = ""
            if implementation_mode in {"official_or_skip", "auto"}:
                try:
                    get_ifno_official_status()
                    direct_import_success = True
                except OfficialImportError as exc:
                    direct_import_warning = f"direct official iFNO import unavailable; using official-aligned reimplementation: {exc}"
            get_ifno_official_aligned_status()
            self.operator = OfficialAlignedIFNO2d(
                self.input_channels,
                self.target_channels,
                width=width,
                modes1=modes1,
                modes2=modes2,
                layers=layers,
                beta=beta,
                padding=padding,
                rank=int(self.config.get("rank", 24)),
                vae_hidden_dims=self.config.get("vae_hidden_dims"),
                vae_resolution=int(self.config.get("vae_resolution", 64)),
            )
            self.set_backend(
                "ifno_official_aligned",
                "ifno",
                fallback_used=False,
                warning=direct_import_warning,
                implementation_mode_effective="official_aligned",
                implementation_source="official_ifno_architecture_and_training_with_fm4pde_task_adapter",
                official_import_success=False,
                official_reimplementation_success=True,
                official_alignment_level="algorithm_training",
                official_alignment_notes=(
                    "Retains the bidirectional invertible coupling, official VanillaVAE topology, posterior-mean inverse "
                    "inference, and iFNO/VAE/joint three-stage AdamW training behind the FM4PDE data adapter."
                ),
                adapter_status="official_training_ifno_task_adapter",
                **official_source_info("ifno"),
            )
            return self
        self.lift_x = nn.Conv2d(self.input_channels + 2, width, 1)
        self.lift_y = nn.Conv2d(self.target_channels + 2, width, 1)
        self.blocks = nn.ModuleList([_CouplingBlock(width, modes1, modes2) for _ in range(layers)])
        self.proj_y = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, self.target_channels, 1))
        self.proj_x = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, self.input_channels, 1))
        self.set_backend(
            "local_ifno_simplified",
            "local",
            fallback_used=False,
            warning="explicit local/debug iFNO path; not paper main-table eligible",
            implementation_mode_effective="adapted",
            implementation_source="local_simplified_ifno",
            official_import_success=False,
            official_reimplementation_success=False,
            official_alignment_level="local",
            official_alignment_notes="Simplified local/debug iFNO-style coupling, not an official architecture claim.",
            adapter_status="local_debug_ifno",
            **official_source_info("ifno"),
        )
        return self

    def fit(self, train_loader, val_loader=None):
        if getattr(self, "official_aligned", False) and str(
            self.config.get("training_protocol", "official_three_stage")
        ).lower() == "official_three_stage":
            return self._fit_official_three_stage(train_loader, val_loader)
        device = torch.device(self.config.get("device", "cpu"))
        epochs = int(self.config.get("epochs", 1))
        lr = float(self.config.get("lr", 1e-3))
        max_steps = self.config.get("max_steps")
        max_val_steps = self.config.get("max_val_steps")
        log_interval = int(self.config.get("log_interval") or 0)
        cycle_weight = float(self.config.get("cycle_weight", 0.1))
        normalize = bool(self.config.get("normalize", False))
        if normalize and self.normalization_stats is None:
            max_norm_batches = self.config.get("normalization_max_batches")
            scope = "entire train_loader" if max_norm_batches is None else f"max_batches={max_norm_batches}"
            print(
                f"[normalization] baseline=ifno pde={self.data_spec.get('pde', '')} "
                f"task={self.data_spec.get('task', '')} start scope={scope}",
                file=sys.stderr,
                flush=True,
            )
            self.normalization_stats = estimate_normalization_stats(
                train_loader,
                max_batches=max_norm_batches,
                eps=float(self.config.get("normalization_eps", 1e-6)),
            )
            stats = self.normalization_stats.json_summary()
            print(
                f"[normalization] baseline=ifno pde={self.data_spec.get('pde', '')} "
                f"task={self.data_spec.get('task', '')} done num_batches={stats['num_batches']} "
                f"num_samples={stats['num_samples']} input_mean={_compact_float_list(stats['input_mean'])} "
                f"input_std={_compact_float_list(stats['input_std'])} target_mean={_compact_float_list(stats['target_mean'])} "
                f"target_std={_compact_float_list(stats['target_std'])}",
                file=sys.stderr,
                flush=True,
            )
        self.uses_normalization = bool(normalize and self.normalization_stats is not None)
        self.to(device)
        opt = _build_optimizer(self, self.config, lr)
        scheduler = _build_lr_scheduler(opt, self.config, epochs)
        monitor_name = _scheduler_monitor_name(self.config, val_loader)
        early_stopping = bool(self.config.get("early_stopping", False))
        early_stopping_patience = int(self.config.get("early_stopping_patience", 20))
        early_stopping_min_delta = float(self.config.get("early_stopping_min_delta", 1e-4))
        min_epochs = int(self.config.get("min_epochs", 1))
        restore_best = bool(self.config.get("restore_best", True))
        grad_clip_norm = self.config.get("grad_clip_norm")
        if grad_clip_norm is not None:
            grad_clip_norm = float(grad_clip_norm)
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
            "normalize": self.uses_normalization,
            "normalization_stats": self.normalization_stats.json_summary() if self.normalization_stats is not None else None,
        }
        best_val = None
        best_train = None
        best_monitor = None
        no_improve_epochs = 0
        best_state = None
        for epoch in range(epochs):
            _set_dataset_epoch(train_loader, epoch)
            if val_loader is not None:
                _set_dataset_epoch(val_loader, 0)
            epoch_start = time.perf_counter()
            total = 0.0
            count = 0
            train_samples = 0
            self.train()
            for step, batch in enumerate(train_loader):
                if max_steps is not None and step >= int(max_steps):
                    break
                batch = _to_device_batch(batch, device)
                train_samples += int(batch.input_fields.shape[0])
                if self.uses_normalization and self.normalization_stats is not None:
                    batch = normalize_batch_input_target(batch, self.normalization_stats)
                opt.zero_grad(set_to_none=True)
                loss = _ifno_training_loss(self, batch, cycle_weight)
                _raise_if_nonfinite_loss(loss, self, epoch + 1, step + 1, "train_loss")
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
                opt.step()
                total += float(loss.detach().cpu())
                count += 1
                if log_interval > 0 and (step + 1) % log_interval == 0:
                    running = total / max(count, 1)
                    print(
                        f"[fit step] baseline=ifno pde={self.data_spec.get('pde', '')} "
                        f"task={self.data_spec.get('task', '')} epoch={epoch + 1}/{epochs} "
                        f"step={step + 1}/{_safe_len(train_loader)} batch_loss={float(loss.detach().cpu()):.6g} "
                        f"running_train_loss={running:.6g} device={device}{_cuda_mem_text(device)}",
                        file=sys.stderr,
                        flush=True,
                    )
            train_loss = total / max(count, 1)
            _raise_if_nonfinite_scalar(train_loss, self, epoch + 1, count, "train_loss")
            history["train_loss"].append(train_loss)
            if val_loader is None and (best_train is None or train_loss < best_train):
                best_train = train_loss
                history["best_epoch"] = epoch + 1
                best_state = snapshot_state_dict(self)
            val_loss = None
            val_count = 0
            if val_loader is not None:
                self.eval()
                val_total = 0.0
                with torch.no_grad():
                    for step, batch in enumerate(val_loader):
                        if max_val_steps is not None and step >= int(max_val_steps):
                            break
                        batch = _to_device_batch(batch, device)
                        if self.uses_normalization and self.normalization_stats is not None:
                            batch = normalize_batch_input_target(batch, self.normalization_stats)
                        loss = _ifno_training_loss(self, batch, cycle_weight)
                        _raise_if_nonfinite_loss(loss, self, epoch + 1, step + 1, "val_loss")
                        val_total += float(loss.detach().cpu())
                        val_count += 1
                val_loss = val_total / max(val_count, 1)
                _raise_if_nonfinite_scalar(val_loss, self, epoch + 1, val_count, "val_loss")
                history["val_loss"].append(val_loss)
                if best_val is None or val_loss < best_val:
                    best_val = val_loss
                    history["best_epoch"] = epoch + 1
                    history["best_val_loss"] = val_loss
                    best_state = snapshot_state_dict(self)
            monitor_loss = train_loss if monitor_name == "train_loss" else val_loss
            if monitor_loss is None:
                monitor_name = "train_loss"
                history["monitor_name"] = monitor_name
                monitor_loss = train_loss
            improved = best_monitor is None or float(monitor_loss) < float(best_monitor) - early_stopping_min_delta
            if improved:
                best_monitor = float(monitor_loss)
                history["best_monitor_loss"] = best_monitor
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            old_lr = float(opt.param_groups[0]["lr"])
            _step_lr_scheduler(scheduler, monitor_loss)
            new_lr = float(opt.param_groups[0]["lr"])
            history["lr_history"].append(new_lr)
            if old_lr != new_lr:
                print(
                    f"[lr scheduler] baseline=ifno pde={self.data_spec.get('pde', '')} "
                    f"task={self.data_spec.get('task', '')} epoch={epoch + 1}/{epochs} "
                    f"old_lr={old_lr:.6g} new_lr={new_lr:.6g} monitor={monitor_name} monitor_loss={float(monitor_loss):.6g}",
                    file=sys.stderr,
                    flush=True,
                )
            epoch_time = time.perf_counter() - epoch_start
            samples_per_sec = train_samples / max(epoch_time, 1e-12)
            should_stop = early_stopping and epoch + 1 >= min_epochs and no_improve_epochs >= early_stopping_patience
            if should_stop:
                history["early_stopped"] = True
                history["stop_epoch"] = epoch + 1
                history["stop_reason"] = (
                    f"no improvement in {monitor_name} for {no_improve_epochs} epochs "
                    f"(patience={early_stopping_patience}, min_delta={early_stopping_min_delta})"
                )
            _write_incremental_history(self.config, history, epoch + 1)
            print(
                f"[fit epoch] baseline=ifno pde={self.data_spec.get('pde', '')} "
                f"task={self.data_spec.get('task', '')} epoch={epoch + 1}/{epochs} "
                f"train_loss={history['train_loss'][-1]:.6g} val_loss={_fmt_optional(val_loss)} "
                f"best_val_loss={_fmt_optional(history['best_val_loss'])} best_epoch={history['best_epoch']} "
                f"monitor={monitor_name} monitor_loss={float(monitor_loss):.6g} "
                f"no_improve_epochs={no_improve_epochs} early_stopping_patience={early_stopping_patience} "
                f"epoch_time_sec={epoch_time:.3f} train_steps={count} val_steps={val_count} "
                f"samples_per_sec={samples_per_sec:.3f} lr={opt.param_groups[0]['lr']:.3e} "
                f"device={device}{_cuda_mem_text(device)}",
                file=sys.stderr,
                flush=True,
            )
            if should_stop:
                break
        if not history["early_stopped"]:
            history["stop_epoch"] = len(history["train_loss"])
        if best_state is not None and restore_best:
            restore_state_dict(self, best_state)
        return history

    def _fit_official_three_stage(self, train_loader, val_loader=None):
        device = torch.device(self.config.get("device", "cpu"))
        normalize = bool(self.config.get("normalize", False))
        if normalize and self.normalization_stats is None:
            self.normalization_stats = estimate_normalization_stats(
                train_loader,
                max_batches=self.config.get("normalization_max_batches"),
                eps=float(self.config.get("normalization_eps", 1e-6)),
            )
        self.uses_normalization = bool(normalize and self.normalization_stats is not None)
        self.to(device)
        stage_epochs = {
            "ifno_pretrain": max(int(self.config.get("ifno_pretrain_epochs", self.config.get("epochs_ifno", 1))), 0),
            "vae_pretrain": max(int(self.config.get("vae_pretrain_epochs", self.config.get("epochs_vae", 1))), 0),
            "joint_train": max(int(self.config.get("joint_epochs", self.config.get("epochs", 1))), 0),
        }
        max_steps = self.config.get("max_steps")
        grad_clip = float(self.config.get("grad_clip_norm", 10.0) or 10.0)
        reconstruction_weight = float(self.config.get("reconstruction_weight", 1.0))
        kl_weight = float(self.config.get("kl_weight", self.config.get("kl", 0.01)))
        pretrain_optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(self.config.get("lr_ifno", self.config.get("lr", 2.5e-4)))
        )
        vae_optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(self.config.get("lr_vae", 1e-3))
        )
        forward_optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(self.config.get("lr_forward", 5e-6))
        )
        backward_optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(self.config.get("lr_backward", 5e-6))
        )
        history = {
            "training_protocol": "official_three_stage",
            "stage_epochs": stage_epochs,
            "stage_losses": {name: [] for name in stage_epochs},
            "train_loss": [],
            "val_loss": [],
            "best_epoch": None,
            "best_val_loss": None,
            "early_stopped": False,
            "stop_epoch": None,
            "stop_reason": "",
            "lr_history": [],
            "normalize": self.uses_normalization,
            "normalization_stats": self.normalization_stats.json_summary() if self.normalization_stats is not None else None,
        }
        completed_epochs = 0

        for epoch in range(stage_epochs["ifno_pretrain"]):
            self.train()
            total = count = 0
            for step, raw_batch in enumerate(train_loader):
                if max_steps is not None and step >= int(max_steps):
                    break
                batch = self._prepare_ifno_batch(raw_batch, device)
                x, y = _physical_pair(batch)
                pretrain_optimizer.zero_grad(set_to_none=True)
                y_pred, recon_x = self.operator.forward_map(x)
                forward_loss = _relative_l2_loss(y_pred, y) + reconstruction_weight * recon_x
                forward_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                pretrain_optimizer.step()
                pretrain_optimizer.zero_grad(set_to_none=True)
                x_pred, recon_y = self.operator.inverse_map(y)
                backward_loss = _relative_l2_loss(x_pred, x) + reconstruction_weight * recon_y
                backward_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                pretrain_optimizer.step()
                total += float((forward_loss + backward_loss).detach().cpu())
                count += 1
            history["stage_losses"]["ifno_pretrain"].append(total / max(count, 1))
            completed_epochs += 1
            history["completed_stage"] = "ifno_pretrain"
            _write_incremental_history(self.config, history, completed_epochs)

        for epoch in range(stage_epochs["vae_pretrain"]):
            self.train()
            total = count = 0
            for step, raw_batch in enumerate(train_loader):
                if max_steps is not None and step >= int(max_steps):
                    break
                batch = self._prepare_ifno_batch(raw_batch, device)
                x, _ = _physical_pair(batch)
                vae_optimizer.zero_grad(set_to_none=True)
                reconstructed, mu, log_var = self.operator.vae_reconstruct(x, sample=True)
                loss = _relative_l2_loss(reconstructed, x) + kl_weight * _vae_kl(mu, log_var)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                vae_optimizer.step()
                total += float(loss.detach().cpu())
                count += 1
            history["stage_losses"]["vae_pretrain"].append(total / max(count, 1))
            completed_epochs += 1
            history["completed_stage"] = "vae_pretrain"
            _write_incremental_history(self.config, history, completed_epochs)

        best_monitor = None
        best_state = None
        no_improve = 0
        patience = max(int(self.config.get("early_stopping_patience", 20)), 1)
        min_delta = float(self.config.get("early_stopping_min_delta", 1e-4))
        min_epochs = max(int(self.config.get("min_epochs", 1)), 1)
        for epoch in range(stage_epochs["joint_train"]):
            self.train()
            total = count = 0
            for step, raw_batch in enumerate(train_loader):
                if max_steps is not None and step >= int(max_steps):
                    break
                batch = self._prepare_ifno_batch(raw_batch, device)
                x, y = _physical_pair(batch)
                forward_optimizer.zero_grad(set_to_none=True)
                y_pred, _ = self.operator.forward_map(x)
                forward_loss = _relative_l2_loss(y_pred, y)
                forward_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                forward_optimizer.step()
                backward_optimizer.zero_grad(set_to_none=True)
                raw_inverse, _ = self.operator.inverse_map(y)
                filtered_inverse, mu, log_var = self.operator.vae_reconstruct(raw_inverse, sample=True)
                backward_loss = _relative_l2_loss(filtered_inverse, x) + kl_weight * _vae_kl(mu, log_var)
                backward_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                backward_optimizer.step()
                total += float((forward_loss + backward_loss).detach().cpu())
                count += 1
            train_loss = total / max(count, 1)
            val_loss = self._official_ifno_eval_loss(val_loader, device, max_steps) if val_loader is not None else None
            monitor = train_loss if val_loss is None else val_loss
            history["stage_losses"]["joint_train"].append(train_loss)
            history["train_loss"].append(train_loss)
            if val_loss is not None:
                history["val_loss"].append(val_loss)
            history["lr_history"].append(float(forward_optimizer.param_groups[0]["lr"]))
            completed_epochs += 1
            history["completed_stage"] = "joint_train"
            if best_monitor is None or monitor < best_monitor - min_delta:
                best_monitor = monitor
                best_state = snapshot_state_dict(self)
                history["best_epoch"] = epoch + 1
                history["best_val_loss"] = val_loss
                no_improve = 0
            else:
                no_improve += 1
            if bool(self.config.get("early_stopping", False)) and epoch + 1 >= min_epochs and no_improve >= patience:
                history["early_stopped"] = True
                history["stop_epoch"] = epoch + 1
                history["stop_reason"] = f"joint loss plateau for {no_improve} epochs"
                _write_incremental_history(self.config, history, completed_epochs)
                break
            _write_incremental_history(self.config, history, completed_epochs)
        if history["stop_epoch"] is None:
            history["stop_epoch"] = len(history["train_loss"])
        if best_state is not None and bool(self.config.get("restore_best", True)):
            restore_state_dict(self, best_state)
        return history

    def _prepare_ifno_batch(self, batch: PDEBatch, device: torch.device) -> PDEBatch:
        prepared = _to_device_batch(batch, device)
        if self.uses_normalization and self.normalization_stats is not None:
            prepared = normalize_batch_input_target(prepared, self.normalization_stats)
        return prepared

    def _official_ifno_eval_loss(self, loader, device: torch.device, max_steps) -> float:
        self.eval()
        total = count = 0
        with torch.no_grad():
            for step, raw_batch in enumerate(loader):
                if max_steps is not None and step >= int(max_steps):
                    break
                batch = self._prepare_ifno_batch(raw_batch, device)
                x, y = _physical_pair(batch)
                forward, _ = self.operator.forward_map(x)
                inverse, _ = self.operator.inverse_map(y)
                inverse, _, _ = self.operator.vae_reconstruct(inverse, sample=False)
                total += float((_relative_l2_loss(forward, y) + _relative_l2_loss(inverse, x)).cpu())
                count += 1
        return total / max(count, 1)

    def predict(self, batch: PDEBatch):
        if batch.task.startswith("sparse"):
            raise NotImplementedError("iFNO sparse reconstruction/inverse is unsupported without an official sparse inference adapter")
        if batch.task == "inverse":
            return self._inverse_map(batch.input_fields)
        x = batch.input_fields
        return self._forward_map(x)

    def _forward_map(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "official_aligned", False):
            pred, _ = self.operator.forward_map(x)
            return pred
        z = self.lift_x(torch.cat([x, grid_channels(x)], dim=1))
        for block in self.blocks:
            z = block(z, inverse=False)
        return self.proj_y(z)

    def _inverse_map(self, y: torch.Tensor) -> torch.Tensor:
        if getattr(self, "official_aligned", False):
            pred, _ = self.operator.inverse_map(y)
            pred, _, _ = self.operator.vae_reconstruct(pred, sample=False)
            return pred
        z = self.lift_y(torch.cat([y, grid_channels(y)], dim=1))
        for block in reversed(self.blocks):
            z = block(z, inverse=True)
        return self.proj_x(z)

    def _forward_map_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "official_aligned", False):
            return self.operator.forward_map(x)
        return self._forward_map(x), torch.tensor(0.0, device=x.device, dtype=x.dtype)

    def _inverse_map_with_aux(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "official_aligned", False):
            return self.operator.inverse_map(y)
        return self._inverse_map(y), torch.tensor(0.0, device=y.device, dtype=y.dtype)


def _physical_pair(batch: PDEBatch) -> tuple[torch.Tensor, torch.Tensor]:
    if batch.task in {"inverse", "sparse_inverse"}:
        return batch.target_fields, batch.input_fields
    return batch.input_fields, batch.target_fields


def _ifno_training_loss(model: IFNOBaseline, batch: PDEBatch, cycle_weight: float) -> torch.Tensor:
    x, y = _physical_pair(batch)
    y_pred, recon_x = model._forward_map_with_aux(x)
    x_pred, recon_y = model._inverse_map_with_aux(y)
    _require_exact_shape(y_pred, y, model.name, "bidirectional-forward")
    _require_exact_shape(x_pred, x, model.name, "bidirectional-inverse")
    loss = F.mse_loss(y_pred, y) + F.mse_loss(x_pred, x)
    if cycle_weight:
        cycle_x = model._inverse_map(y_pred)
        cycle_y = model._forward_map(x_pred)
        _require_exact_shape(cycle_x, x, model.name, "cycle-input")
        _require_exact_shape(cycle_y, y, model.name, "cycle-target")
        loss = loss + cycle_weight * (F.mse_loss(cycle_x, x) + F.mse_loss(cycle_y, y))
    return loss + float(model.config.get("reconstruction_weight", 0.1)) * (recon_x + recon_y)


def _relative_l2_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = torch.linalg.vector_norm((prediction - target).reshape(prediction.shape[0], -1), dim=1)
    denominator = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(1e-12)
    return (difference / denominator).sum()


def _vae_kl(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return torch.mean(-0.5 * torch.sum(1.0 + log_var - mu.square() - log_var.exp(), dim=1))
