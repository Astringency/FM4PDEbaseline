from __future__ import annotations

import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.data_adapter import PDEBatch, pde_collate, slice_pde_batch

from .base import BaselineModel, record_optimization_status
from .official import (
    OfficialImportError,
    get_vivid_official_architecture_status,
    get_vivid_official_status,
    official_source_info,
    requested_implementation_mode,
)
from .variational import observation_residuals, periodic_balgovind_quadratic_2d, scipy_lbfgsb
from .var4d import _synchronize, _synchronized_time, _validate_burgers_trajectory


class KerasSameConv2d(nn.Module):
    """PyTorch Conv2d with TensorFlow/Keras SAME padding for an even kernel."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        total_padding = int(kernel_size) - 1
        self.padding_before = total_padding // 2
        self.padding_after = total_padding - self.padding_before
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=int(kernel_size), padding=0)
        # Keras Conv2D defaults to Glorot-uniform kernels and zero bias.
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(
            x,
            (self.padding_before, self.padding_after, self.padding_before, self.padding_after),
        )
        return self.conv(x)


class VIVIDVCNN(nn.Module):
    """Exact VCNN layer recipe from ``offical/VIVID/VCNN_training.py``."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        for _ in range(6):
            layers.append(KerasSameConv2d(in_channels, 48, 8))
            in_channels = 48
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = KerasSameConv2d(48, 1, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"VIVID VCNN expects [B,1,H,W], got {tuple(x.shape)}")
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)


class VIVIDBaseline(BaselineModel):
    """Official-core VIVID adapted to a Burgers time-space state field.

    VIVID itself is a 3D-Var method, not a 4D-Var method. For the Burgers
    reconstruction task the complete ``T x X`` solution is therefore treated
    as the two-dimensional state in the original three-term VIVID objective.
    """

    name = "vivid"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        pde = str(data_spec.get("pde", "")).lower()
        if pde != "burger":
            raise ValueError("The official-core VIVID adapter is scoped only to Burgers sparse trajectory reconstruction")

        implementation_mode = requested_implementation_mode(self.config)
        if implementation_mode == "official":
            # The vendored Keras/ADAO scripts execute global experiment code
            # at import time and cannot be called as a safe library backend.
            get_vivid_official_status()
            raise OfficialImportError("Vendored VIVID scripts are not directly importable")
        get_vivid_official_architecture_status()

        self.inverse_operator = VIVIDVCNN()
        self.register_buffer("_inverse_operator_trained_flag", torch.tensor(False, dtype=torch.bool), persistent=True)
        self.optimized_numel = _trajectory_state_numel(data_spec)
        self.set_backend(
            "vivid_vcnn_3dvar_burgers",
            "vivid",
            fallback_used=False,
            implementation_mode_effective="official_architecture",
            implementation_source="pytorch_port_of_vendored_vivid_vcnn_and_three_term_3dvar",
            official_import_success=False,
            official_reimplementation_success=True,
            official_alignment_level="architecture_objective_training",
            official_alignment_notes=(
                "Preserves the vendored VIVID VCNN architecture, Glorot initialization, Adam/MSE recipe, three-term "
                "background/inverse/observation objective, L-BFGS-B optimizer, covariance scales, and budgets. "
                "Burgers adapts the official 2-D spatial state to a 2-D time-space state and replaces the shallow-water "
                "observation operator with direct sparse solution sampling. The dense radial Balgovind covariance uses "
                "a matrix-free circulant embedding at 128x128."
            ),
            adapter_status="official_architecture_vivid_burgers_task_adapter",
            **official_source_info("vivid"),
        )
        return self

    @property
    def inverse_operator_trained(self) -> bool:
        return bool(self._inverse_operator_trained_flag.item())

    def parameter_count(self) -> int:
        network = sum(
            parameter.numel() * (2 if parameter.is_complex() else 1)
            for parameter in self.inverse_operator.parameters()
            if parameter.requires_grad
        )
        return int(self.optimized_numel + network)

    def parameter_storage_count(self) -> int:
        network = sum(parameter.numel() for parameter in self.inverse_operator.parameters() if parameter.requires_grad)
        return int(self.optimized_numel + network)

    def fit(self, train_loader, val_loader=None):
        if not bool(self.config.get("train_inverse_operator", True)):
            return {
                "status": "inverse_operator_training_disabled",
                "offline_training": False,
                "inverse_operator_trained": self.inverse_operator_trained,
            }

        device = torch.device(self.config.get("device", "cpu"))
        epochs = int(self.config.get("epochs", 20))
        learning_rate = float(self.config.get("inverse_learning_rate", 1e-4))
        micro_batch_size = max(int(self.config.get("vcnn_micro_batch_size", 4)), 1)
        max_steps = self.config.get("max_steps")
        max_val_steps = self.config.get("max_val_steps")
        early_stopping = bool(self.config.get("early_stopping", True))
        early_stopping_patience = max(int(self.config.get("early_stopping_patience", 20)), 1)
        early_stopping_min_delta = float(self.config.get("early_stopping_min_delta", 1e-4))
        min_epochs = max(int(self.config.get("min_epochs", 1)), 1)
        self.to(device)
        optimizer = torch.optim.Adam(
            self.inverse_operator.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=float(self.config.get("inverse_adam_epsilon", 1e-7)),
        )
        history: dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": None,
            "best_val_loss": None,
            "early_stopped": False,
            "stop_epoch": epochs,
            "stop_reason": "configured_fixed_epoch_budget_completed",
            "optimizer": "adam",
            "training_loss": "mse",
            "learning_rate": learning_rate,
            "requested_epochs": epochs,
            "effective_batch_size": int(getattr(train_loader, "batch_size", 0) or 0),
            "micro_batch_size": micro_batch_size,
            "validation_source": "unified_nonoverlapping_validation_split",
            "official_reference_epochs": 20,
            "official_reference_batch_size": 64,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
        }
        best_val: float | None = None
        best_monitor: float | None = None
        no_improve_epochs = 0

        for epoch in range(epochs):
            _set_dataset_epoch(train_loader, epoch)
            _set_dataset_epoch(val_loader, 0)
            self.inverse_operator.train()
            train_sum = 0.0
            train_values = 0
            train_batches = 0
            epoch_start = time.perf_counter()
            for step, batch in enumerate(train_loader):
                if max_steps is not None and step >= int(max_steps):
                    break
                batch = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                batch_values = int(batch.target_fields.numel())
                for chunk in _batch_chunks(batch, micro_batch_size):
                    prediction = self.inverse_operator(_voronoi_input(chunk))
                    loss_sum = F.mse_loss(prediction, chunk.target_fields, reduction="sum")
                    (loss_sum / max(batch_values, 1)).backward()
                    train_sum += float(loss_sum.detach().cpu())
                    train_values += int(chunk.target_fields.numel())
                optimizer.step()
                train_batches += 1
            train_loss = train_sum / max(train_values, 1)
            history["train_loss"].append(train_loss)

            val_loss = None
            if val_loader is not None:
                self.inverse_operator.eval()
                val_sum = 0.0
                val_values = 0
                with torch.no_grad():
                    for step, batch in enumerate(val_loader):
                        if max_val_steps is not None and step >= int(max_val_steps):
                            break
                        batch = _move_batch(batch, device)
                        for chunk in _batch_chunks(batch, micro_batch_size):
                            prediction = self.inverse_operator(_voronoi_input(chunk))
                            val_sum += float(F.mse_loss(prediction, chunk.target_fields, reduction="sum").detach().cpu())
                            val_values += int(chunk.target_fields.numel())
                val_loss = val_sum / max(val_values, 1)
                history["val_loss"].append(val_loss)
                if best_val is None or val_loss < best_val:
                    best_val = val_loss
                    history["best_epoch"] = epoch + 1
                    history["best_val_loss"] = val_loss

            monitor = train_loss if val_loss is None else val_loss
            if best_monitor is None or monitor < best_monitor - early_stopping_min_delta:
                best_monitor = monitor
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            should_stop = (
                early_stopping
                and epoch + 1 >= min_epochs
                and no_improve_epochs >= early_stopping_patience
            )

            print(
                f"[fit epoch] baseline=vivid pde=burger epoch={epoch + 1}/{epochs} "
                f"train_loss={train_loss:.6g} val_loss={_format_optional(val_loss)} "
                f"train_steps={train_batches} effective_batch_size={history['effective_batch_size']} "
                f"micro_batch_size={micro_batch_size} epoch_time_sec={time.perf_counter() - epoch_start:.3f}",
                file=sys.stderr,
                flush=True,
            )
            if should_stop:
                history["early_stopped"] = True
                history["stop_epoch"] = epoch + 1
                history["stop_reason"] = (
                    f"no improvement for {no_improve_epochs} epochs "
                    f"(patience={early_stopping_patience}, min_delta={early_stopping_min_delta})"
                )
                break

        self._inverse_operator_trained_flag.fill_(True)
        history["completed_epochs"] = len(history["train_loss"])
        if not history["early_stopped"]:
            history["stop_epoch"] = history["completed_epochs"]
        history["inverse_operator_trained"] = True
        return history

    def predict(self, batch: PDEBatch):
        start = _synchronized_time(batch.input_fields)
        _validate_burgers_trajectory(batch, "VIVID")
        if not self.inverse_operator_trained:
            raise RuntimeError("VIVID inference requires a trained or checkpoint-loaded VCNN inverse operator")

        refine_steps = int(self.config.get("refine_steps", 1000))
        tolerance = float(self.config.get("cost_decrement_tolerance", 1e-6))
        gradient_tolerance = float(self.config.get("gradient_tolerance", 1e-8))
        background_variance = float(self.config.get("background_variance", 1000.0))
        background_length = float(self.config.get("background_correlation_length", 5.0))
        inverse_variance = float(self.config.get("inverse_operator_variance", 100.0))
        observation_variance = float(self.config.get("observation_variance", 1.0))

        learned_fields = self._predict_inverse_fields(batch)
        predictions: list[torch.Tensor] = []
        statuses: list[dict[str, Any]] = []
        for item in range(batch.target_fields.shape[0]):
            sample = slice_pde_batch(batch, item)
            background = _trajectory_background(sample)
            inverse_state = learned_fields[item : item + 1].detach()

            def objective(state: torch.Tensor) -> torch.Tensor:
                background_term = periodic_balgovind_quadratic_2d(
                    state - background,
                    variance=background_variance,
                    length_scale=background_length,
                )
                inverse_term = 0.5 * (state - inverse_state).square().sum() / max(inverse_variance, 1e-12)
                observation_term = 0.5 * observation_residuals(state, sample).square().sum() / max(
                    observation_variance, 1e-12
                )
                return background_term + inverse_term + observation_term

            optimized, status = scipy_lbfgsb(
                background,
                objective,
                max_steps=refine_steps,
                cost_decrement_tolerance=tolerance,
                gradient_tolerance=gradient_tolerance,
            )
            status.update(
                {
                    "sample_index": item,
                    "optimization_variable": "complete_time_space_state",
                    "optimized_numel": int(optimized.numel()),
                    "objective": "VIVID_Jb_plus_Jp_plus_Jo",
                }
            )
            predictions.append(optimized.detach())
            statuses.append(status)

        record_optimization_status(batch, statuses)
        batch.metadata.update(
            {
                "assimilation_mode": "vivid_3dvar_time_space_state",
                "assimilation_background_source": "voronoi_grid_from_sparse_observations",
                "assimilation_uses_hidden_truth": False,
                "inverse_observation_operator_used": True,
                "inverse_observation_operator": "official_vivid_vcnn_architecture",
                "learned_state_shape": tuple(learned_fields.shape),
                "optimization_variable": "complete_time_space_state",
                "optimized_state_shape": tuple(batch.target_fields.shape),
                "optimization_optimizer": "scipy_L-BFGS-B",
                "optimization_budget_steps_per_sample": refine_steps,
                "optimization_function_evaluations": sum(int(status["function_evaluations"]) for status in statuses),
                "vivid_objective": "Jb_background_plus_Jp_inverse_plus_Jo_observation",
                "background_covariance": "Balgovind_2d_circulant_embedding",
                "background_variance": background_variance,
                "background_correlation_length": background_length,
                "inverse_operator_variance": inverse_variance,
                "observation_variance": observation_variance,
                "official_alignment_level": self.official_alignment_level,
            }
        )
        _synchronize(batch.input_fields)
        batch.metadata["inference_optimization_time"] = time.perf_counter() - start
        return torch.cat(predictions, dim=0)

    def predict_physical(self, batch: PDEBatch):
        return self.predict(batch)

    def _predict_inverse_fields(self, batch: PDEBatch) -> torch.Tensor:
        micro_batch_size = max(int(self.config.get("vcnn_inference_batch_size", 4)), 1)
        self.inverse_operator.eval()
        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for chunk in _batch_chunks(batch, micro_batch_size):
                outputs.append(self.inverse_operator(_voronoi_input(chunk)))
        return torch.cat(outputs, dim=0)


def _trajectory_state_numel(data_spec: dict) -> int:
    target_shape = tuple(int(value) for value in data_spec.get("target_shape", ()))
    if len(target_shape) != 4:
        return 0
    return int(torch.tensor(target_shape[1:]).prod().item())


def _voronoi_input(batch: PDEBatch) -> torch.Tensor:
    field = batch.metadata.get("voronoi_grid", batch.input_fields)
    if not isinstance(field, torch.Tensor) or tuple(field.shape) != tuple(batch.target_fields.shape):
        raise ValueError(
            "VIVID VCNN requires a Voronoi-filled field matching the target shape; "
            f"expected {tuple(batch.target_fields.shape)}, got {getattr(field, 'shape', None)}"
        )
    return field.to(device=batch.target_fields.device, dtype=batch.target_fields.dtype)


def _trajectory_background(batch: PDEBatch) -> torch.Tensor:
    return _voronoi_input(batch).detach().clone()


def _batch_chunks(batch: PDEBatch, chunk_size: int):
    items = int(batch.target_fields.shape[0])
    for start in range(0, items, max(int(chunk_size), 1)):
        samples = [slice_pde_batch(batch, index) for index in range(start, min(start + chunk_size, items))]
        yield samples[0] if len(samples) == 1 else pde_collate(samples)


def _move_batch(batch: PDEBatch, device: torch.device) -> PDEBatch:
    from .base import _to_device_batch

    return _to_device_batch(batch, device)


def _set_dataset_epoch(loader, epoch: int) -> None:
    if loader is None:
        return
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "set_epoch"):
        dataset.set_epoch(int(epoch))


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"
