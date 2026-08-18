from __future__ import annotations

import time
import warnings

import torch
import torch.nn.functional as F
import torch.nn as nn

from baselines.common.data_adapter import PDEBatch
from baselines.common.metrics import NotImplementedWarning, physics_loss_metric

from .base import BaselineModel
from .official import OfficialImportError, get_deepxde_fnn_class, official_source_info, requested_implementation_mode
from .shared import NeuralField

STATIC_SPARSE_INVERSE_PDES = {"poisson", "helmholtz", "darcy", "steady_heat_conduction"}


def _synchronized_perf_counter(batch: PDEBatch) -> float:
    if batch.input_fields.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(batch.input_fields.device)
    return time.perf_counter()


class PINNSparseBaseline(BaselineModel):
    name = "pinn_sparse"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        self.coord_dim = len(tuple(data_spec["target_shape"])[2:])
        self.task = str(data_spec.get("task", ""))
        self.input_channels = int(data_spec["input_channels"])
        self.target_channels = int(data_spec["target_channels"])
        self.hidden = int(self.config.get("hidden", 64))
        self.depth = int(self.config.get("depth", 4))
        self.deepxde_fnn_cls = None
        backend = str(self.config.get("official_backend", "auto")).lower()
        implementation_mode = requested_implementation_mode(self.config)
        fallback_reason = ""
        if implementation_mode != "adapted" and backend in {"auto", "deepxde", "official"}:
            try:
                self.deepxde_fnn_cls = get_deepxde_fnn_class()
                self.set_backend(
                    "deepxde",
                    "deepxde",
                    fallback_used=False,
                    implementation_mode_effective="canonical_math",
                    implementation_source="pinn_style_deepxde_fnn_local_pde_objective",
                    official_import_success=True,
                    adapter_status="canonical_math",
                    **official_source_info("deepxde"),
                )
            except OfficialImportError as exc:
                fallback_reason = f"deepxde unavailable: {exc}"
                if backend in {"deepxde", "official"}:
                    warnings.warn(f"DeepXDE FNN unavailable, using local neural field fallback: {exc}", RuntimeWarning, stacklevel=2)
        if self.deepxde_fnn_cls is None:
            requested_local = backend in {"local", "none"} or implementation_mode == "adapted"
            self.set_backend(
                "local",
                "local" if requested_local else ("official" if backend == "official" else backend),
                fallback_used=False,
                warning=fallback_reason,
                implementation_mode_effective="canonical_math",
                implementation_source="pinn_style_local_neural_field_local_pde_objective",
                official_import_success=False,
                adapter_status="canonical_math",
            )
        return self

    def parameter_count(self) -> int:
        channel_counts = [self.target_channels]
        if self.task in {"sparse_inverse", "sparse_forward"}:
            # Both protocols optimize an unknown/source field and a solution
            # field per test sample. Report the complete optimized model, not
            # only the tensor returned to the evaluator.
            channel_counts.append(self.input_channels)
        return int(
            sum(
                parameter.numel()
                for channels in channel_counts
                for parameter in self._new_field(self.coord_dim, channels).parameters()
                if parameter.requires_grad
            )
        )

    def parameter_storage_count(self) -> int:
        return self.parameter_count()

    def fit(self, train_loader, val_loader=None):
        return {"status": "per_instance_method_no_amortized_fit"}

    def predict(self, batch: PDEBatch):
        if batch.task == "sparse_inverse":
            return self._predict_sparse_inverse(batch)
        if batch.task == "sparse_forward":
            return self._predict_sparse_forward(batch)
        if batch.task in {"sparse_solution", "sparse_reconstruction"}:
            raise RuntimeError(
                "PINN-Sparse is disabled for the sensor-only sparse_solution protocol because its PDE objective "
                "requires hidden source/coefficient/initial fields. Use a separately named equal-context protocol."
            )
        start = _synchronized_perf_counter(batch)
        preds = []
        steps = int(self.config.get("steps", 2))
        lr = float(self.config.get("lr", 1e-2))
        lam_obs = float(self.config.get("lambda_obs", 1.0))
        hidden = self.hidden
        depth = self.depth
        opt_name = str(self.config.get("optimizer", "adam")).lower()
        for item in range(batch.target_fields.shape[0]):
            coords = batch.coords[item].to(batch.target_fields.device, batch.target_fields.dtype)
            target = batch.target_fields[item : item + 1]
            model = self._new_field(coords.shape[-1], target.shape[1]).to(target.device)
            optimizer = (
                torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=steps)
                if opt_name == "lbfgs"
                else torch.optim.Adam(model.parameters(), lr=lr)
            )

            def closure():
                optimizer.zero_grad(set_to_none=True)
                pred = model(coords).T.reshape_as(target)
                loss = lam_obs * observation_loss_from_batch(pred, batch, item=item)
                meta = _single_meta(batch, item)
                meta.update(_physics_weight_metadata(self.config))
                physics_value = _select_physics_loss(physics_loss_metric(pred, batch.pde_name, meta), self.config, pred)
                if torch.isfinite(physics_value):
                    loss = loss + physics_value
                loss.backward()
                return loss

            if opt_name == "lbfgs":
                optimizer.step(closure)
            else:
                for _ in range(max(steps, 0)):
                    closure()
                    optimizer.step()
            with torch.no_grad():
                preds.append(model(coords).T.reshape_as(target))
        batch.metadata["inference_optimization_time"] = _synchronized_perf_counter(batch) - start
        return torch.cat(preds, dim=0).detach()

    def _predict_sparse_inverse(self, batch: PDEBatch):
        pde = batch.pde_name.lower()
        if pde not in STATIC_SPARSE_INVERSE_PDES:
            raise NotImplementedError(f"PINN-Sparse sparse_inverse is only enabled for static PDEs, got {batch.pde_name}")
        start = _synchronized_perf_counter(batch)
        preds = []
        steps = int(self.config.get("steps", 2))
        lr = float(self.config.get("lr", 1e-2))
        lam_obs = float(self.config.get("lambda_obs", 1.0))
        lam_reg = float(self.config.get("lambda_reg", 1e-6))
        opt_name = str(self.config.get("optimizer", "adam")).lower()
        for item in range(batch.target_fields.shape[0]):
            coords = batch.coords[item].to(batch.target_fields.device, batch.target_fields.dtype)
            unknown_target_shape = batch.target_fields[item : item + 1].shape
            solution_channels = int(batch.input_fields.shape[1])
            unknown = self._new_field(coords.shape[-1], unknown_target_shape[1]).to(batch.target_fields.device)
            solution = self._new_field(coords.shape[-1], solution_channels).to(batch.target_fields.device)
            params = list(unknown.parameters()) + list(solution.parameters())
            optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=steps) if opt_name == "lbfgs" else torch.optim.Adam(params, lr=lr)

            def closure():
                optimizer.zero_grad(set_to_none=True)
                unknown_grid = unknown(coords).T.reshape(unknown_target_shape)
                solution_grid = solution(coords).T.reshape(batch.input_fields[item : item + 1].shape)
                loss = lam_obs * sparse_inverse_observation_loss(solution_grid, batch, item=item)
                physics_value = sparse_inverse_physics_loss(unknown_grid, solution_grid, batch, item, self.config)
                if torch.isfinite(physics_value):
                    loss = loss + physics_value
                loss = loss + lam_reg * (_smoothness_reg(unknown_grid) + _smoothness_reg(solution_grid))
                loss.backward()
                return loss

            if opt_name == "lbfgs":
                optimizer.step(closure)
            else:
                for _ in range(max(steps, 0)):
                    closure()
                    optimizer.step()
            with torch.no_grad():
                preds.append(unknown(coords).T.reshape(unknown_target_shape))
        batch.metadata["inference_optimization_time"] = _synchronized_perf_counter(batch) - start
        return torch.cat(preds, dim=0).detach()

    def _predict_sparse_forward(self, batch: PDEBatch):
        pde = batch.pde_name.lower()
        if pde not in STATIC_SPARSE_INVERSE_PDES:
            raise NotImplementedError(f"PINN-Sparse sparse_forward is only enabled for static PDEs, got {batch.pde_name}")
        start = _synchronized_perf_counter(batch)
        preds = []
        steps = int(self.config.get("steps", 2))
        lr = float(self.config.get("lr", 1e-2))
        lam_obs = float(self.config.get("lambda_obs", 1.0))
        lam_reg = float(self.config.get("lambda_reg", 1e-6))
        opt_name = str(self.config.get("optimizer", "adam")).lower()
        for item in range(batch.target_fields.shape[0]):
            coords = batch.coords[item].to(batch.target_fields.device, batch.target_fields.dtype)
            unknown_target_shape = batch.input_fields[item : item + 1].shape
            solution_target_shape = batch.target_fields[item : item + 1].shape
            unknown = self._new_field(coords.shape[-1], unknown_target_shape[1]).to(batch.target_fields.device)
            solution = self._new_field(coords.shape[-1], solution_target_shape[1]).to(batch.target_fields.device)
            params = list(unknown.parameters()) + list(solution.parameters())
            optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=steps) if opt_name == "lbfgs" else torch.optim.Adam(params, lr=lr)

            def closure():
                optimizer.zero_grad(set_to_none=True)
                unknown_grid = unknown(coords).T.reshape(unknown_target_shape)
                solution_grid = solution(coords).T.reshape(solution_target_shape)
                loss = lam_obs * sparse_forward_observation_loss(unknown_grid, batch, item=item)
                physics_value = sparse_inverse_physics_loss(unknown_grid, solution_grid, batch, item, self.config)
                if torch.isfinite(physics_value):
                    loss = loss + physics_value
                loss = loss + lam_reg * (_smoothness_reg(unknown_grid) + _smoothness_reg(solution_grid))
                loss.backward()
                return loss

            if opt_name == "lbfgs":
                optimizer.step(closure)
            else:
                for _ in range(max(steps, 0)):
                    closure()
                    optimizer.step()
            with torch.no_grad():
                preds.append(solution(coords).T.reshape(solution_target_shape))
        batch.metadata["inference_optimization_time"] = _synchronized_perf_counter(batch) - start
        return torch.cat(preds, dim=0).detach()

    def _new_field(self, coord_dim: int, out_channels: int) -> nn.Module:
        if self.deepxde_fnn_cls is not None:
            layers = [coord_dim] + [self.hidden] * max(self.depth - 1, 1) + [out_channels]
            return self.deepxde_fnn_cls(
                layers,
                str(self.config.get("activation", "gelu")),
                str(self.config.get("kernel_initializer", "Glorot normal")),
            )
        return NeuralField(coord_dim, out_channels, hidden=self.hidden, depth=self.depth)


def residual_supported(pde_name: str) -> bool:
    if pde_name.lower() in {"poisson", "darcy", "helmholtz", "burger", "nsnonbounded", "reaction_diffusion", "shallow_water"}:
        return True
    warnings.warn(f"PINN residual for {pde_name} is currently an interface only.", NotImplementedWarning, stacklevel=2)
    return False


def observation_loss_from_batch(pred: torch.Tensor, batch: PDEBatch, item: int | None = None) -> torch.Tensor:
    if batch.task == "sparse_inverse":
        return sparse_inverse_observation_loss(pred, batch, item=item)
    target = batch.target_fields[item : item + 1] if item is not None else batch.target_fields
    obs_values = batch.obs_values[item : item + 1] if item is not None and batch.obs_values is not None else batch.obs_values
    return _observation_loss(pred, target.to(pred.device, pred.dtype), _item_mask(batch, item), obs_values)


def sparse_inverse_observation_loss(solution: torch.Tensor, batch: PDEBatch, item: int | None = None) -> torch.Tensor:
    target = batch.input_fields[item : item + 1] if item is not None else batch.input_fields
    obs_values = batch.obs_values[item : item + 1] if item is not None and batch.obs_values is not None else batch.obs_values
    return _observation_loss(solution, target.to(solution.device, solution.dtype), _item_mask(batch, item), obs_values)


def sparse_forward_observation_loss(unknown: torch.Tensor, batch: PDEBatch, item: int | None = None) -> torch.Tensor:
    # sparse_forward observes the source/coefficient field, so the reconstructed
    # `unknown` (source) must match the observed source values at sensor points.
    target = batch.input_fields[item : item + 1] if item is not None else batch.input_fields
    obs_values = batch.obs_values[item : item + 1] if item is not None and batch.obs_values is not None else batch.obs_values
    return _observation_loss(unknown, target.to(unknown.device, unknown.dtype), _item_mask(batch, item), obs_values)


def sparse_inverse_physics_loss(
    unknown: torch.Tensor,
    solution: torch.Tensor,
    batch: PDEBatch,
    item: int | None,
    config: dict,
) -> torch.Tensor:
    if batch.pde_name.lower() not in STATIC_SPARSE_INVERSE_PDES:
        return torch.tensor(float("nan"), device=unknown.device, dtype=unknown.dtype)
    meta = _single_meta(batch, item) if item is not None else _metadata_without_private_truth(batch)
    meta.update(_physics_weight_metadata(config))
    meta["task"] = "sparse_inverse"
    meta["solution_fields"] = solution
    meta["input_fields"] = solution
    losses = physics_loss_metric(unknown, batch.pde_name, meta)
    return _select_physics_loss(losses, config, unknown)


def _observation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    obs_values: torch.Tensor | None = None,
) -> torch.Tensor:
    if obs_values is not None and mask is not None:
        c = min(pred.shape[1], obs_values.shape[-1])
        masks = mask
        if tuple(masks.shape) == tuple(pred.shape[1:]):
            masks = masks.unsqueeze(0).expand(pred.shape[0], *masks.shape)
        if tuple(masks.shape) != tuple(pred.shape):
            raise ValueError(
                f"Observation mask shape {tuple(mask.shape)} must match prediction with or without batch "
                f"({tuple(pred.shape)} or {tuple(pred.shape[1:])})"
            )
        flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
        pred_obs = torch.stack(
            [flat[i, :, masks[i, 0].bool().reshape(-1)].transpose(0, 1) for i in range(pred.shape[0])],
            dim=0,
        )[..., :c]
        obs = obs_values.to(pred.device, pred.dtype)[..., :c]
        return F.mse_loss(pred_obs, obs)
    if mask is None:
        return F.mse_loss(pred, target)
    local_mask = mask
    if tuple(local_mask.shape) == tuple(pred.shape[1:]):
        local_mask = local_mask.unsqueeze(0).expand(pred.shape[0], *local_mask.shape)
    if tuple(local_mask.shape) != tuple(pred.shape):
        raise ValueError(f"Observation mask shape {tuple(mask.shape)} does not match prediction {tuple(pred.shape)}")
    local_mask = local_mask.to(pred.device, pred.dtype)
    return (((pred - target) ** 2) * local_mask).sum() / local_mask.sum().clamp_min(1.0)


def _item_mask(batch: PDEBatch, item: int | None) -> torch.Tensor | None:
    mask = batch.mask
    if mask is None or item is None:
        return mask
    if mask.ndim == batch.input_fields.ndim and mask.shape[0] == batch.input_fields.shape[0]:
        return mask[item : item + 1]
    return mask


def _physics_weight_metadata(config: dict) -> dict:
    lam_pde = float(config.get("lambda_pde", config.get("lambda_dynamics", 0.01)))
    return {
        "lambda_int": float(config.get("lambda_int", lam_pde)),
        "lambda_bc": float(config.get("lambda_bc", lam_pde)),
        "lambda_ic": float(config.get("lambda_ic", lam_pde)),
    }


def _select_physics_loss(losses: dict, config: dict, ref: torch.Tensor) -> torch.Tensor:
    mode = str(config.get("physics_loss_mode", "total")).lower()
    if mode == "interior":
        lam_int = float(config.get("lambda_int", config.get("lambda_pde", config.get("lambda_dynamics", 0.01))))
        return lam_int * losses["interior"]
    if mode != "total":
        raise ValueError(f"Unsupported physics_loss_mode '{mode}'")
    total = losses["total"]
    if isinstance(total, torch.Tensor):
        return total
    return ref.sum() * 0.0


def _smoothness_reg(field: torch.Tensor) -> torch.Tensor:
    if field.ndim < 4:
        return field.pow(2).mean()
    terms = []
    if field.shape[-2] > 1:
        terms.append((field[..., 1:, :] - field[..., :-1, :]).pow(2).mean())
    if field.shape[-1] > 1:
        terms.append((field[..., :, 1:] - field[..., :, :-1]).pow(2).mean())
    if not terms:
        return field.pow(2).mean()
    return sum(terms)


def _single_meta(batch: PDEBatch, item: int) -> dict:
    meta = _metadata_without_private_truth(batch)
    for key, value in list(meta.items()):
        if isinstance(value, torch.Tensor) and value.shape[:1] == batch.target_fields.shape[:1]:
            meta[key] = value[item : item + 1]
    return {
        "input_fields": batch.input_fields[item : item + 1],
        "task": batch.task,
        **meta,
    }


def _metadata_without_private_truth(batch: PDEBatch) -> dict:
    private_truth_keys = {
        "full_tensor",
        "original_input_fields",
        "observed_solution_fields",
        "observation_source_fields",
        "background_fields",
        "full_trajectory",
        "solution_fields",
        "source_fields",
        "coeff_fields",
        "initial_1d",
    }
    return {key: value for key, value in batch.metadata.items() if key not in private_truth_keys}
