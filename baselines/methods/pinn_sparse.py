from __future__ import annotations

import time
import warnings

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from baselines.common.data_adapter import PDEBatch
from baselines.common.metrics import NotImplementedWarning, physics_loss_metric

from .base import BaselineModel, record_optimization_status, run_per_instance_optimizer
from .official import (
    OfficialImportError,
    get_deepxde_fnn_class,
    get_deepxde_module,
    official_source_info,
    requested_implementation_mode,
)
from .shared import NeuralField

STATIC_SPARSE_INVERSE_PDES = {"poisson", "helmholtz", "darcy", "steady_heat_conduction"}
DEEPXDE_STATIC_PDES = {"poisson", "helmholtz", "darcy"}


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
        self.deepxde = None
        backend = str(self.config.get("official_backend", "auto")).lower()
        implementation_mode = requested_implementation_mode(self.config)
        pde_name = str(data_spec.get("pde", "")).lower()
        fallback_reason = ""
        native_deepxde = pde_name in DEEPXDE_STATIC_PDES and (
            bool(self.config.get("deepxde_native", False))
            or (
                implementation_mode in {"official", "official_or_skip", "official_aligned"}
                and backend in {"deepxde", "official"}
            )
        )
        if native_deepxde:
            try:
                self.deepxde = get_deepxde_module()
                self.deepxde_fnn_cls = self.deepxde.nn.FNN
                self.set_backend(
                    "deepxde_native",
                    "deepxde",
                    fallback_used=False,
                    implementation_mode_effective="official_aligned",
                    implementation_source="deepxde_pde_model_with_fm4pde_task_adapter",
                    official_import_success=True,
                    official_reimplementation_success=False,
                    official_alignment_level="training_api",
                    official_alignment_notes=(
                        "Uses DeepXDE geometry, PDE data, PointSetBC, automatic differentiation, FNN, "
                        "and Adam/L-BFGS while adapting field components to FM4PDE tasks."
                    ),
                    adapter_status="deepxde_native_task_adapter",
                    **official_source_info("deepxde"),
                )
            except OfficialImportError:
                if backend in {"deepxde", "official"}:
                    raise
        elif implementation_mode != "adapted" and backend in {"auto", "deepxde", "official"}:
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
        if (
            self.deepxde is not None
            and batch.pde_name.lower() in DEEPXDE_STATIC_PDES
            and batch.task in {"sparse_forward", "sparse_inverse"}
        ):
            return self._predict_deepxde(batch)
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
        statuses = []
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
                physics_value = _select_physics_loss(
                    physics_loss_metric(pred, batch.pde_name, meta, strict=True), self.config, pred
                )
                loss = loss + physics_value
                loss.backward()
                return loss

            statuses.append(run_per_instance_optimizer(optimizer, closure, steps, self.config))
            with torch.no_grad():
                preds.append(model(coords).T.reshape_as(target))
        batch.metadata["inference_optimization_time"] = _synchronized_perf_counter(batch) - start
        record_optimization_status(batch, statuses)
        return torch.cat(preds, dim=0).detach()

    def _predict_deepxde(self, batch: PDEBatch) -> torch.Tensor:
        requested_device = torch.device(self.config.get("device", batch.target_fields.device))
        previous_device = torch.get_default_device()
        torch.set_default_device(requested_device)
        try:
            return self._predict_deepxde_on_default_device(batch)
        finally:
            torch.set_default_device(previous_device)

    def _predict_deepxde_on_default_device(self, batch: PDEBatch) -> torch.Tensor:
        pde = batch.pde_name.lower()
        if pde not in DEEPXDE_STATIC_PDES:
            raise NotImplementedError(f"DeepXDE PINN-Sparse supports static PDEs only, got {batch.pde_name}")
        if batch.obs_coords is None or batch.obs_values is None:
            raise ValueError("DeepXDE PINN-Sparse requires explicit obs_coords and obs_values")
        dde = self.deepxde
        start = _synchronized_perf_counter(batch)
        predictions: list[torch.Tensor] = []
        statuses: list[dict] = []
        adam_iterations = max(int(self.config.get("adam_iterations", self.config.get("steps", 1000))), 0)
        lbfgs_steps = max(int(self.config.get("lbfgs_steps", 0)), 0)
        lr = float(self.config.get("lr", 1e-3))
        num_domain = int(self.config.get("num_domain", min(int(batch.coords.shape[1]), 4096)))
        height, width = (int(value) for value in batch.target_fields.shape[-2:])
        num_boundary = int(self.config.get("num_boundary", 2 * (height + width)))
        seed = int(self.config.get("seed", 0))
        dde.config.set_random_seed(seed)
        for item in range(batch.target_fields.shape[0]):
            observation_coords = batch.obs_coords[item].detach().cpu().numpy().astype(np.float32, copy=False)
            observation_values = batch.obs_values[item].detach().cpu().numpy().astype(np.float32, copy=False)
            geometry = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
            pde_fn = _deepxde_static_pde(dde, pde, batch, item, self.config)
            observed_component = 0 if batch.task == "sparse_inverse" else 1
            conditions = [
                dde.icbc.PointSetBC(observation_coords, observation_values, component=observed_component)
            ]
            if pde == "helmholtz":
                boundary_conditions = _deepxde_helmholtz_boundary_conditions(
                    dde, geometry, batch, self.config
                )
                conditions[0:0] = boundary_conditions
            elif bool(self.config.get("deepxde_zero_boundary", True)):
                boundary_conditions = [
                    dde.icbc.DirichletBC(
                        geometry,
                        lambda x: np.zeros((len(x), 1), dtype=np.float32),
                        lambda _, on_boundary: on_boundary,
                        component=0,
                    )
                ]
                conditions.insert(
                    0,
                    boundary_conditions[0],
                )
            else:
                boundary_conditions = []
            data = dde.data.PDE(
                geometry,
                pde_fn,
                conditions,
                num_domain=num_domain,
                num_boundary=num_boundary if boundary_conditions else 0,
                train_distribution=str(self.config.get("train_distribution", "Hammersley")),
                anchors=observation_coords,
                num_test=int(self.config.get("num_test_collocation", min(num_domain, 1024))),
            )
            layers = [2] + [self.hidden] * max(self.depth, 1) + [2]
            net = dde.nn.FNN(
                layers,
                str(self.config.get("activation", "tanh")),
                str(self.config.get("kernel_initializer", "Glorot normal")),
            )
            if pde == "darcy":
                coefficient_floor = float(self.config.get("coefficient_floor", 1e-6))

                def output_transform(x, y):
                    del x
                    return torch.cat([y[:, :1], F.softplus(y[:, 1:2]) + coefficient_floor], dim=1)

                net.apply_output_transform(output_transform)
            model = dde.Model(data, net)
            loss_weights = _deepxde_loss_weights(self.config, len(boundary_conditions))
            callbacks = []
            if bool(self.config.get("early_stopping", False)) and adam_iterations > 0:
                callbacks.append(
                    dde.callbacks.EarlyStopping(
                        min_delta=float(self.config.get("early_stopping_min_delta", 1e-4)),
                        patience=int(self.config.get("early_stopping_patience", 20)),
                        monitor="loss_train",
                        start_from_epoch=int(self.config.get("min_steps", 0)),
                    )
                )
            completed = 0
            early_stopped = False
            if adam_iterations > 0:
                model.compile("adam", lr=lr, loss_weights=loss_weights)
                model.train(iterations=adam_iterations, callbacks=callbacks, display_every=max(adam_iterations, 1), verbose=0)
                completed += int(model.train_state.iteration)
                early_stopped = any(getattr(callback, "stopped_epoch", 0) > 0 for callback in callbacks)
            if lbfgs_steps > 0 and not early_stopped:
                dde.optimizers.config.set_LBFGS_options(
                    maxiter=lbfgs_steps,
                    ftol=float(self.config.get("lbfgs_ftol", 0.0)),
                    gtol=float(self.config.get("lbfgs_gtol", 1e-8)),
                )
                model.compile("L-BFGS", loss_weights=loss_weights)
                before = int(model.train_state.iteration)
                model.train(verbose=0)
                completed += max(int(model.train_state.iteration) - before, 0)
            grid_coords = batch.coords[item].detach().cpu().numpy().astype(np.float32, copy=False)
            joint = torch.as_tensor(model.predict(grid_coords), device=batch.target_fields.device, dtype=batch.target_fields.dtype)
            returned_component = 1 if batch.task == "sparse_inverse" else 0
            returned = joint[:, returned_component].reshape(1, 1, height, width)
            predictions.append(returned)
            statuses.append(
                {
                    "completed_steps": completed,
                    "early_stopped": early_stopped,
                    "best_loss": None,
                    "min_delta": float(self.config.get("early_stopping_min_delta", 1e-4)),
                    "patience": int(self.config.get("early_stopping_patience", 20)),
                    "stop_reason": "deepxde_early_stopping" if early_stopped else "",
                }
            )
        batch.metadata["inference_optimization_time"] = _synchronized_perf_counter(batch) - start
        batch.metadata["pinn_sparse_training_protocol"] = {
            "data": "dde.data.PDE",
            "observation_bc": "dde.icbc.PointSetBC",
            "derivatives": "deepxde_autodiff",
            "network": "dde.nn.FNN",
            "network_outputs": ["solution", "unknown"],
            "adam_iterations": adam_iterations,
            "lbfgs_steps": lbfgs_steps,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "boundary_conditions": "generator_aligned_kronecker" if pde == "helmholtz" else "zero_dirichlet",
        }
        record_optimization_status(batch, statuses)
        return torch.cat(predictions, dim=0).detach()

    def _predict_sparse_inverse(self, batch: PDEBatch):
        pde = batch.pde_name.lower()
        if pde not in STATIC_SPARSE_INVERSE_PDES:
            raise NotImplementedError(f"PINN-Sparse sparse_inverse is only enabled for static PDEs, got {batch.pde_name}")
        start = _synchronized_perf_counter(batch)
        preds = []
        statuses = []
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
                loss = loss + physics_value
                loss = loss + lam_reg * (_smoothness_reg(unknown_grid) + _smoothness_reg(solution_grid))
                loss.backward()
                return loss

            statuses.append(run_per_instance_optimizer(optimizer, closure, steps, self.config))
            with torch.no_grad():
                preds.append(unknown(coords).T.reshape(unknown_target_shape))
        batch.metadata["inference_optimization_time"] = _synchronized_perf_counter(batch) - start
        record_optimization_status(batch, statuses)
        return torch.cat(preds, dim=0).detach()

    def _predict_sparse_forward(self, batch: PDEBatch):
        pde = batch.pde_name.lower()
        if pde not in STATIC_SPARSE_INVERSE_PDES:
            raise NotImplementedError(f"PINN-Sparse sparse_forward is only enabled for static PDEs, got {batch.pde_name}")
        start = _synchronized_perf_counter(batch)
        preds = []
        statuses = []
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
                loss = loss + physics_value
                loss = loss + lam_reg * (_smoothness_reg(unknown_grid) + _smoothness_reg(solution_grid))
                loss.backward()
                return loss

            statuses.append(run_per_instance_optimizer(optimizer, closure, steps, self.config))
            with torch.no_grad():
                preds.append(solution(coords).T.reshape(solution_target_shape))
        batch.metadata["inference_optimization_time"] = _synchronized_perf_counter(batch) - start
        record_optimization_status(batch, statuses)
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
    losses = physics_loss_metric(unknown, batch.pde_name, meta, strict=True)
    return _select_physics_loss(losses, config, unknown)


def _deepxde_static_pde(dde, pde_name: str, batch: PDEBatch, item: int, config: dict):
    """Build an autodiff residual with outputs ordered as ``(solution, unknown)``."""

    del item
    operator_sign = float(
        config.get("elliptic_operator_sign", batch.metadata.get("elliptic_operator_sign", 1.0))
    )
    if pde_name == "poisson":

        def poisson(x, y):
            solution_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
            solution_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
            return operator_sign * (solution_xx + solution_yy) - y[:, 1:2]

        return poisson
    if pde_name == "helmholtz":
        wave_number = float(config.get("k", batch.metadata.get("k", 1.0)))

        def helmholtz(x, y):
            solution_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
            solution_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
            operator = solution_xx + solution_yy + wave_number**2 * y[:, :1]
            return operator_sign * operator - y[:, 1:2]

        return helmholtz
    if pde_name == "darcy":

        def darcy(x, y):
            pressure_x = dde.grad.jacobian(y, x, i=0, j=0)
            pressure_y = dde.grad.jacobian(y, x, i=0, j=1)
            pressure_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
            pressure_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
            coefficient_x = dde.grad.jacobian(y, x, i=1, j=0)
            coefficient_y = dde.grad.jacobian(y, x, i=1, j=1)
            coefficient = y[:, 1:2]
            return -(coefficient_x * pressure_x + coefficient_y * pressure_y + coefficient * (pressure_xx + pressure_yy)) - 1.0

        return darcy
    raise NotImplementedError(f"No DeepXDE-native static residual for {pde_name}")


def _deepxde_helmholtz_boundary_conditions(dde, geometry, batch: PDEBatch, config: dict):
    """Express the MATLAB Kronecker boundary rows as DeepXDE operator BCs."""

    wave_number = float(config.get("k", batch.metadata.get("k", 1.0)))
    operator_sign = float(
        config.get("elliptic_operator_sign", batch.metadata.get("elliptic_operator_sign", 1.0))
    )

    def horizontal(inputs, outputs, _):
        tangent = dde.grad.hessian(outputs, inputs, component=0, i=1, j=1)
        return operator_sign * (tangent + (1.0 + wave_number**2) * outputs[:, :1])

    def vertical(inputs, outputs, _):
        tangent = dde.grad.hessian(outputs, inputs, component=0, i=0, j=0)
        return operator_sign * (tangent + (1.0 + wave_number**2) * outputs[:, :1])

    def corners(_, outputs, __):
        return operator_sign * (2.0 + wave_number**2) * outputs[:, :1]

    def is_horizontal(x, on_boundary):
        return bool(on_boundary and np.isclose(x[0], (0.0, 1.0)).any() and not np.isclose(x[1], (0.0, 1.0)).any())

    def is_vertical(x, on_boundary):
        return bool(on_boundary and np.isclose(x[1], (0.0, 1.0)).any() and not np.isclose(x[0], (0.0, 1.0)).any())

    return [
        dde.icbc.OperatorBC(geometry, horizontal, is_horizontal),
        dde.icbc.OperatorBC(geometry, vertical, is_vertical),
        dde.icbc.PointSetOperatorBC(
            np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            np.zeros((4, 1), dtype=np.float32),
            corners,
        ),
    ]


def _deepxde_loss_weights(config: dict, boundary_condition_count: int) -> list[float]:
    weights = [float(config.get("lambda_int", config.get("lambda_pde", 1.0)))]
    weights.extend([float(config.get("lambda_bc", 1.0))] * int(boundary_condition_count))
    weights.append(float(config.get("lambda_obs", 1.0)))
    return weights


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
