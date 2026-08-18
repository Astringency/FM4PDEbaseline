from __future__ import annotations

import time
import warnings

import torch
import torch.nn.functional as F

from baselines.common.data_adapter import PDEBatch
from baselines.common.metrics import physics_loss_metric
from baselines.common.normalization import (
    denormalize_prediction,
    estimate_normalization_stats,
    normalize_batch_input_target,
)

from .base import BaselineModel, restore_state_dict, snapshot_state_dict
from .official import (
    OfficialImportError,
    get_vivid_official_aligned_status,
    get_vivid_official_status,
    official_source_info,
    requested_implementation_mode,
)
from .pinn_sparse import _physics_weight_metadata, _select_physics_loss, observation_loss_from_batch
from .var4d import _assimilation_mode, _background_view, _initial_trajectory, _optimized_state_numel
from .vivid_official_aligned import OfficialAlignedVIVIDInverseObservation
from .voronoicnn import VoronoiCNNBaseline


class VIVIDBaseline(BaselineModel):
    name = "vivid"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        self.optimized_numel = _optimized_state_numel(data_spec)
        self.inverse_operator_trained = False
        self.official_aligned = False
        implementation_mode = requested_implementation_mode(self.config)
        backend = str(self.config.get("official_backend", "auto")).lower()
        if implementation_mode == "official" and backend not in {"local", "none", "adapted"}:
            get_vivid_official_status()
            raise OfficialImportError("direct official VIVID/invobs inverse-observation adapter is not implemented after import status validation")
        official_like = bool(self.config.get("uses_official_inverse_observation_operator", False)) or implementation_mode in {
            "official_or_skip",
            "official_aligned",
            "auto",
        }
        official_like = official_like and implementation_mode != "adapted" and backend not in {"local", "none", "adapted"}
        if official_like:
            target_shape = tuple(data_spec["target_shape"])
            tensor_ndim = len(target_shape)
            channels = int(data_spec["target_channels"])
            self.inverse_operator = OfficialAlignedVIVIDInverseObservation(
                channels=channels,
                tensor_ndim=tensor_ndim,
                width=int(self.config.get("inverse_width", self.config.get("width", 48))),
                depth=int(self.config.get("inverse_depth", 6)),
            )
            direct_import_warning = ""
            if implementation_mode in {"official_or_skip", "auto"}:
                try:
                    get_vivid_official_status()
                except OfficialImportError as exc:
                    direct_import_warning = f"direct official VIVID/invobs import unavailable; using official-aligned reimplementation: {exc}"
            get_vivid_official_aligned_status()
            self.official_aligned = True
            self.set_backend(
                "vivid_official_aligned",
                "vivid",
                fallback_used=False,
                warning=direct_import_warning,
                implementation_mode_effective="official_aligned",
                implementation_source="vivid_invobs_official_aligned_reimplementation",
                official_import_success=False,
                official_reimplementation_success=True,
                official_alignment_level="objective",
                official_alignment_notes=(
                    "Reimplements VIVID Voronoi inverse-observation initialization and invobs-style "
                    "time-space convolutional inversion, followed by observation/background/dynamics variational refinement."
                ),
                adapter_status="official_aligned_vivid_invobs_reimplementation",
                **official_source_info("vivid_invobs"),
            )
        elif bool(self.config.get("train_inverse_operator", False)):
            self.inverse_operator = VoronoiCNNBaseline().build(config.get("inverse_operator", config), data_spec)
            self.set_backend(
                "vivid_style",
                "vivid_style",
                fallback_used=False,
                implementation_mode_effective="adapted",
                implementation_source="vivid_invobs_structure",
                official_import_success=False,
                adapter_status="vivid_style_trained_inverse_operator",
                **official_source_info("vivid_invobs"),
            )
        else:
            self.inverse_operator = VoronoiCNNBaseline().build(config.get("inverse_operator", config), data_spec)
            self.set_backend(
                "vivid_style",
                "vivid_style",
                fallback_used=False,
                implementation_mode_effective="adapted",
                implementation_source="vivid_style_no_inverse_operator",
                official_import_success=False,
                adapter_status="vivid_style_no_trained_inverse_operator",
                **official_source_info("vivid_invobs"),
            )
        return self

    def parameter_count(self) -> int:
        if hasattr(self.inverse_operator, "parameter_count"):
            inv_params = int(self.inverse_operator.parameter_count())
        else:
            inv_params = int(sum(p.numel() for p in self.inverse_operator.parameters() if p.requires_grad))
        return int(self.optimized_numel + inv_params)

    def fit(self, train_loader, val_loader=None):
        if getattr(self, "official_aligned", False):
            return self._fit_official_aligned_inverse_operator(train_loader, val_loader)
        if bool(self.config.get("train_inverse_operator", False)):
            self.inverse_operator_trained = True
            return self.inverse_operator.fit(train_loader, val_loader)
        return {"status": "per_instance_vivid_no_amortized_inverse_fit"}

    def predict(self, batch: PDEBatch):
        start = time.perf_counter()
        if not batch.metadata.get("supports_trajectory", False):
            mode = _assimilation_mode(batch)
            if mode == "two_level_surrogate":
                warnings.warn("VIVID requested without full trajectory; using a two-level dynamics surrogate.", RuntimeWarning, stacklevel=2)
            else:
                warnings.warn("VIVID requested without full trajectory; using state reconstruction variant.", RuntimeWarning, stacklevel=2)
        with torch.no_grad():
            if getattr(self, "official_aligned", False):
                inverse_batch = batch
                if self.uses_normalization and self.normalization_stats is not None:
                    inverse_batch = normalize_batch_input_target(batch, self.normalization_stats)
                learned_state = self.inverse_operator(inverse_batch)
                if self.uses_normalization and self.normalization_stats is not None:
                    learned_state = denormalize_prediction(learned_state, self.normalization_stats)
            elif self.inverse_operator_trained:
                if hasattr(self.inverse_operator, "predict_physical"):
                    learned_state = self.inverse_operator.predict_physical(batch)
                else:
                    learned_state = self.inverse_operator.predict(batch)
            else:
                learned_state = batch.metadata.get("voronoi_grid", batch.input_fields).detach()
        state0, output_view, background, dyn_meta = _initial_trajectory(batch)
        batch.metadata["assimilation_mode"] = str(dyn_meta.get("assimilation_mode", _assimilation_mode(batch)))
        batch.metadata["inverse_observation_operator_used"] = bool(getattr(self, "official_aligned", False) or self.inverse_operator_trained)
        batch.metadata["official_alignment_level"] = self.official_alignment_level
        state0, injection_mode = _inject_learned_state_into_trajectory(state0, learned_state, batch)
        batch.metadata["learned_state_injection_mode"] = injection_mode
        batch.metadata["learned_state_shape"] = tuple(learned_state.shape)
        batch.metadata["optimized_state_shape"] = tuple(state0.shape)
        state = torch.nn.Parameter(state0.detach().clone())
        steps = int(self.config.get("refine_steps", 2))
        lr = float(self.config.get("lr", 1e-2))
        lam_inv = float(self.config.get("lambda_inverse_operator", 0.25))
        lam_obs = float(self.config.get("lambda_obs", 1.0))
        lam_b = float(self.config.get("lambda_background", 0.05))
        opt = torch.optim.Adam([state], lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            output = output_view(state)
            obs = observation_loss_from_batch(output, batch)
            inv = _mse_aligned(output, learned_state)
            bg = F.mse_loss(_background_view(state, batch), background)
            dyn_meta = {**dyn_meta, **_physics_weight_metadata(self.config)}
            dyn = _select_physics_loss(physics_loss_metric(state, batch.pde_name, dyn_meta), self.config, state)
            if not torch.isfinite(dyn):
                dyn = torch.tensor(0.0, device=state.device)
            loss = lam_obs * obs + lam_inv * inv + lam_b * bg + dyn
            loss.backward()
            opt.step()
        batch.metadata["inference_optimization_time"] = time.perf_counter() - start
        return output_view(state).detach()

    def predict_physical(self, batch: PDEBatch):
        # VIVID.predict normalizes only the inverse-observation operator input,
        # denormalizes the learned state before variational refinement, and
        # returns the refined state in physical units. The base implementation
        # would normalize the whole batch and denormalize the already-physical
        # refined output a second time.
        return self.predict(batch)

    def _fit_official_aligned_inverse_operator(self, train_loader, val_loader=None):
        device = torch.device(self.config.get("device", "cpu"))
        epochs = int(self.config.get("epochs", 1))
        lr = float(self.config.get("inverse_lr", self.config.get("lr", 1e-3)))
        max_steps = self.config.get("max_steps")
        max_val_steps = self.config.get("max_val_steps")
        lam_obs = float(self.config.get("lambda_obs", 1.0))
        lam_dyn = float(self.config.get("lambda_inverse_dynamics", self.config.get("lambda_pde", 0.01)))
        normalize = bool(self.config.get("normalize", False))
        if normalize and self.normalization_stats is None:
            self.normalization_stats = estimate_normalization_stats(
                train_loader,
                max_batches=self.config.get("normalization_max_batches"),
                eps=float(self.config.get("normalization_eps", 1e-6)),
            )
        self.uses_normalization = bool(normalize and self.normalization_stats is not None)
        self.inverse_operator.to(device)
        opt = torch.optim.Adam(self.inverse_operator.parameters(), lr=lr)
        history = {
            "inverse_operator_loss": [],
            "obs_loss": [],
            "dynamics_loss": [],
            "val_loss": [],
            "best_epoch": None,
            "best_val_loss": None,
            "physics_trajectory_mode": [],
            "physics_trajectory_shape": [],
            "physics_loss_mode": [],
            "normalize": self.uses_normalization,
            "normalization_stats": self.normalization_stats.json_summary() if self.normalization_stats is not None else None,
        }
        best_val = None
        best_state = None
        for epoch in range(epochs):
            self.inverse_operator.train()
            inv_total = obs_total = dyn_total = 0.0
            count = 0
            for step, batch in enumerate(train_loader):
                if max_steps is not None and step >= int(max_steps):
                    break
                batch = _move_batch_tensors(batch, device)
                train_batch = batch
                if self.uses_normalization and self.normalization_stats is not None:
                    train_batch = normalize_batch_input_target(batch, self.normalization_stats)
                opt.zero_grad(set_to_none=True)
                pred = self.inverse_operator(train_batch)
                if tuple(pred.shape) != tuple(train_batch.target_fields.shape):
                    raise ValueError(
                        f"VIVID inverse-operator prediction {tuple(pred.shape)} must exactly match target "
                        f"{tuple(train_batch.target_fields.shape)}"
                    )
                inv = F.mse_loss(pred, train_batch.target_fields)
                pred_physical = denormalize_prediction(pred, self.normalization_stats) if self.uses_normalization and self.normalization_stats is not None else pred
                obs = observation_loss_from_batch(pred_physical, batch)
                dyn_meta = {**batch.metadata, **_physics_weight_metadata(self.config)}
                pred_for_physics, physics_mode = _trajectory_for_inverse_operator_physics(pred_physical, batch)
                physics_metrics = physics_loss_metric(pred_for_physics, batch.pde_name, dyn_meta)
                dyn = _select_physics_loss(physics_metrics, self.config, pred_for_physics)
                if not torch.isfinite(dyn):
                    dyn = torch.tensor(0.0, device=pred_physical.device, dtype=pred_physical.dtype)
                loss = inv + lam_obs * obs + lam_dyn * dyn
                loss.backward()
                opt.step()
                inv_total += float(inv.detach().cpu())
                obs_total += float(obs.detach().cpu())
                dyn_total += float(dyn.detach().cpu())
                if not history["physics_trajectory_mode"]:
                    history["physics_trajectory_mode"].append(physics_mode)
                    history["physics_trajectory_shape"].append(tuple(pred_for_physics.shape))
                    history["physics_loss_mode"].append(str(physics_metrics["mode"]))
                count += 1
            history["inverse_operator_loss"].append(inv_total / max(count, 1))
            history["obs_loss"].append(obs_total / max(count, 1))
            history["dynamics_loss"].append(dyn_total / max(count, 1))
            if val_loader is not None:
                self.inverse_operator.eval()
                val_total = 0.0
                val_count = 0
                with torch.no_grad():
                    for step, batch in enumerate(val_loader):
                        if max_val_steps is not None and step >= int(max_val_steps):
                            break
                        batch = _move_batch_tensors(batch, device)
                        val_batch = normalize_batch_input_target(batch, self.normalization_stats) if self.uses_normalization and self.normalization_stats is not None else batch
                        pred = self.inverse_operator(val_batch)
                        if tuple(pred.shape) != tuple(val_batch.target_fields.shape):
                            raise ValueError(
                                f"VIVID validation prediction {tuple(pred.shape)} must exactly match target "
                                f"{tuple(val_batch.target_fields.shape)}"
                            )
                        val_total += float(F.mse_loss(pred, val_batch.target_fields).detach().cpu())
                        val_count += 1
                val_loss = val_total / max(val_count, 1)
                history["val_loss"].append(val_loss)
                if best_val is None or val_loss < best_val:
                    best_val = val_loss
                    history["best_epoch"] = epoch
                    history["best_val_loss"] = val_loss
                    best_state = snapshot_state_dict(self.inverse_operator)
        if best_state is not None:
            restore_state_dict(self.inverse_operator, best_state)
        self.inverse_operator_trained = True
        return history


def _mse_aligned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape == b.shape:
        return F.mse_loss(a, b)
    if a.ndim == 5 and b.ndim == 4:
        terminal = a[:, :, -1]
        if tuple(terminal.shape) == tuple(b.shape):
            return F.mse_loss(terminal, b)
    if a.ndim == 4 and b.ndim == 5:
        terminal = b[:, :, -1]
        if tuple(a.shape) == tuple(terminal.shape):
            return F.mse_loss(a, terminal)
    raise ValueError(
        f"VIVID state shapes must match exactly (or be an explicit full-trajectory/terminal pair), got "
        f"{tuple(a.shape)} and {tuple(b.shape)}"
    )


def _inject_learned_state_into_trajectory(state0: torch.Tensor, learned_state: torch.Tensor, batch: PDEBatch) -> tuple[torch.Tensor, str]:
    pde = batch.pde_name.lower()
    if state0.ndim == 5:
        if pde == "nsnonbounded" and learned_state.ndim == 5 and learned_state.shape[2] == state0.shape[2] - 1:
            out = state0.clone()
            out[:, :, 1:] = learned_state[:, : state0.shape[1]]
            return out, "future_trajectory_inserted"
        if learned_state.ndim == 5 and tuple(learned_state.shape) == tuple(state0.shape):
            return learned_state.clone(), "full_trajectory_replaced"
        if pde in {"reaction_diffusion", "shallow_water"} and learned_state.ndim == 5 and tuple(learned_state.shape) == tuple(state0.shape):
            return learned_state.clone(), "full_trajectory_replaced"
        out = state0.clone()
        terminal = learned_state[:, : state0.shape[1]]
        if terminal.ndim == 5:
            terminal = terminal[:, :, -1]
        out[:, :, -1] = terminal
        return out, "terminal_state_inserted"
    if pde == "burger" and learned_state.ndim == state0.ndim and tuple(learned_state.shape) == tuple(state0.shape):
        return learned_state.clone(), "full_trajectory_replaced"
    return learned_state, "state_replaced"


def _trajectory_for_inverse_operator_physics(pred: torch.Tensor, batch: PDEBatch) -> tuple[torch.Tensor, str]:
    pde = batch.pde_name.lower()
    if pde == "nsnonbounded":
        if pred.ndim == 5:
            full = batch.full_tensor
            if full.ndim == 5 and pred.shape[2] == full.shape[2]:
                return pred, "full_trajectory"
            initial = _initial_frame_for_physics(batch, pred, channels=pred.shape[1])
            if full.ndim != 5:
                return torch.cat([initial[:, :, None], pred], dim=2), "initial_prepended"
            if full.ndim == 5 and pred.shape[2] == full.shape[2] - 1:
                return torch.cat([initial[:, :, None], pred], dim=2), "initial_prepended"
        return pred, "as_predicted"

    if pde in {"reaction_diffusion", "shallow_water"}:
        if pred.ndim == 5:
            full = batch.full_tensor
            input_idx = int(batch.metadata.get("input_time_index", 0))
            if full.ndim == 5:
                expected_with_initial = max(full.shape[2] - input_idx, 0)
                expected_future = max(full.shape[2] - input_idx - 1, 0)
                if pred.shape[2] == expected_with_initial:
                    return pred, "full_trajectory"
                if pred.shape[2] == expected_future:
                    initial = _initial_frame_for_physics(batch, pred, channels=pred.shape[1])
                    return torch.cat([initial[:, :, None], pred], dim=2), "initial_prepended"
            return pred, "full_trajectory"
        return pred, "final_state"

    if pde == "burger":
        if pred.ndim == 4:
            full = batch.full_tensor
            if full.ndim == 4 and pred.shape[-2] == full.shape[-2]:
                return pred, "full_trajectory"
            if full.ndim == 4 and pred.shape[-2] == full.shape[-2] - 1:
                initial = _initial_frame_for_physics(batch, pred, channels=pred.shape[1])
                return torch.cat([initial, pred], dim=2), "initial_prepended"
        return pred, "full_trajectory" if pred.ndim == 4 else "as_predicted"

    return pred, "as_predicted"


def _initial_frame_for_physics(batch: PDEBatch, ref: torch.Tensor, channels: int) -> torch.Tensor:
    background = batch.metadata.get("background_fields")
    if isinstance(background, torch.Tensor):
        bg = background.to(ref.device, ref.dtype)
        if bg.ndim == 5:
            bg = bg[:, :, 0]
        if bg.ndim == 4 and batch.pde_name.lower() == "burger":
            if bg.shape[-2] == 1:
                return bg[:, :channels]
            return bg[:, :channels, :1, :]
        if bg.ndim == 4:
            return bg[:, :channels]
    full = batch.full_tensor.to(ref.device, ref.dtype)
    if full.ndim == 5:
        return full[:, :channels, 0]
    if full.ndim == 4 and batch.pde_name.lower() == "burger":
        return full[:, :channels, :1, :]
    if batch.input_fields.ndim == 4:
        return batch.input_fields.to(ref.device, ref.dtype)[:, :channels]
    raise ValueError(f"Cannot recover initial frame for {batch.pde_name}")


def _move_batch_tensors(batch: PDEBatch, device: torch.device) -> PDEBatch:
    from .base import _to_device_batch

    return _to_device_batch(batch, device)
