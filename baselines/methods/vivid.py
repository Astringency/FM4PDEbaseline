from __future__ import annotations

import time
import warnings

import torch
import torch.nn.functional as F

from baselines.common.data_adapter import PDEBatch
from baselines.common.metrics import physics_loss_metric

from .base import BaselineModel
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
        official_like = bool(self.config.get("uses_official_inverse_observation_operator", False)) or implementation_mode in {
            "official",
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
            try:
                get_vivid_official_status()
                self.set_backend(
                    "vivid_official",
                    "vivid",
                    fallback_used=False,
                    implementation_mode_effective="official",
                    implementation_source="vivid_invobs_official",
                    official_import_success=True,
                    adapter_status="official_code_adapter",
                    **official_source_info("vivid_invobs"),
                )
            except OfficialImportError as exc:
                get_vivid_official_aligned_status()
                self.official_aligned = True
                self.set_backend(
                    "vivid_official_aligned",
                    "vivid",
                    fallback_used=False,
                    warning=f"direct official VIVID/invobs import unavailable; using official-aligned reimplementation: {exc}",
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
                learned_state = self.inverse_operator(batch)
            elif self.inverse_operator_trained:
                learned_state = self.inverse_operator.predict(batch)
            else:
                learned_state = batch.metadata.get("voronoi_grid", batch.input_fields).detach()
        state0, output_view, background, dyn_meta = _initial_trajectory(batch)
        batch.metadata["assimilation_mode"] = str(dyn_meta.get("assimilation_mode", _assimilation_mode(batch)))
        batch.metadata["inverse_observation_operator_used"] = bool(getattr(self, "official_aligned", False) or self.inverse_operator_trained)
        batch.metadata["official_alignment_level"] = self.official_alignment_level
        # Inject the learned inverse-operator estimate as the terminal state.
        if state0.ndim == 5:
            if learned_state.ndim == 5 and tuple(learned_state.shape) == tuple(state0.shape):
                state0 = learned_state.clone()
            else:
                state0 = state0.clone()
                terminal = learned_state[:, : state0.shape[1]]
                if terminal.ndim == 5:
                    terminal = terminal[:, :, -1]
                state0[:, :, -1] = terminal
        else:
            state0 = learned_state
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

    def _fit_official_aligned_inverse_operator(self, train_loader, val_loader=None):
        device = torch.device(self.config.get("device", "cpu"))
        epochs = int(self.config.get("epochs", 1))
        lr = float(self.config.get("inverse_lr", self.config.get("lr", 1e-3)))
        max_steps = self.config.get("max_steps")
        max_val_steps = self.config.get("max_val_steps")
        lam_obs = float(self.config.get("lambda_obs", 1.0))
        lam_dyn = float(self.config.get("lambda_inverse_dynamics", self.config.get("lambda_pde", 0.01)))
        self.inverse_operator.to(device)
        opt = torch.optim.Adam(self.inverse_operator.parameters(), lr=lr)
        history = {"inverse_operator_loss": [], "obs_loss": [], "dynamics_loss": [], "val_loss": []}
        for _ in range(epochs):
            self.inverse_operator.train()
            inv_total = obs_total = dyn_total = 0.0
            count = 0
            for step, batch in enumerate(train_loader):
                if max_steps is not None and step >= int(max_steps):
                    break
                batch = _move_batch_tensors(batch, device)
                opt.zero_grad(set_to_none=True)
                pred = self.inverse_operator(batch)
                inv = F.mse_loss(pred, batch.target_fields)
                obs = observation_loss_from_batch(pred, batch)
                dyn_meta = {**batch.metadata, **_physics_weight_metadata(self.config)}
                dyn = _select_physics_loss(physics_loss_metric(pred, batch.pde_name, dyn_meta), self.config, pred)
                if not torch.isfinite(dyn):
                    dyn = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                loss = inv + lam_obs * obs + lam_dyn * dyn
                loss.backward()
                opt.step()
                inv_total += float(inv.detach().cpu())
                obs_total += float(obs.detach().cpu())
                dyn_total += float(dyn.detach().cpu())
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
                        pred = self.inverse_operator(batch)
                        val_total += float(F.mse_loss(pred, batch.target_fields).detach().cpu())
                        val_count += 1
                history["val_loss"].append(val_total / max(val_count, 1))
        self.inverse_operator_trained = True
        return history


def _mse_aligned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape == b.shape:
        return F.mse_loss(a, b)
    if a.ndim == 5 and b.ndim == 4:
        return F.mse_loss(a[:, :, -1], b[:, : a.shape[1]])
    if a.ndim == 4 and b.ndim == 5:
        return F.mse_loss(a[:, : b.shape[1]], b[:, :, -1])
    return F.mse_loss(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)[:, : a.reshape(a.shape[0], -1).shape[1]])


def _move_batch_tensors(batch: PDEBatch, device: torch.device) -> PDEBatch:
    from .base import _to_device_batch

    return _to_device_batch(batch, device)
