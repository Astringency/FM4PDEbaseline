from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from baselines.common.data_adapter import PDEBatch
from baselines.common.metrics import physics_loss_metric

from .base import BaselineModel
from .official import OfficialImportError, get_vivid_official_status, official_source_info, requested_implementation_mode
from .pinn_sparse import _physics_weight_metadata, _select_physics_loss, observation_loss_from_batch
from .var4d import _assimilation_mode, _background_view, _initial_trajectory, _optimized_state_numel
from .voronoicnn import VoronoiCNNBaseline


class VIVIDBaseline(BaselineModel):
    """Burgers-only VIVID-style adapter.

    The vendored VIVID and invobs repositories do not expose an importable
    Burgers pipeline. This adapter therefore remains explicitly local/adapted:
    a trained Voronoi inverse model initializes a weak-constraint trajectory
    refinement. It must not be reported as official VIVID or invobs.
    """

    name = "vivid"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        pde = str(data_spec.get("pde", "")).lower()
        if pde != "burger":
            raise ValueError("The audited VIVID adapter is scoped only to Burgers sparse trajectory reconstruction")

        implementation_mode = requested_implementation_mode(self.config)
        if implementation_mode in {"official", "official_or_skip", "official_aligned", "official_architecture"}:
            # This always raises until a stable official Burgers component is
            # available. Do not silently relabel a local reimplementation.
            get_vivid_official_status()
            raise OfficialImportError("No importable official VIVID/invobs Burgers adapter is available")

        self.optimized_numel = _optimized_state_numel(data_spec)
        self.inverse_operator_trained = False
        self.inverse_operator = VoronoiCNNBaseline().build(config.get("inverse_operator", config), data_spec)
        self.set_backend(
            "vivid_style_burgers",
            "vivid_invobs_reference",
            fallback_used=False,
            implementation_mode_effective="adapted",
            implementation_source="local_voronoi_inverse_plus_weak_constraint_refinement",
            official_import_success=False,
            official_reimplementation_success=False,
            official_alignment_level="local",
            official_alignment_notes=(
                "Uses a VoronoiCNN inverse model and Adam refinement of the complete Burgers trajectory. The vendored "
                "VIVID recipe instead uses a 7-layer 8x8 Keras CNN and ADAO L-BFGS-B 3DVAR; invobs uses a "
                "dataset-specific JAX dynamics pipeline and SciPy L-BFGS-B."
            ),
            adapter_status="vivid_style_burgers_adapted",
            **official_source_info("vivid_invobs"),
        )
        return self

    def parameter_count(self) -> int:
        return int(self.optimized_numel + self.inverse_operator.parameter_count())

    def fit(self, train_loader, val_loader=None):
        if not bool(self.config.get("train_inverse_operator", False)):
            return {"status": "per_instance_vivid_no_amortized_inverse_fit"}
        history = self.inverse_operator.fit(train_loader, val_loader)
        self.inverse_operator_trained = True
        return history

    def predict(self, batch: PDEBatch):
        if batch.pde_name.lower() != "burger" or not batch.metadata.get("supports_trajectory", False):
            raise ValueError("VIVID requires a loaded Burgers T x X trajectory")

        start = time.perf_counter()
        with torch.no_grad():
            if self.inverse_operator_trained:
                if hasattr(self.inverse_operator, "predict_physical"):
                    learned_state = self.inverse_operator.predict_physical(batch)
                else:
                    learned_state = self.inverse_operator.predict(batch)
            else:
                learned_state = batch.metadata.get("voronoi_grid", batch.input_fields).detach()

        state0, output_view, background, dyn_meta = _initial_trajectory(batch)
        batch.metadata["assimilation_mode"] = str(dyn_meta.get("assimilation_mode", _assimilation_mode(batch)))
        batch.metadata["assimilation_background_source"] = str(
            dyn_meta.get("assimilation_background_source", "task_background")
        )
        batch.metadata["assimilation_uses_hidden_truth"] = bool(dyn_meta.get("assimilation_uses_hidden_truth", False))
        batch.metadata["inverse_observation_operator_used"] = bool(self.inverse_operator_trained)
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
            physics_meta = {**dyn_meta, **_physics_weight_metadata(self.config)}
            dyn = _select_physics_loss(physics_loss_metric(state, batch.pde_name, physics_meta), self.config, state)
            if not torch.isfinite(dyn):
                dyn = torch.tensor(0.0, device=state.device)
            loss = lam_obs * obs + lam_inv * inv + lam_b * bg + dyn
            loss.backward()
            opt.step()

        batch.metadata["inference_optimization_time"] = time.perf_counter() - start
        return output_view(state).detach()

    def predict_physical(self, batch: PDEBatch):
        # The inverse model returns physical values and refinement also runs in
        # physical units, so the base class must not denormalize a second time.
        return self.predict(batch)


def _mse_aligned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(f"VIVID Burgers trajectory shapes must match exactly, got {tuple(a.shape)} and {tuple(b.shape)}")
    return F.mse_loss(a, b)


def _inject_learned_state_into_trajectory(
    state0: torch.Tensor,
    learned_state: torch.Tensor,
    batch: PDEBatch,
) -> tuple[torch.Tensor, str]:
    if batch.pde_name.lower() != "burger":
        raise ValueError("VIVID learned-state injection supports only Burgers")
    if tuple(learned_state.shape) != tuple(state0.shape):
        raise ValueError(
            f"VIVID Burgers inverse prediction {tuple(learned_state.shape)} must match trajectory {tuple(state0.shape)}"
        )
    return learned_state.clone(), "full_trajectory_replaced"
