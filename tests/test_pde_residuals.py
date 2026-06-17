from __future__ import annotations

import pytest
import torch

from baselines.common.data_adapter import build_default_registry
from baselines.common.metrics import bc_residual_metric, ic_residual_metric, pde_residual_metric
from baselines.common.physics import periodic_bc_loss, physics_losses


CURRENT_PDES = [
    "poisson",
    "helmholtz",
    "darcy",
    "burger",
    "nsnonbounded",
    "reaction_diffusion",
    "shallow_water",
    "heat",
    "wave",
    "advection_diffusion",
    "steady_heat_conduction",
]


@pytest.mark.parametrize("pde", CURRENT_PDES)
def test_physics_losses_return_structured_finite_scalars(pde):
    registry = build_default_registry()
    raw = registry.synthetic_raw(pde, n=1, resolution=8)
    batch = registry.make_task(raw, pde, "forward", num_sensors=4, seed=2)
    pred = batch.target_fields.clone()
    if pde == "shallow_water":
        pred[:, 0] = pred[:, 0].abs() + 1.0
    losses = physics_losses(
        pred,
        pde,
        {"input_fields": batch.input_fields, "full_tensor": batch.full_tensor, "task": batch.task, **batch.metadata},
    )
    for key in ("interior", "bc", "ic", "total"):
        assert isinstance(losses[key], torch.Tensor)
        assert losses[key].ndim == 0
        assert torch.isfinite(losses[key])
    assert isinstance(losses["mode"], str)
    assert "residual" in losses


@pytest.mark.parametrize("pde", ["darcy", "poisson", "helmholtz"])
def test_steady_dirichlet_pdes_have_zero_ic_loss(pde):
    pred = torch.zeros(1, 1, 8, 8)
    meta = {"input_fields": torch.ones(1, 1, 8, 8), "task": "forward"}
    assert ic_residual_metric(pred, pde, meta).item() == 0.0


@pytest.mark.parametrize("pde", ["darcy", "poisson", "helmholtz"])
def test_dirichlet_bc_loss_detects_boundary_values(pde):
    pred = torch.zeros(1, 1, 8, 8)
    meta = {"input_fields": torch.ones(1, 1, 8, 8), "task": "forward", "k": 1.0}
    assert bc_residual_metric(pred, pde, meta).item() == pytest.approx(0.0)
    pred_bad = pred.clone()
    pred_bad[..., 0, :] = 1.0
    assert bc_residual_metric(pred_bad, pde, meta).item() > 0.0


def test_periodic_bc_loss_for_burgers_and_ns_consistent_tensors_is_zero():
    burgers = torch.zeros(1, 1, 4, 8)
    ns = torch.zeros(1, 1, 4, 8, 8)
    assert periodic_bc_loss(burgers, dims=(-1,)).item() == pytest.approx(0.0)
    assert periodic_bc_loss(ns, dims=(-2, -1)).item() == pytest.approx(0.0)


@pytest.mark.parametrize("pde,channels", [("reaction_diffusion", 2), ("shallow_water", 3)])
def test_neumann_bc_loss_for_constant_rd_swe_fields_is_zero(pde, channels):
    pred = torch.ones(1, channels, 8, 8)
    if pde == "shallow_water":
        pred[:, 1:] = 0.0
    meta = {"input_fields": pred.clone(), "task": "forward", "final_time": 1.0}
    if pde == "reaction_diffusion":
        meta.update({"D_u": 1e-3, "D_v": 5e-3, "k": 5e-3, "final_time": 5.0})
    else:
        meta.update({"g": 1.0, "domain_length": 5.0})
    assert bc_residual_metric(pred, pde, meta).item() == pytest.approx(0.0)


def test_time_dependent_ic_losses_are_zero_when_prediction_starts_from_metadata_initial_state():
    burgers = torch.zeros(1, 1, 5, 8)
    burgers_meta = {"initial_1d": torch.zeros(1, 8), "task": "forward", "nu": 0.01, "final_time": 1.0}
    assert ic_residual_metric(burgers, "burger", burgers_meta).item() == pytest.approx(0.0)

    ns_pred = torch.zeros(1, 4, 8, 8)
    ns_full = torch.zeros(1, 1, 5, 8, 8)
    ns_meta = {"input_fields": torch.zeros(1, 1, 8, 8), "full_tensor": ns_full, "task": "forward", "nu": 1e-3, "final_time": 1.0}
    assert ic_residual_metric(ns_pred, "nsnonbounded", ns_meta).item() == pytest.approx(0.0)

    rd = torch.zeros(1, 2, 8, 8)
    rd_meta = {"input_fields": torch.zeros_like(rd), "task": "forward", "D_u": 1e-3, "D_v": 5e-3, "k": 5e-3}
    assert ic_residual_metric(rd, "reaction_diffusion", rd_meta).item() == pytest.approx(0.0)

    swe = torch.zeros(1, 3, 8, 8)
    swe[:, 0] = 1.0
    swe_meta = {"input_fields": swe.clone(), "task": "forward", "g": 1.0, "domain_length": 5.0}
    assert ic_residual_metric(swe, "shallow_water", swe_meta).item() == pytest.approx(0.0)


def test_future_time_dependent_constant_fields_have_zero_residual_terms():
    heat = torch.ones(1, 1, 8, 8)
    heat_meta = {"input_fields": heat.clone(), "task": "forward", "alpha": 1e-3, "final_time": 1.0, "bc": "periodic"}
    assert pde_residual_metric(heat, "heat", heat_meta).item() == pytest.approx(0.0)
    assert bc_residual_metric(heat, "heat", heat_meta).item() == pytest.approx(0.0)
    assert ic_residual_metric(heat, "heat", heat_meta).item() == pytest.approx(0.0)

    wave = torch.zeros(1, 2, 8, 8)
    wave[:, 0] = 1.0
    wave_meta = {"input_fields": wave.clone(), "task": "forward", "fixed_c": 1.0, "final_time": 1.0, "bc": "periodic"}
    assert pde_residual_metric(wave, "wave", wave_meta).item() == pytest.approx(0.0)
    assert bc_residual_metric(wave, "wave", wave_meta).item() == pytest.approx(0.0)
    assert ic_residual_metric(wave, "wave", wave_meta).item() == pytest.approx(0.0)

    adv = torch.ones(1, 1, 8, 8)
    adv_meta = {"input_fields": adv.clone(), "task": "forward", "b_x": 0.5, "b_y": -0.25, "kappa": 1e-3, "final_time": 1.0}
    assert pde_residual_metric(adv, "advection_diffusion", adv_meta).item() == pytest.approx(0.0)
    assert bc_residual_metric(adv, "advection_diffusion", adv_meta).item() == pytest.approx(0.0)
    assert ic_residual_metric(adv, "advection_diffusion", adv_meta).item() == pytest.approx(0.0)


def test_steady_heat_conduction_constant_solution_satisfies_zero_source_case():
    solution = torch.full((1, 1, 8, 8), 298.0)
    source = torch.zeros_like(solution)
    u_d = torch.full_like(solution, 298.0)
    meta = {"input_fields": torch.cat([source, u_d], dim=1), "task": "forward", "u_D": 298.0}
    assert pde_residual_metric(solution, "steady_heat_conduction", meta).item() == pytest.approx(0.0)
    assert bc_residual_metric(solution, "steady_heat_conduction", meta).item() == pytest.approx(0.0)
    assert ic_residual_metric(solution, "steady_heat_conduction", meta).item() == pytest.approx(0.0)


@pytest.mark.parametrize("pde", CURRENT_PDES)
def test_metric_wrappers_do_not_return_nan(pde):
    registry = build_default_registry()
    raw = registry.synthetic_raw(pde, n=1, resolution=8)
    batch = registry.make_task(raw, pde, "forward", num_sensors=4, seed=3)
    pred = batch.target_fields.clone()
    if pde == "shallow_water":
        pred[:, 0] = pred[:, 0].abs() + 1.0
    meta = {"input_fields": batch.input_fields, "full_tensor": batch.full_tensor, "task": batch.task, **batch.metadata}
    for fn in (pde_residual_metric, bc_residual_metric, ic_residual_metric):
        value = fn(pred, pde, meta)
        assert value.ndim == 0
        assert torch.isfinite(value)
