from __future__ import annotations

import math

import pytest
import torch

from baselines.common.data_adapter import build_default_registry
from baselines.common.metrics import bc_residual_metric, ic_residual_metric, pde_residual_metric
from baselines.common.physics import (
    _ns_forcing,
    heat_residual,
    helmholtz_inverse_residual,
    navier_stokes_vorticity_residual,
    periodic_bc_loss,
    physics_losses,
    poisson_inverse_residual,
    wave_residual,
)


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


@pytest.mark.parametrize("pde", ["darcy", "poisson"])
def test_dirichlet_bc_loss_detects_boundary_values(pde):
    pred = torch.zeros(1, 1, 8, 8)
    meta = {"input_fields": torch.ones(1, 1, 8, 8), "task": "forward", "k": 1.0}
    assert bc_residual_metric(pred, pde, meta).item() == pytest.approx(0.0)
    pred_bad = pred.clone()
    pred_bad[..., 0, :] = 1.0
    assert bc_residual_metric(pred_bad, pde, meta).item() > 0.0


def test_helmholtz_generator_boundary_rows_are_part_of_the_pde_residual():
    pred = torch.zeros(1, 1, 8, 8)
    meta = {"input_fields": torch.ones(1, 1, 8, 8), "task": "forward", "k": 1.0}
    assert bc_residual_metric(pred, "helmholtz", meta).item() == pytest.approx(0.0)
    pred[..., 0, :] = 1.0
    assert pde_residual_metric(pred, "helmholtz", meta).item() > 0.0


def test_periodic_bc_loss_for_burgers_and_ns_consistent_tensors_is_zero():
    burgers = torch.zeros(1, 1, 4, 8)
    ns = torch.zeros(1, 1, 4, 8, 8)
    assert periodic_bc_loss(burgers, dims=(-1,)).item() == pytest.approx(0.0)
    assert periodic_bc_loss(ns, dims=(-2, -1)).item() == pytest.approx(0.0)


def test_periodic_bc_loss_does_not_force_distinct_endpoint_excluded_nodes_equal():
    x = torch.arange(32, dtype=torch.float64) / 32.0
    periodic_field = torch.sin(2.0 * math.pi * x).reshape(1, 1, 1, -1)
    assert periodic_bc_loss(periodic_field, dims=(-1,)).item() == pytest.approx(0.0)


@pytest.mark.parametrize("resolution", [32, 64])
def test_poisson_residual_matches_dataset_sign_convention(resolution):
    x = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    solution = (torch.sin(math.pi * xx) * torch.sin(math.pi * yy)).reshape(1, 1, resolution, resolution)
    source = -2.0 * math.pi**2 * solution
    rms = poisson_inverse_residual(source, solution).square().mean().sqrt()
    assert rms.item() < 0.03


@pytest.mark.parametrize("resolution", [32, 64])
def test_helmholtz_residual_matches_dataset_sign_convention(resolution):
    x = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    solution = (torch.sin(math.pi * xx) * torch.sin(math.pi * yy)).reshape(1, 1, resolution, resolution)
    source = (-2.0 * math.pi**2 + 1.0) * solution
    rms = helmholtz_inverse_residual(source, solution, k=1.0).square().mean().sqrt()
    assert rms.item() < 0.03


def test_navier_stokes_residual_uses_generator_spatial_axis_convention():
    n = 32
    x = torch.arange(n, dtype=torch.float64) / n
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    # Non-separable streamfunction gives a nonzero Jacobian, so exchanging the
    # generator's x/y axes or reversing advection is observable in this test.
    psi = torch.sin(2.0 * math.pi * xx) * torch.sin(4.0 * math.pi * yy) + 0.3 * torch.cos(
        6.0 * math.pi * xx
    ) * torch.sin(2.0 * math.pi * yy)
    psi_ft = torch.fft.rfft2(psi)
    kx = 2.0 * math.pi * torch.fft.fftfreq(n, d=1.0 / n, dtype=torch.float64).reshape(n, 1)
    ky = 2.0 * math.pi * torch.fft.rfftfreq(n, d=1.0 / n, dtype=torch.float64).reshape(1, -1)
    omega_ft = (kx.square() + ky.square()) * psi_ft
    omega = torch.fft.irfft2(omega_ft, s=(n, n))
    velocity_x = torch.fft.irfft2(1j * ky * psi_ft, s=(n, n))
    velocity_y = torch.fft.irfft2(-1j * kx * psi_ft, s=(n, n))
    omega_x = torch.fft.irfft2(1j * kx * omega_ft, s=(n, n))
    omega_y = torch.fft.irfft2(1j * ky * omega_ft, s=(n, n))
    lap_omega = torch.fft.irfft2(-(kx.square() + ky.square()) * omega_ft, s=(n, n))
    forcing = velocity_x * omega_x + velocity_y * omega_y - 1e-3 * lap_omega
    trajectory = omega.reshape(1, 1, 1, n, n).repeat(1, 1, 3, 1, 1)
    residual = navier_stokes_vorticity_residual(
        trajectory,
        {"nu": 1e-3, "final_time": 1.0, "forcing_field": forcing},
    )
    assert residual.square().mean().sqrt().item() < 1e-8


def test_navier_stokes_default_forcing_uses_endpoint_excluded_generator_grid():
    n = 16
    grid = torch.arange(n, dtype=torch.float64) / n
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    expected = 0.1 * (torch.sin(2.0 * math.pi * (xx + yy)) + torch.cos(2.0 * math.pi * (xx + yy)))
    assert torch.allclose(_ns_forcing(n, n, grid.device, grid.dtype), expected)


def test_two_level_heat_residual_uses_full_endpoint_interval_and_midpoint_state():
    n = 32
    dt = 1.0
    alpha = 1e-3
    x = torch.arange(n, dtype=torch.float64) / n
    _, xx = torch.meshgrid(x, x, indexing="ij")
    u0 = torch.sin(2.0 * math.pi * xx).reshape(1, 1, 1, n, n)
    spectral_eigenvalue = -(2.0 * math.pi) ** 2
    factor = (1.0 + 0.5 * dt * alpha * spectral_eigenvalue) / (
        1.0 - 0.5 * dt * alpha * spectral_eigenvalue
    )
    trajectory = torch.cat([u0, factor * u0], dim=2)
    residual = heat_residual(
        trajectory,
        metadata={"alpha": alpha, "final_time": dt, "time_values": [i / 10 for i in range(11)], "bc": "periodic"},
    )
    assert residual.shape[2] == 1
    assert residual.square().mean().sqrt().item() < 1e-10


def test_displacement_only_wave_trajectory_uses_second_order_equation():
    n = 32
    dt = 0.02
    steps = 9
    x = torch.arange(n, dtype=torch.float64) / n
    xx, _ = torch.meshgrid(x, x, indexing="ij")
    spatial = torch.sin(2.0 * math.pi * xx)
    wavenumber_sq = (2.0 * math.pi) ** 2
    theta = math.acos(1.0 - 0.5 * dt**2 * wavenumber_sq)
    frames = [math.cos(index * theta) * spatial for index in range(steps)]
    trajectory = torch.stack(frames, dim=0).reshape(1, 1, steps, n, n)
    residual = wave_residual(
        trajectory,
        metadata={"fixed_c": 1.0, "final_time": dt * (steps - 1), "bc": "periodic"},
    )
    assert residual.square().mean().sqrt().item() < 1e-9


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


def test_future_scalar_param_missing_returns_nan_with_warning():
    pred = torch.ones(1, 1, 8, 8)
    meta = {"input_fields": pred.clone(), "task": "forward", "final_time": 1.0, "bc": "periodic"}
    with pytest.warns(RuntimeWarning):
        value = pde_residual_metric(pred, "heat", meta)
    assert torch.isnan(value)


def test_steady_heat_conduction_constant_solution_satisfies_zero_source_case():
    solution = torch.full((1, 1, 8, 8), 298.0)
    source = torch.zeros_like(solution)
    u_d = torch.full_like(solution, 298.0)
    meta = {"input_fields": torch.cat([source, u_d], dim=1), "task": "forward", "u_D": 298.0}
    assert pde_residual_metric(solution, "steady_heat_conduction", meta).item() == pytest.approx(0.0)
    assert bc_residual_metric(solution, "steady_heat_conduction", meta).item() == pytest.approx(0.0)
    assert ic_residual_metric(solution, "steady_heat_conduction", meta).item() == pytest.approx(0.0)


def test_steady_heat_conduction_dirichlet_boundary_is_first_spatial_row():
    # This field satisfies bottom-row Dirichlet, top-row Neumann, and side
    # Neumann conditions.  A reversed top/bottom interpretation does not.
    rows = torch.tensor([298.0, 299.0, 300.0, 301.0, 302.0, 303.0, 304.0, 304.0])
    solution = rows.reshape(1, 1, 8, 1).expand(1, 1, 8, 8).clone()
    source = torch.zeros_like(solution)
    meta = {"input_fields": source, "task": "forward", "u_D": 298.0}
    assert bc_residual_metric(solution, "steady_heat_conduction", meta).item() == pytest.approx(0.0)
    solution[..., 0, :] = 299.0
    assert bc_residual_metric(solution, "steady_heat_conduction", meta).item() > 0.0


def test_unobservable_periodic_and_ghost_cell_boundaries_are_labeled():
    for pde, channels in (("burger", 1), ("nsnonbounded", 1), ("reaction_diffusion", 2), ("shallow_water", 3)):
        pred = torch.zeros(1, channels, 8, 8)
        if pde == "shallow_water":
            pred[:, 0] = 1.0
        meta = {"input_fields": pred.clone(), "task": "forward", "final_time": 1.0}
        losses = physics_losses(pred, pde, meta)
        assert losses["bc_status"] != "measured"


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
