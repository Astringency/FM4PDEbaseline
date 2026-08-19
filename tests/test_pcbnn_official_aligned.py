from __future__ import annotations

import torch

from baselines.capabilities import paper_table_eligible, resolve_capability
from baselines.common.data_adapter import build_default_registry
from baselines.methods.pc_bnn import PCBNNBaseline, PCBNNParticle, _posterior_result, _run_svgd_optimizer
from baselines.run import _backend_info, build_data_spec
import pytest


def _sparse_batch(pde: str):
    registry = build_default_registry()
    raw = registry.synthetic_raw(pde, n=1, resolution=8)
    return registry.make_task(raw, pde, "sparse_solution", num_sensors=4, seed=1)


def test_pcbnn_does_not_claim_shallow_water_is_the_official_uvp_flow_task():
    cap = resolve_capability("pc_bnn", "shallow_water", "sparse_solution", "random")
    assert cap.paper_table_eligible is False
    assert "u,v,p" in cap.reason


def test_pcbnn_scalar_pde_remains_explicitly_adapted():
    batch = _sparse_batch("poisson")
    cfg = {"implementation_mode": "adapted", "official_backend": "local", "particles": 2, "steps": 1, "hidden": 8}
    model = PCBNNBaseline().build(cfg, build_data_spec(batch))
    backend = _backend_info(model, model.config)
    cap = resolve_capability("pc_bnn", "poisson", "sparse_solution", "random")
    assert cap.support_status == "adapted"
    assert backend["adapter_status"] == "pc_bnn_adapted_reconstruction"
    assert paper_table_eligible(cap, backend_info=backend) is False


@pytest.mark.parametrize("task", ["sparse_forward", "sparse_inverse"])
def test_pcbnn_adapted_static_tasks_return_calibratable_particle_artifacts(task):
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=1, resolution=8)
    batch = registry.make_task(raw, "poisson", task, num_sensors=4, sensor_mode="fixed", seed=1)
    cfg = {
        "implementation_mode": "adapted",
        "official_backend": "local",
        "particles": 2,
        "steps": 1,
        "hidden": 8,
        "depth": 2,
    }
    model = PCBNNBaseline().build(cfg, build_data_spec(batch))

    pred = model.predict(batch)

    assert pred.shape == batch.target_fields.shape
    assert batch.metadata["pc_bnn_joint_field_posterior"] is True
    assert batch.metadata["pc_bnn_posterior_objective"] == (
        "gaussian_observation+student_t_weight_prior+gamma_noise_precision+pde_likelihood"
    )
    assert batch.metadata["posterior_samples"].shape == (
        1,
        cfg["particles"],
        *batch.target_fields.shape[1:],
    )
    assert batch.metadata["predictive_std"].shape == batch.target_fields.shape
    assert torch.isfinite(batch.metadata["predictive_std"]).all()
    assert all(value > 0 for value in batch.metadata["posterior_noise_precision"])
    assert batch.metadata["pc_bnn_training_protocol"] == {
        "particle_optimizer": "adam",
        "weight_lr": cfg.get("lr", 1e-2),
        "noise_lr": cfg.get("lr_noise", 1e-5),
        "initialization": "independent_kaiming_normal",
        "noise_precision_initialization": "gamma_prior",
            "svgd_kernel": "official_rbf_median",
            "equation_precision": 1.0e4,
            "equation_likelihood_reduction": "collocation_sum",
            "boundary_likelihood": "learned_noise_precision",
        }

    backend = _backend_info(model, model.config)
    cap = resolve_capability("pc_bnn", "poisson", task, "fixed")
    assert backend["adapter_status"] == "pc_bnn_adapted_static_pde"
    assert cap.unified_comparison_eligible is True
    assert paper_table_eligible(cap, backend_info=backend) is False


def test_pcbnn_svgd_repulsion_increases_posterior_sample_diversity_without_field_gradients():
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=1, resolution=4)
    batch0 = registry.make_task(raw, "poisson", "sparse_forward", num_sensors=2, sensor_mode="fixed")
    batch1 = registry.make_task(raw, "poisson", "sparse_forward", num_sensors=2, sensor_mode="fixed")
    common = {
        "implementation_mode": "adapted",
        "official_backend": "local",
        "particles": 2,
        "hidden": 4,
        "lr": 1e-3,
        "lambda_obs": 0.0,
        "lambda_int": 0.0,
        "lambda_bc": 0.0,
        "lambda_ic": 0.0,
        "weight_prior_shape": -0.5,
        "beta_prior_shape": 1.0,
        "beta_prior_rate": 0.0,
        "restore_best": False,
    }
    torch.manual_seed(123)
    PCBNNBaseline().build({**common, "steps": 0}, build_data_spec(batch0)).predict(batch0)
    torch.manual_seed(123)
    PCBNNBaseline().build({**common, "steps": 1}, build_data_spec(batch1)).predict(batch1)

    before = batch0.metadata["posterior_samples"][0]
    after = batch1.metadata["posterior_samples"][0]
    before_distance = torch.linalg.vector_norm(before[0] - before[1])
    after_distance = torch.linalg.vector_norm(after[0] - after[1])
    assert after_distance > before_distance


def test_pcbnn_predictive_std_includes_learned_observation_noise():
    particles = [PCBNNParticle(1, 2, 1, initial_noise_precision=4.0) for _ in range(2)]
    predictions = [torch.zeros(1, 1, 1, 1), torch.full((1, 1, 1, 1), 2.0)]

    result = _posterior_result(predictions, particles, {})

    # Particle variance is 1 and E[beta^-1] is 1/4.
    assert torch.allclose(result.std, torch.full_like(result.std, 1.25**0.5))


@pytest.mark.parametrize("task", ["sparse_forward", "sparse_inverse"])
def test_pcbnn_parameter_count_includes_joint_unknown_and_solution_outputs(task):
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=1, resolution=8)
    batch = registry.make_task(raw, "poisson", task, num_sensors=4, sensor_mode="fixed")
    model = PCBNNBaseline().build(
        {"particles": 2, "hidden": 8, "implementation_mode": "official_aligned"},
        build_data_spec(batch),
    )
    proto = model._new_particle(model.coord_dim, model.input_channels + model.target_channels)

    assert model.parameter_count() == 2 * sum(parameter.numel() for parameter in proto.parameters())


def test_pcbnn_svgd_restores_best_particle_ensemble():
    particle = PCBNNParticle(1, 1, 1, initial_noise_precision=1.0)
    with torch.no_grad():
        for parameter in particle.parameters():
            parameter.zero_()
    initial = [parameter.detach().clone() for parameter in particle.parameters()]
    optimizer = torch.optim.Adam(particle.parameters(), lr=2.0)

    def objective(current):
        return sum((parameter - 1.0).square().sum() for parameter in current.parameters())

    status = _run_svgd_optimizer(
        [particle],
        [optimizer],
        1,
        {"restore_best": True, "early_stopping": False},
        objective,
    )

    assert all(torch.allclose(parameter, value) for parameter, value in zip(particle.parameters(), initial))
    assert status["best_step"] == 0
    assert status["restored_best"] is True
