from __future__ import annotations

import torch

from baselines.capabilities import paper_table_eligible, resolve_capability
from baselines.common.data_adapter import build_default_registry
from baselines.methods.pc_bnn import PCBNNBaseline
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


def test_pcbnn_scalar_pde_remains_supplement_only():
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

    backend = _backend_info(model, model.config)
    cap = resolve_capability("pc_bnn", "poisson", task, "fixed")
    assert backend["adapter_status"] == "pc_bnn_adapted_static_pde"
    assert cap.unified_comparison_eligible is True
    assert paper_table_eligible(cap, backend_info=backend) is False
