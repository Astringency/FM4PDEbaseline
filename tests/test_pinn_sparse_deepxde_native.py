from __future__ import annotations

import pytest
import torch

from baselines.common.data_adapter import build_default_registry
from baselines.methods.pinn_sparse import PINNSparseBaseline, _deepxde_static_pde
from baselines.run import _backend_info, build_data_spec


@pytest.mark.parametrize("pde", ["poisson", "helmholtz", "darcy"])
@pytest.mark.parametrize("task", ["sparse_forward", "sparse_inverse"])
def test_deepxde_native_static_pde_uses_sparse_observations_and_returns_requested_field(pde, task):
    registry = build_default_registry()
    raw = registry.synthetic_raw(pde, n=1, resolution=4, seed=4)
    batch = registry.make_task(raw, pde, task, num_sensors=3, sensor_mode="fixed", seed=2)
    model = PINNSparseBaseline().build(
        {
            "implementation_mode": "official_aligned",
            "official_backend": "deepxde",
            "hidden": 8,
            "depth": 2,
            "steps": 1,
            "lbfgs_steps": 0,
            "num_domain": 8,
            "num_boundary": 4,
            "seed": 3,
        },
        build_data_spec(batch),
    )

    prediction = model.predict(batch)

    assert prediction.shape == batch.target_fields.shape
    assert torch.isfinite(prediction).all()
    assert batch.metadata["pinn_sparse_training_protocol"]["data"] == "dde.data.PDE"
    assert batch.metadata["pinn_sparse_training_protocol"]["observation_bc"] == "dde.icbc.PointSetBC"
    assert batch.metadata["pinn_sparse_training_protocol"]["derivatives"] == "deepxde_autodiff"
    assert batch.metadata["pinn_sparse_training_protocol"]["network_outputs"] == ["solution", "unknown"]
    expected_boundary = "generator_aligned_kronecker" if pde == "helmholtz" else "zero_dirichlet"
    assert batch.metadata["pinn_sparse_training_protocol"]["boundary_conditions"] == expected_boundary
    backend = _backend_info(model, model.config)
    assert backend["adapter_status"] == "deepxde_native_task_adapter"
    assert backend["official_alignment_level"] == "training_api"


def test_deepxde_poisson_residual_respects_dataset_operator_sign():
    class _Grad:
        @staticmethod
        def hessian(y, x, component, i, j):
            del x, component, i, j
            return torch.ones_like(y[:, :1])

    class _DDE:
        grad = _Grad()

    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=1, resolution=4, seed=4)
    batch = registry.make_task(raw, "poisson", "sparse_forward", num_sensors=3, seed=2)
    batch.metadata["elliptic_operator_sign"] = -1.0
    residual = _deepxde_static_pde(_DDE(), "poisson", batch, 0, {})
    x = torch.zeros(3, 2)
    y = torch.cat((torch.zeros(3, 1), torch.full((3, 1), 0.5)), dim=1)

    assert torch.allclose(residual(x, y), torch.full((3, 1), -2.5))
