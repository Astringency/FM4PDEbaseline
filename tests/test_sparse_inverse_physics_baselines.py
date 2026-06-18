from __future__ import annotations

import torch

from baselines.common.data_adapter import build_default_registry
from baselines.run import build_data_spec
from baselines.methods.pde_opt import PDEOptBaseline
from baselines.methods.pinn_sparse import PINNSparseBaseline, sparse_inverse_physics_loss


STATIC_INVERSE_PDES = ("poisson", "helmholtz", "darcy", "steady_heat_conduction")


def _batch(pde: str):
    registry = build_default_registry()
    raw = registry.synthetic_raw(pde, n=1, resolution=8, split="train", seed=5)
    return registry.make_task(raw, pde, "sparse_inverse", num_sensors=4, sensor_mode="random", seed=7)


def test_pde_opt_static_sparse_inverse_tiny_runs():
    for pde in STATIC_INVERSE_PDES:
        batch = _batch(pde)
        model = PDEOptBaseline().build({"steps": 1, "lr": 1e-2}, build_data_spec(batch))
        pred = model.predict(batch)
        assert tuple(pred.shape) == tuple(batch.target_fields.shape)
        assert torch.isfinite(pred).all()
        assert batch.metadata["inference_optimization_time"] >= 0.0
        solution = batch.metadata.get("observed_solution_fields", batch.input_fields)
        loss = sparse_inverse_physics_loss(pred, solution, batch, item=None, config={})
        assert loss.ndim == 0


def test_pinn_sparse_static_sparse_inverse_tiny_runs():
    for pde in STATIC_INVERSE_PDES:
        batch = _batch(pde)
        model = PINNSparseBaseline().build(
            {"official_backend": "local", "steps": 1, "hidden": 8, "depth": 2, "lr": 1e-2},
            build_data_spec(batch),
        )
        pred = model.predict(batch)
        assert tuple(pred.shape) == tuple(batch.target_fields.shape)
        assert torch.isfinite(pred).all()
        assert batch.metadata["inference_optimization_time"] >= 0.0
        solution = batch.metadata.get("observed_solution_fields", batch.input_fields)
        loss = sparse_inverse_physics_loss(pred, solution, batch, item=None, config={})
        assert loss.ndim == 0
