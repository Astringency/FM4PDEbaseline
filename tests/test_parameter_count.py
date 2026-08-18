from __future__ import annotations

import torch

from baselines.common.data_adapter import build_default_registry
from baselines.methods.base import BaselineModel
from baselines.methods.pde_opt import PDEOptBaseline
from baselines.methods.pinn_sparse import PINNSparseBaseline
from baselines.run import build_data_spec


class _ComplexModel(BaselineModel):
    def __init__(self) -> None:
        super().__init__()
        self.real = torch.nn.Parameter(torch.zeros(3))
        self.complex = torch.nn.Parameter(torch.zeros(5, dtype=torch.cfloat))


def test_parameter_count_reports_real_scalar_degrees_of_freedom():
    model = _ComplexModel()
    assert model.parameter_storage_count() == 8
    assert model.parameter_count() == 13


def test_per_instance_sparse_counts_include_unknown_and_solution_fields():
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=1, resolution=8)
    batch = registry.make_task(
        raw,
        "poisson",
        "sparse_forward",
        num_sensors=8,
        sensor_mode="random_per_sample",
        seed=1,
    )
    spec = build_data_spec(batch)

    pde_opt = PDEOptBaseline().build({}, spec)
    assert pde_opt.parameter_count() == spec["input_numel"] + spec["target_numel"]

    pinn = PINNSparseBaseline().build(
        {"implementation_mode": "adapted", "official_backend": "local", "hidden": 8, "depth": 2},
        spec,
    )
    unknown = pinn._new_field(pinn.coord_dim, pinn.target_channels)
    solution = pinn._new_field(pinn.coord_dim, pinn.input_channels)
    expected = sum(parameter.numel() for parameter in (*unknown.parameters(), *solution.parameters()))
    assert pinn.parameter_count() == expected
