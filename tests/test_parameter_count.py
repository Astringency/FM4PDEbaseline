from __future__ import annotations

import torch

from baselines.methods.base import BaselineModel


class _ComplexModel(BaselineModel):
    def __init__(self) -> None:
        super().__init__()
        self.real = torch.nn.Parameter(torch.zeros(3))
        self.complex = torch.nn.Parameter(torch.zeros(5, dtype=torch.cfloat))


def test_parameter_count_reports_real_scalar_degrees_of_freedom():
    model = _ComplexModel()
    assert model.parameter_storage_count() == 8
    assert model.parameter_count() == 13
