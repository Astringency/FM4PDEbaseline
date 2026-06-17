from __future__ import annotations

import pytest
import torch

from baselines.common.data_adapter import build_default_registry
from baselines.run import BASELINES, build_data_spec


CASES = [
    ("fno", "darcy", "forward"),
    ("deeponet", "poisson", "forward"),
    ("voronoicnn", "darcy", "sparse_solution"),
    ("recfno", "poisson", "sparse_solution"),
    ("senseiver", "shallow_water", "sparse_solution"),
    ("ifno", "helmholtz", "inverse"),
    ("pinn_sparse", "burger", "sparse_solution"),
    ("pc_bnn", "poisson", "sparse_solution"),
    ("pde_opt", "darcy", "inverse"),
    ("var4d", "shallow_water", "sparse_solution"),
    ("vivid", "shallow_water", "sparse_solution"),
]


@pytest.mark.parametrize("baseline,pde,task", CASES)
def test_baseline_predict_shape(baseline, pde, task):
    torch.manual_seed(1)
    registry = build_default_registry()
    raw = registry.synthetic_raw(pde, n=2, resolution=8)
    batch = registry.make_task(raw, pde, task, num_sensors=6, seed=1)
    cfg = {"width": 8, "modes1": 4, "modes2": 4, "layers": 2, "hidden": 16, "basis": 8, "token_dim": 16, "num_latents": 8, "heads": 2, "steps": 1, "refine_steps": 1}
    model = BASELINES[baseline]().build(cfg, build_data_spec(batch))
    with torch.set_grad_enabled(baseline in {"pinn_sparse", "pc_bnn", "pde_opt", "var4d", "vivid"}):
        pred = model.predict(batch)
    assert tuple(pred.shape) == tuple(batch.target_fields.shape)

