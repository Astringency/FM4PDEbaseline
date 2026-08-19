from __future__ import annotations

import pytest
import torch

from baselines.common.data_adapter import build_default_registry
from baselines.methods.pinn_sparse import observation_loss_from_batch
from baselines.run import BASELINES, _align, build_data_spec


CASES = [
    ("fno", "darcy", "forward"),
    ("deeponet", "poisson", "forward"),
    ("voronoicnn", "darcy", "sparse_solution"),
    ("recfno", "poisson", "sparse_solution"),
    ("senseiver", "shallow_water", "sparse_solution"),
    ("ifno", "helmholtz", "inverse"),
    ("pinn_sparse", "poisson", "sparse_forward"),
    ("pc_bnn", "poisson", "sparse_solution"),
    ("pde_opt", "darcy", "inverse"),
    ("var4d", "burger", "sparse_solution"),
]


@pytest.mark.parametrize("baseline,pde,task", CASES)
def test_baseline_predict_shape(baseline, pde, task):
    torch.manual_seed(1)
    registry = build_default_registry()
    raw = registry.synthetic_raw(pde, n=2, resolution=8)
    batch = registry.make_task(raw, pde, task, num_sensors=6, seed=1)
    cfg = {"width": 8, "modes1": 4, "modes2": 4, "layers": 2, "hidden": 16, "basis": 8, "num_latents": 8, "steps": 1, "refine_steps": 1}
    model = BASELINES[baseline]().build(cfg, build_data_spec(batch))
    with torch.set_grad_enabled(baseline in {"pinn_sparse", "pc_bnn", "pde_opt", "var4d", "vivid"}):
        pred = model.predict(batch)
    assert tuple(pred.shape) == tuple(batch.target_fields.shape)


def test_per_instance_observation_loss_uses_noisy_obs_values():
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=1, resolution=8)
    batch = registry.make_task(raw, "poisson", "sparse_solution", num_sensors=6, seed=4, noise_level=0.0)
    pred = batch.target_fields.clone()
    clean_loss = observation_loss_from_batch(pred, batch)
    batch.obs_values = batch.obs_values + 1.0
    noisy_loss = observation_loss_from_batch(pred, batch)
    assert clean_loss.item() == pytest.approx(0.0)
    assert noisy_loss.item() > 0.0


def test_align_rejects_shape_mismatch_instead_of_cropping():
    pred = torch.randn(2, 1, 128, 128)
    target = torch.randn(2, 1, 1, 128)
    with pytest.raises(ValueError, match="exactly match"):
        _align(pred, target)
