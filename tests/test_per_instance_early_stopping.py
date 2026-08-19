from __future__ import annotations

import pytest
import torch

from baselines.common.data_adapter import build_default_registry
from baselines.methods.base import run_per_instance_optimizer
from baselines.methods.pc_bnn import PCBNNBaseline
from baselines.methods.pde_opt import PDEOptBaseline
from baselines.methods.pinn_sparse import PINNSparseBaseline
from baselines.run import build_data_spec


@pytest.mark.parametrize(
    ("model_cls", "extra"),
    [
        (PINNSparseBaseline, {"implementation_mode": "adapted", "official_backend": "local"}),
        (PDEOptBaseline, {}),
        (
            PCBNNBaseline,
            {"implementation_mode": "adapted", "official_backend": "local", "particles": 2},
        ),
    ],
)
def test_per_instance_sparse_optimizers_stop_when_loss_is_flat(model_cls, extra):
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=1, resolution=4)
    batch = registry.make_task(raw, "poisson", "sparse_forward", num_sensors=3, sensor_mode="fixed")
    config = {
        "steps": 20,
        "lr": 0.0,
        "hidden": 8,
        "depth": 2,
        "early_stopping": True,
        "early_stopping_patience": 2,
        "early_stopping_min_delta": 1e-4,
        **extra,
    }
    model = model_cls().build(config, build_data_spec(batch))

    prediction = model.predict(batch)

    assert prediction.shape == batch.target_fields.shape
    assert batch.metadata["optimization_early_stopped"] is True
    assert batch.metadata["optimization_steps_completed"] < config["steps"]


@pytest.mark.parametrize("optimizer_name", ["sgd", "lbfgs"])
def test_per_instance_optimizer_restores_the_best_post_update_parameters(optimizer_name):
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    if optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS([parameter], lr=1.0, max_iter=8)
    else:
        optimizer = torch.optim.SGD([parameter], lr=1.0)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = (parameter - 1.0).square()
        loss.backward()
        return loss

    status = run_per_instance_optimizer(
        optimizer,
        closure,
        steps=1,
        config={"early_stopping": False, "restore_best": True},
    )

    expected_parameter = 1.0 if optimizer_name == "lbfgs" else 0.0
    expected_best_step = 1 if optimizer_name == "lbfgs" else 0
    # SGD moves 0 -> 2 and ties the initial loss, whereas one bounded LBFGS
    # outer step reaches the minimizer.  Both must restore the best evaluated
    # state and report the loss at that state, not LBFGS.step's stale return.
    assert parameter.item() == pytest.approx(expected_parameter)
    assert status["completed_steps"] == 1
    assert status["best_step"] == expected_best_step
    assert status["best_loss"] == pytest.approx(0.0 if optimizer_name == "lbfgs" else 1.0)
    assert status["restored_best"] is True
