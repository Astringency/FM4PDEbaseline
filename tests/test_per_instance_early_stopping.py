from __future__ import annotations

import pytest

from baselines.common.data_adapter import build_default_registry
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
