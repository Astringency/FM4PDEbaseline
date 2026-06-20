from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from baselines.common.data_adapter import PDEBatchDataset, build_default_registry, pde_collate
from baselines.common.metrics import physics_loss_metric
from baselines.methods.vivid import VIVIDBaseline, _trajectory_for_inverse_operator_physics
from baselines.run import build_data_spec


def _ns_time_varying_batch():
    registry = build_default_registry()
    raw = registry.synthetic_raw("nsnonbounded", n=2, resolution=8)
    return registry.make_task(
        raw,
        "nsnonbounded",
        "sparse_solution",
        num_sensors=4,
        sensor_mode="time_varying",
        seed=2,
        experiment_mode="debug",
    )


def test_vivid_ns_future_trajectory_inserted_for_initialization():
    batch = _ns_time_varying_batch()
    cfg = {
        "implementation_mode": "official_aligned",
        "official_backend": "vivid",
        "uses_official_inverse_observation_operator": True,
        "epochs": 1,
        "max_steps": 1,
        "refine_steps": 1,
        "inverse_width": 8,
        "inverse_depth": 1,
        "lambda_pde": 0.0,
    }
    model = VIVIDBaseline().build(cfg, build_data_spec(batch))
    loader = DataLoader(PDEBatchDataset(batch), batch_size=1, collate_fn=pde_collate)
    model.fit(loader)
    pred = model.predict(batch)
    assert batch.metadata["learned_state_injection_mode"] == "future_trajectory_inserted"
    assert tuple(pred.shape) == tuple(batch.target_fields.shape)
    assert batch.metadata["assimilation_mode"] == "full_trajectory"


def test_vivid_ns_inverse_physics_prepends_initial_frame():
    batch = _ns_time_varying_batch()
    future_only = batch.target_fields.clone()
    pred_for_physics, mode = _trajectory_for_inverse_operator_physics(future_only, batch)
    assert mode == "initial_prepended"
    assert pred_for_physics.shape[2] == future_only.shape[2] + 1
    torch.testing.assert_close(pred_for_physics[:, :, 0], batch.full_tensor[:, :, 0])
    metrics = physics_loss_metric(pred_for_physics, "nsnonbounded", {**batch.metadata, "full_tensor": batch.full_tensor})
    assert metrics["mode"] == "full_trajectory"


def test_vivid_ns_training_history_records_full_physics_trajectory():
    batch = _ns_time_varying_batch()
    cfg = {
        "implementation_mode": "official_aligned",
        "official_backend": "vivid",
        "uses_official_inverse_observation_operator": True,
        "epochs": 1,
        "max_steps": 1,
        "refine_steps": 0,
        "inverse_width": 8,
        "inverse_depth": 1,
    }
    model = VIVIDBaseline().build(cfg, build_data_spec(batch))
    loader = DataLoader(PDEBatchDataset(batch), batch_size=1, collate_fn=pde_collate)
    history = model.fit(loader)
    assert history["physics_trajectory_mode"] == ["initial_prepended"]
    assert history["physics_loss_mode"] == ["full_trajectory"]
    assert history["physics_trajectory_shape"][0][2] == batch.target_fields.shape[2] + 1
