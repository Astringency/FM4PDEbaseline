from __future__ import annotations

from torch.utils.data import DataLoader

from baselines.capabilities import paper_table_eligible, resolve_capability
from baselines.common.data_adapter import PDEBatchDataset, build_default_registry, pde_collate
from baselines.methods.vivid import VIVIDBaseline
from baselines.run import _backend_info, build_data_spec


def _time_varying_batch(pde: str = "reaction_diffusion"):
    registry = build_default_registry()
    raw = registry.synthetic_raw(pde, n=2, resolution=8)
    return registry.make_task(
        raw,
        pde,
        "sparse_solution",
        num_sensors=4,
        sensor_mode="time_varying",
        seed=1,
        experiment_mode="debug",
    )


def test_vivid_official_aligned_time_varying_da_trains_and_predicts():
    batch = _time_varying_batch("reaction_diffusion")
    cfg = {
        "implementation_mode": "official_aligned",
        "official_backend": "vivid",
        "uses_official_inverse_observation_operator": True,
        "epochs": 1,
        "max_steps": 1,
        "refine_steps": 1,
        "inverse_width": 8,
        "inverse_depth": 1,
    }
    model = VIVIDBaseline().build(cfg, build_data_spec(batch))
    loader = DataLoader(PDEBatchDataset(batch), batch_size=1, collate_fn=pde_collate)
    history = model.fit(loader)
    pred = model.predict(batch)
    backend = _backend_info(model, model.config)
    cap = resolve_capability(
        "vivid",
        "reaction_diffusion",
        "sparse_solution",
        "time_varying",
        "time_varying",
        load_full_trajectory=True,
        train_inverse_operator=True,
        uses_official_inverse_observation_operator=True,
    )
    assert tuple(pred.shape) == tuple(batch.target_fields.shape)
    assert batch.metadata["assimilation_mode"] == "full_trajectory"
    assert batch.metadata["inverse_observation_operator_used"] is True
    assert history["inverse_operator_loss"]
    assert "style" not in backend["adapter_status"]
    assert "local" not in backend["adapter_status"]
    assert paper_table_eligible(cap, backend_info=backend) is True


def test_vivid_style_adapted_path_is_supplement_only():
    batch = _time_varying_batch("shallow_water")
    cfg = {"implementation_mode": "adapted", "official_backend": "local", "train_inverse_operator": True, "epochs": 1}
    model = VIVIDBaseline().build(cfg, build_data_spec(batch))
    backend = _backend_info(model, model.config)
    cap = resolve_capability(
        "vivid",
        "shallow_water",
        "sparse_solution",
        "time_varying",
        "time_varying",
        load_full_trajectory=True,
        train_inverse_operator=True,
        uses_official_inverse_observation_operator=False,
    )
    assert backend["implementation_mode_effective"] == "adapted"
    assert paper_table_eligible(cap, backend_info=backend) is False
