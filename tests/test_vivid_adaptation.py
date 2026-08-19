from __future__ import annotations

import pytest
from torch.utils.data import DataLoader

from baselines.capabilities import paper_table_eligible, resolve_capability
from baselines.common.data_adapter import PDEBatchDataset, build_default_registry, pde_collate
from baselines.methods.official import OfficialImportError
from baselines.methods.vivid import VIVIDBaseline
from baselines.run import _backend_info, build_data_spec


def _burger_batch():
    registry = build_default_registry()
    raw = registry.synthetic_raw("burger", n=2, resolution=8)
    return registry.make_task(
        raw,
        "burger",
        "sparse_solution",
        num_sensors=4,
        sensor_mode="random_per_sample",
        sensor_budget_mode="total",
        seed=1,
        experiment_mode="debug",
        build_voronoi_grid=True,
    )


def test_vivid_burgers_adaptation_trains_and_predicts_without_official_claim():
    batch = _burger_batch()
    cfg = {
        "implementation_mode": "adapted",
        "official_backend": "adapted",
        "train_inverse_operator": True,
        "epochs": 1,
        "max_steps": 1,
        "refine_steps": 1,
        "normalize": False,
    }
    model = VIVIDBaseline().build(cfg, build_data_spec(batch))
    loader = DataLoader(PDEBatchDataset(batch), batch_size=1, collate_fn=pde_collate)
    history = model.fit(loader)
    pred = model.predict(batch)
    backend = _backend_info(model, model.config)
    capability = resolve_capability(
        "vivid",
        "burger",
        "sparse_solution",
        "random_per_sample",
        "time_varying_da_main",
        load_full_trajectory=True,
        train_inverse_operator=True,
        uses_official_inverse_observation_operator=False,
    )

    assert tuple(pred.shape) == tuple(batch.target_fields.shape)
    assert history["train_loss"]
    assert batch.metadata["assimilation_mode"] == "full_trajectory"
    assert batch.metadata["inverse_observation_operator_used"] is True
    assert backend["implementation_mode_effective"] == "adapted"
    assert backend["official_import_success"] is False
    assert backend["official_reimplementation_success"] is False
    assert "adapted" in backend["adapter_status"]
    assert capability.support_status == "adapted"
    assert paper_table_eligible(capability, backend_info=backend) is False


def test_vivid_strict_official_mode_fails_instead_of_using_local_adapter():
    batch = _burger_batch()
    with pytest.raises(OfficialImportError):
        VIVIDBaseline().build(
            {"implementation_mode": "official", "official_backend": "vivid"},
            build_data_spec(batch),
        )


def test_vivid_rejects_removed_non_burgers_entry():
    registry = build_default_registry()
    raw = registry.synthetic_raw("nsnonbounded", n=1, resolution=8)
    batch = registry.make_task(raw, "nsnonbounded", "sparse_solution", num_sensors=4)
    with pytest.raises(ValueError, match="only to Burgers"):
        VIVIDBaseline().build({"implementation_mode": "adapted"}, build_data_spec(batch))
