from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from baselines.capabilities import paper_table_eligible, resolve_capability
from baselines.common.data_adapter import PDEBatchDataset, build_default_registry, pde_collate
from baselines.methods.official import OfficialImportError
from baselines.methods.vivid import VIVIDBaseline, VIVIDVCNN
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


def test_vivid_burgers_official_core_port_trains_predicts_and_restores_checkpoint(tmp_path):
    batch = _burger_batch()
    cfg = {
        "implementation_mode": "official_architecture",
        "official_backend": "vivid",
        "train_inverse_operator": True,
        "uses_official_inverse_observation_operator": True,
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
        uses_official_inverse_observation_operator=True,
    )

    assert tuple(pred.shape) == tuple(batch.target_fields.shape)
    assert history["train_loss"]
    assert batch.metadata["assimilation_mode"] == "vivid_3dvar_time_space_state"
    assert batch.metadata["inverse_observation_operator_used"] is True
    assert batch.metadata["vivid_objective"] == "Jb_background_plus_Jp_inverse_plus_Jo_observation"
    assert batch.metadata["optimization_optimizer"] == "scipy_L-BFGS-B"
    assert backend["implementation_mode_effective"] == "official_architecture"
    assert backend["official_import_success"] is False
    assert backend["official_reimplementation_success"] is True
    assert "official_architecture" in backend["adapter_status"]
    assert capability.support_status == "official_adapter"
    assert paper_table_eligible(capability, backend_info=backend) is True

    checkpoint = tmp_path / "vivid.pt"
    model.save(checkpoint)
    restored = VIVIDBaseline().build(cfg, build_data_spec(batch)).load(checkpoint)
    assert restored.inverse_operator_trained is True
    assert tuple(restored.predict(batch).shape) == tuple(batch.target_fields.shape)


def test_vivid_vcnn_matches_vendored_layer_recipe_and_shape():
    network = VIVIDVCNN()
    prediction = network(torch.randn(2, 1, 13, 17))
    assert len(network.hidden_layers) == 6
    assert all(layer.conv.out_channels == 48 for layer in network.hidden_layers)
    assert all(layer.conv.kernel_size == (8, 8) for layer in network.hidden_layers)
    assert network.output_layer.conv.out_channels == 1
    assert network.output_layer.conv.kernel_size == (8, 8)
    assert tuple(prediction.shape) == (2, 1, 13, 17)


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
        VIVIDBaseline().build({"implementation_mode": "official_architecture"}, build_data_spec(batch))
