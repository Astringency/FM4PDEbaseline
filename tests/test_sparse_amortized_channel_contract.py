from __future__ import annotations

import pytest

from baselines.common.data_adapter import build_default_registry, pde_collate
from baselines.methods.recfno import RecFNOBaseline
from baselines.methods.senseiver import SenseiverBaseline
from baselines.methods.voronoicnn import VoronoiCNNBaseline
from baselines.run import build_data_spec


METHODS = {
    "recfno": (
        RecFNOBaseline,
        {
            "implementation_mode": "adapted",
            "official_backend": "local",
            "width": 4,
            "modes1": 2,
            "modes2": 2,
        },
    ),
    "senseiver": (
        SenseiverBaseline,
        {
            "implementation_mode": "adapted",
            "official_backend": "local",
            "space_bands": 2,
            "enc_preproc_ch": 8,
            "num_latents": 2,
            "enc_num_latent_channels": 4,
            "num_layers": 1,
            "num_cross_attention_heads": 1,
            "enc_num_self_attention_heads": 1,
            "num_self_attention_layers_per_block": 1,
            "dec_num_latent_channels": 4,
            "dec_num_cross_attention_heads": 1,
        },
    ),
    "voronoicnn": (
        VoronoiCNNBaseline,
        {
            "implementation_mode": "adapted",
            "official_backend": "local",
            "width": 4,
        },
    ),
}


@pytest.mark.parametrize("task", ["sparse_forward", "sparse_inverse"])
@pytest.mark.parametrize("method_name", list(METHODS))
def test_ns_full_trajectory_sparse_methods_support_unequal_input_and_target_channels(method_name, task):
    registry = build_default_registry()
    raw = registry.synthetic_raw("nsnonbounded", n=2, resolution=4, split="train")
    batch = registry.make_task(
        raw,
        "nsnonbounded",
        task,
        num_sensors=3,
        sensor_mode="random_per_sample",
        seed=7,
        experiment_mode="paper",
    )
    dataset = registry.make_dataset_from_batch(batch)
    materialized = pde_collate([dataset[index] for index in range(len(dataset))])
    model_cls, config = METHODS[method_name]
    model = model_cls().build(config, build_data_spec(materialized))

    prediction = model.predict(materialized)

    assert materialized.input_fields.shape[1] != materialized.target_fields.shape[1]
    assert prediction.shape == materialized.target_fields.shape
