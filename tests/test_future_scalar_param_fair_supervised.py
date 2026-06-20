from __future__ import annotations

from baselines.common.data_adapter import build_default_registry


def test_future_materialize_adds_scalar_input_channels(tiny_data_root):
    registry = build_default_registry()
    cases = {
        "heat": 2,
        "advection_diffusion": 4,
        "steady_heat_conduction": 2,
    }
    for pde, expected_channels in cases.items():
        raw = registry.load_raw(pde, tiny_data_root, split="train", max_samples=2, scalar_param_mode="materialize")
        batch = registry.make_task(raw, pde, "forward")
        assert batch.input_fields.shape[1] == expected_channels
        assert any(name in batch.input_channel_names for name in batch.pde_params)
