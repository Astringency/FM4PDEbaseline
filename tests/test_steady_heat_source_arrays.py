from __future__ import annotations

from baselines.common.data_adapter import build_default_registry


def test_steady_heat_source_arrays_preserve_padded_shape(tiny_data_root):
    registry = build_default_registry()
    raw = registry.load_raw("steady_heat_conduction", tiny_data_root, split="train", max_samples=2)
    batch = registry.make_task(raw, "steady_heat_conduction", "forward")
    assert "u_D" in batch.pde_params
    assert "source_x" not in batch.pde_params
    assert "source_params" in batch.metadata
    assert batch.metadata["source_params"]["source_x"].shape == (2, 3)
    assert batch.metadata["source_params"]["source_y"].shape == (2, 3)
    assert batch.metadata["source_params"]["source_amp"].shape == (2, 3)
    assert batch.metadata["source_params"]["source_sigma"].shape == (2, 3)
