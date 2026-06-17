from __future__ import annotations

import pytest

from baselines.common.data_adapter import build_default_registry


CURRENT_PDES = [
    "darcy",
    "poisson",
    "helmholtz",
    "nsnonbounded",
    "burger",
    "reaction_diffusion",
    "shallow_water",
    "heat",
    "wave",
    "advection_diffusion",
    "steady_heat_conduction",
]


@pytest.mark.parametrize("pde", CURRENT_PDES)
def test_adapter_loads_tiny_native_formats(tiny_data_root, pde):
    registry = build_default_registry()
    raw = registry.load_raw(pde, tiny_data_root, split="train", max_samples=2)
    batch = registry.make_task(raw, pde, "forward", num_sensors=5, seed=3)
    assert batch.input_fields.shape[0] == 2
    assert batch.target_fields.shape[0] == 2
    assert batch.full_tensor.shape[0] == 2
    assert batch.mask is not None
    assert batch.obs_values is not None
    assert batch.obs_coords is not None


@pytest.mark.parametrize("pde", ["heat", "wave", "advection_diffusion", "steady_heat_conduction"])
def test_hdf5_pdes_warn_and_synthetic_fallback_when_files_missing(tmp_path, pde):
    registry = build_default_registry()
    with pytest.warns(RuntimeWarning):
        raw = registry.load_raw(pde, tmp_path, synthetic_if_missing=True, max_samples=2, synthetic_resolution=8)
    batch = registry.make_task(raw, pde, "forward")
    assert batch.target_fields.shape[-2:] == (8, 8)
