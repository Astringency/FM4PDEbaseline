from __future__ import annotations

from baselines.capabilities import paper_table_eligible, resolve_capability
from baselines.common.data_adapter import build_default_registry
from baselines.methods.pc_bnn import PCBNNBaseline
from baselines.run import _backend_info, build_data_spec


def _sparse_batch(pde: str):
    registry = build_default_registry()
    raw = registry.synthetic_raw(pde, n=1, resolution=8)
    return registry.make_task(raw, pde, "sparse_solution", num_sensors=4, seed=1)


def test_pcbnn_official_aligned_for_matched_shallow_water_field():
    batch = _sparse_batch("shallow_water")
    cfg = {"implementation_mode": "official_aligned", "particles": 2, "steps": 1, "hidden": 8, "depth": 3}
    model = PCBNNBaseline().build(cfg, build_data_spec(batch))
    pred = model.predict(batch)
    backend = _backend_info(model, model.config)
    cap = resolve_capability("pc_bnn", "shallow_water", "sparse_solution", "random")
    assert tuple(pred.shape) == tuple(batch.target_fields.shape)
    assert backend["implementation_mode_effective"] == "official_aligned"
    assert backend["official_reimplementation_success"] is True
    assert paper_table_eligible(cap, backend_info=backend) is True


def test_pcbnn_scalar_pde_remains_supplement_only():
    batch = _sparse_batch("poisson")
    cfg = {"implementation_mode": "adapted", "official_backend": "local", "particles": 2, "steps": 1, "hidden": 8}
    model = PCBNNBaseline().build(cfg, build_data_spec(batch))
    backend = _backend_info(model, model.config)
    cap = resolve_capability("pc_bnn", "poisson", "sparse_solution", "random")
    assert cap.support_status == "adapted"
    assert "generic_svgd" in backend["adapter_status"]
    assert paper_table_eligible(cap, backend_info=backend) is False
