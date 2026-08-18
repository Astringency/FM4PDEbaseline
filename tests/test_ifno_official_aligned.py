from __future__ import annotations

import pytest
import torch

from baselines.capabilities import paper_table_eligible, resolve_capability
from baselines.common.data_adapter import build_default_registry
from baselines.methods.ifno import IFNOBaseline
from baselines.methods.official import OfficialImportError
from baselines.run import _backend_info, build_data_spec


def _batch(task: str):
    registry = build_default_registry()
    raw = registry.synthetic_raw("darcy", n=2, resolution=8)
    return registry.make_task(raw, "darcy", task, num_sensors=4 if task.startswith("sparse") else None, seed=1)


def _cfg(**extra):
    cfg = {"implementation_mode": "official_aligned", "width": 8, "modes1": 4, "modes2": 4, "layers": 1, "epochs": 1}
    cfg.update(extra)
    return cfg


def test_ifno_adapted_reimplementation_forward_and_inverse_shapes_and_metadata():
    for task in ("forward", "inverse"):
        batch = _batch(task)
        model = IFNOBaseline().build(_cfg(), build_data_spec(batch))
        pred = model.predict(batch)
        assert tuple(pred.shape) == tuple(batch.target_fields.shape)
        backend = _backend_info(model, model.config)
        cap = resolve_capability("ifno", "darcy", task)
        assert backend["implementation_mode_effective"] == "adapted"
        assert backend["official_reimplementation_success"] is False
        assert backend["official_alignment_level"] == "concept"
        assert paper_table_eligible(cap, backend_info=backend) is False


def test_ifno_sparse_tasks_remain_unsupported():
    cap = resolve_capability("ifno", "darcy", "sparse_solution", "random")
    assert cap.support_status == "unsupported"
    batch = _batch("sparse_solution")
    model = IFNOBaseline().build(_cfg(), build_data_spec(batch))
    with pytest.raises(NotImplementedError):
        model.predict(batch)


def test_ifno_local_debug_path_never_enters_main_table():
    batch = _batch("forward")
    model = IFNOBaseline().build(_cfg(implementation_mode="adapted", official_backend="local"), build_data_spec(batch))
    backend = _backend_info(model, model.config)
    cap = resolve_capability("ifno", "darcy", "forward")
    assert backend["implementation_mode_effective"] == "adapted"
    assert backend["adapter_status"] == "local_debug_ifno"
    assert paper_table_eligible(cap, backend_info=backend) is False


def test_ifno_strict_official_does_not_fallback_to_aligned():
    batch = _batch("forward")
    with pytest.raises(OfficialImportError):
        IFNOBaseline().build(_cfg(implementation_mode="official", official_backend="ifno"), build_data_spec(batch))
