from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from baselines.capabilities import paper_table_eligible, resolve_capability
from baselines.common.data_adapter import PDEBatchDataset, build_default_registry, pde_collate
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


def test_ifno_official_training_adapter_forward_and_inverse_shapes_and_metadata():
    for task in ("forward", "inverse"):
        batch = _batch(task)
        model = IFNOBaseline().build(_cfg(), build_data_spec(batch))
        pred = model.predict(batch)
        assert tuple(pred.shape) == tuple(batch.target_fields.shape)
        backend = _backend_info(model, model.config)
        cap = resolve_capability("ifno", "darcy", task)
        assert backend["implementation_mode_effective"] == "official_aligned"
        assert backend["official_reimplementation_success"] is True
        assert backend["official_alignment_level"] == "algorithm_training"
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


def test_ifno_official_aligned_runs_vae_three_stage_training_and_uses_posterior_mean_for_inverse():
    batch = _batch("forward")
    loader = DataLoader(PDEBatchDataset(batch), batch_size=2, collate_fn=pde_collate)
    model = IFNOBaseline().build(
        _cfg(
            device="cpu",
            normalize=False,
            ifno_pretrain_epochs=1,
            vae_pretrain_epochs=1,
            joint_epochs=1,
            max_steps=1,
            rank=2,
            vae_hidden_dims=[2, 4, 8, 16, 32],
        ),
        build_data_spec(batch),
    )

    history = model.fit(loader)
    prediction = model.predict(_batch("inverse"))

    assert history["training_protocol"] == "official_three_stage"
    assert history["stage_epochs"] == {"ifno_pretrain": 1, "vae_pretrain": 1, "joint_train": 1}
    assert len(history["stage_losses"]["ifno_pretrain"]) == 1
    assert len(history["stage_losses"]["vae_pretrain"]) == 1
    assert len(history["stage_losses"]["joint_train"]) == 1
    assert prediction.shape == batch.input_fields.shape
    backend = _backend_info(model, model.config)
    assert backend["adapter_status"] == "official_training_ifno_task_adapter"
    assert backend["official_alignment_level"] == "algorithm_training"
