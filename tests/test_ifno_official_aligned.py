from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import DataLoader

from baselines.capabilities import paper_table_eligible, resolve_capability
from baselines.common.data_adapter import PDEBatchDataset, build_default_registry, pde_collate
from baselines.methods.ifno import (
    IFNOBaseline,
    _ifno_official_stage_order,
    _ifno_stage_early_stopping_settings,
    _ifno_vae_augmentation,
)
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


def test_ifno_official_aligned_runs_vae_three_stage_training_and_uses_posterior_mean_for_inverse(tmp_path):
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
            train_history_jsonl_path=str(tmp_path / "history.jsonl"),
        ),
        build_data_spec(batch),
    )

    history = model.fit(loader)
    prediction = model.predict(_batch("inverse"))

    assert history["training_protocol"] == "official_three_stage"
    assert history["stage_epochs"] == {"ifno_pretrain": 1, "vae_pretrain": 1, "joint_train": 1}
    assert history["stage_order"] == ["ifno_pretrain", "vae_pretrain", "joint_train"]
    assert len(history["stage_losses"]["ifno_pretrain"]) == 1
    assert len(history["stage_losses"]["vae_pretrain"]) == 1
    assert len(history["stage_losses"]["joint_train"]) == 1
    assert history["requested_epochs"] == 3
    assert history["completed_epochs"] == 3
    assert history["completed_stage_epochs"] == {
        "ifno_pretrain": 1,
        "vae_pretrain": 1,
        "joint_train": 1,
    }
    assert prediction.shape == batch.input_fields.shape
    backend = _backend_info(model, model.config)
    assert backend["adapter_status"] == "official_training_ifno_task_adapter"
    assert backend["official_alignment_level"] == "algorithm_training"
    assert model.operator.layers[0].conv12.__class__.__module__.startswith("neuralop.layers")
    stages = [
        json.loads(line)["completed_stage"]
        for line in (tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert stages == ["ifno_pretrain", "vae_pretrain", "joint_train"]


def test_ifno_three_stage_early_stopping_is_independent_and_restores_each_stage(tmp_path):
    batch = _batch("forward")
    loader = DataLoader(PDEBatchDataset(batch), batch_size=2, collate_fn=pde_collate)
    model = IFNOBaseline().build(
        _cfg(
            device="cpu",
            normalize=False,
            ifno_pretrain_epochs=5,
            vae_pretrain_epochs=5,
            joint_epochs=5,
            max_steps=1,
            max_val_steps=1,
            rank=2,
            vae_hidden_dims=[2, 4, 8, 16, 32],
            early_stopping=True,
            restore_best=True,
            ifno_pretrain_early_stopping_patience=1,
            ifno_pretrain_min_epochs=1,
            vae_pretrain_early_stopping_patience=1,
            vae_pretrain_min_epochs=1,
            joint_train_early_stopping_patience=1,
            joint_train_min_epochs=1,
            train_history_jsonl_path=str(tmp_path / "history.jsonl"),
        ),
        build_data_spec(batch),
    )
    model._official_ifno_pretrain_eval_loss = lambda *args, **kwargs: 1.0
    model._official_vae_eval_loss = lambda *args, **kwargs: 1.0
    model._official_ifno_eval_loss = lambda *args, **kwargs: 1.0

    history = model.fit(loader, loader)

    assert history["requested_epochs"] == 15
    assert history["completed_epochs"] == 6
    assert history["completed_stage_epochs"] == {
        "ifno_pretrain": 2,
        "vae_pretrain": 2,
        "joint_train": 2,
    }
    assert history["early_stopped"] is True
    for stage in history["stage_order"]:
        status = history["stage_early_stopping"][stage]
        assert status["monitor_name"] == "val_loss"
        assert status["best_epoch"] == 1
        assert status["stop_epoch"] == 2
        assert status["early_stopped"] is True
        assert status["restored_best"] is True
        assert len(history["stage_val_losses"][stage]) == 2


def test_ifno_stage_early_stopping_defaults_and_overrides_are_auditable():
    config = {"early_stopping": True, "early_stopping_patience": 12, "min_epochs": 5}

    ifno = _ifno_stage_early_stopping_settings(config, "ifno_pretrain")
    vae = _ifno_stage_early_stopping_settings(config, "vae_pretrain")
    joint = _ifno_stage_early_stopping_settings(config, "joint_train")

    assert (ifno["patience"], ifno["min_delta"], ifno["min_epochs"]) == (20, 1e-4, 50)
    assert (vae["patience"], vae["min_delta"], vae["min_epochs"]) == (20, 1e-4, 30)
    assert (joint["patience"], joint["min_delta"], joint["min_epochs"]) == (12, 1e-4, 5)
    overridden = _ifno_stage_early_stopping_settings(
        {
            **config,
            "vae_pretrain_early_stopping_patience": 7,
            "vae_pretrain_early_stopping_min_delta": 0.002,
            "vae_pretrain_min_epochs": 11,
        },
        "vae_pretrain",
    )
    assert (overridden["patience"], overridden["min_delta"], overridden["min_epochs"]) == (
        7,
        0.002,
        11,
    )


def test_ifno_official_stage_order_and_vae_augmentation_match_upstream_scripts():
    assert _ifno_official_stage_order("darcy") == ["ifno_pretrain", "vae_pretrain", "joint_train"]
    assert _ifno_official_stage_order("nsnonbounded") == ["vae_pretrain", "ifno_pretrain", "joint_train"]
    field = torch.arange(2 * 1 * 3 * 3, dtype=torch.float32).reshape(2, 1, 3, 3)

    augmented = _ifno_vae_augmentation(field, enabled=True)

    assert augmented.shape[0] == 4 * field.shape[0]
    assert torch.equal(augmented[:2], field)
    assert torch.equal(augmented[2:4], field.transpose(-2, -1))
