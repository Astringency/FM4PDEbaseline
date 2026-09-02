from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from baselines.common.data_adapter import PDEBatchDataset, build_default_registry, pde_collate
from baselines.common.normalization import estimate_normalization_stats, normalize_batch_input_target
from baselines.common.sensors import (
    build_multicondition_observation_tensors,
    deterministic_condition_mode,
    validate_condition_probabilities,
)
from baselines.methods.recfno import RecFNOBaseline
from baselines.methods.senseiver import SenseiverBaseline
from baselines.methods.voronoicnn import VoronoiCNNBaseline
from baselines.run import _make_inference_batch, build_data_spec, load_baseline_checkpoint, main as run_main
from scripts.build_sparse_solution_multicondition_report import (
    load_evaluation_summaries,
    validate_and_build_rows,
    write_report,
)
from scripts.experiments.build_matrix import build_matrix, load_config
from scripts.experiments import run_one as run_one_module


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("a_only", "u_only", "both")
MODEL_CONFIGS = {
    "recfno": (
        RecFNOBaseline,
        {
            "implementation_mode": "adapted",
            "official_backend": "local",
            "width": 4,
            "modes1": 2,
            "modes2": 2,
            "training_loss": "l1",
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
            "training_loss": "mse",
        },
    ),
    "voronoicnn": (
        VoronoiCNNBaseline,
        {
            "implementation_mode": "adapted",
            "official_backend": "local",
            "width": 4,
            "training_loss": "mse",
        },
    ),
}


def _dataset(
    *, split: str = "train", n: int = 4, condition_mode: str = "mixed"
) -> PDEBatchDataset:
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=n, resolution=8, split=split)
    batch = registry.make_task(
        raw,
        "poisson",
        "sparse_solution_multicondition",
        num_sensors=5,
        sensor_mode="random_per_sample",
        sensor_budget_mode="total",
        seed=17,
        condition_mode=condition_mode,
        experiment_mode="paper",
    )
    return PDEBatchDataset(batch)


def _view(dataset: PDEBatchDataset, condition: str, epoch: int = 0):
    dataset.set_epoch(epoch)
    dataset.set_condition_mode(condition)
    return pde_collate([dataset[index] for index in range(len(dataset))])


@pytest.mark.parametrize("condition", CONDITIONS)
def test_multicondition_data_contract_and_sensor_counts(condition):
    dataset = _dataset(split="test")
    batch = _view(dataset, condition)
    split = int(batch.metadata["joint_input_channels"])
    assert split == 1
    assert batch.input_fields.shape == batch.target_fields.shape == (4, 2, 8, 8)
    assert batch.obs_values.shape == batch.metadata["sensor_presence"].shape == (4, 5, 2)
    for private_truth_key in (
        "full_tensor",
        "full_trajectory",
        "original_input_fields",
        "observation_source_fields",
        "background_fields",
        "solution_fields",
        "source_fields",
        "coeff_fields",
    ):
        assert private_truth_key not in batch.metadata

    expected = {
        "a_only": (5, 0, 5),
        "u_only": (0, 5, 5),
        "both": (5, 5, 10),
    }[condition]
    assert batch.metadata["num_sensor_locations"] == 5
    assert (
        batch.metadata["num_scalar_observations_a"],
        batch.metadata["num_scalar_observations_u"],
        batch.metadata["num_scalar_observations_total"],
    ) == expected
    if condition == "a_only":
        assert torch.count_nonzero(batch.mask[:, split:]) == 0
        assert torch.count_nonzero(batch.input_fields[:, split:]) == 0
        assert torch.count_nonzero(batch.metadata["voronoi_grid"][:, split:]) == 0
        assert torch.equal(batch.metadata["sensor_presence"][..., split:], torch.zeros_like(batch.obs_values[..., split:]))
    elif condition == "u_only":
        assert torch.count_nonzero(batch.mask[:, :split]) == 0
        assert torch.count_nonzero(batch.input_fields[:, :split]) == 0
        assert torch.count_nonzero(batch.metadata["voronoi_grid"][:, :split]) == 0
        assert torch.equal(batch.metadata["sensor_presence"][..., :split], torch.zeros_like(batch.obs_values[..., :split]))
    else:
        assert torch.equal(batch.mask[:, :split], batch.mask[:, split:])


def test_multicondition_normalizes_complete_training_fields_then_masks():
    train = _dataset(split="train")
    train.set_condition_mode(None)
    loader = DataLoader(train, batch_size=2, shuffle=False, collate_fn=pde_collate)
    stats = estimate_normalization_stats(loader)
    full_target = train.batch.target_fields
    assert stats.target_mean.tolist() == pytest.approx(
        full_target.mean(dim=(0, 2, 3)).tolist()
    )
    assert torch.equal(stats.input_mean, stats.target_mean)
    assert torch.equal(stats.input_std, stats.target_std)

    batch = _view(train, "a_only")
    normalized = normalize_batch_input_target(batch, stats)
    assert torch.count_nonzero(normalized.input_fields[:, 1:]) == 0
    assert torch.count_nonzero(normalized.metadata["voronoi_grid"][:, 1:]) == 0
    assert torch.count_nonzero(normalized.obs_values[..., 1:]) == 0
    assert torch.count_nonzero(normalized.metadata["sensor_presence"][..., 1:]) == 0
    observed = normalized.mask.bool()
    assert torch.allclose(normalized.input_fields[observed], normalized.target_fields[observed])
    assert normalized.metadata["normalization_order"] == "full_joint_field_then_active_mask"


def test_unobserved_truth_cannot_change_model_inputs():
    fields = torch.arange(2 * 2 * 4 * 4, dtype=torch.float32).reshape(2, 2, 4, 4)
    first = build_multicondition_observation_tensors(
        fields,
        num_sensors=3,
        mode="random_per_sample",
        seed=8,
        condition_modes=["a_only", "u_only"],
        sensor_budget_mode="total",
        sample_ids=["s0", "s1"],
        split="test",
    )
    changed = fields.clone()
    changed[~first["base_mask"].bool()] += 100000.0
    second = build_multicondition_observation_tensors(
        changed,
        num_sensors=3,
        mode="random_per_sample",
        seed=8,
        condition_modes=["a_only", "u_only"],
        sensor_budget_mode="total",
        sample_ids=["s0", "s1"],
        split="test",
    )
    for key in ("mask", "base_mask", "obs_values", "masked_grid", "voronoi_grid", "obs_presence"):
        assert torch.equal(first[key], second[key])

    inference = _make_inference_batch(_view(_dataset(split="test"), "both"))
    assert torch.count_nonzero(inference.target_fields) == 0
    assert torch.count_nonzero(inference.full_tensor) == 0
    assert inference.metadata["inference_truth_hidden"] is True


def test_multicondition_reproducibility_across_epochs_splits_views_and_workers():
    assert deterministic_condition_mode(3, "train", "sample", 2) == deterministic_condition_mode(
        3, "train", "sample", 2
    )
    assert deterministic_condition_mode(3, "test", "sample", 2) == deterministic_condition_mode(
        3, "test", "sample", 99
    )

    train = _dataset(split="train", n=8)
    train.set_condition_mode(None)
    train.set_epoch(0)
    epoch0 = [train[index] for index in range(len(train))]
    assert len({item.metadata["condition_mode"] for item in epoch0}) > 1
    train.set_epoch(1)
    epoch1 = [train[index] for index in range(len(train))]
    assert any(not torch.equal(left.mask, right.mask) for left, right in zip(epoch0, epoch1))

    test = _dataset(split="test", n=4)
    views = {condition: _view(test, condition, epoch=11) for condition in CONDITIONS}
    for condition in CONDITIONS[1:]:
        assert torch.equal(views["a_only"].metadata["base_mask"], views[condition].metadata["base_mask"])
        assert views["a_only"].metadata["base_mask_ids"] == views[condition].metadata["base_mask_ids"]
    view_late_epoch = _view(test, "a_only", epoch=999)
    assert torch.equal(views["a_only"].metadata["base_mask"], view_late_epoch.metadata["base_mask"])

    def collect(num_workers: int):
        worker_dataset = _dataset(split="train", n=6)
        worker_dataset.set_epoch(4)
        loader = DataLoader(
            worker_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=pde_collate,
        )
        return [
            (batch.metadata["condition_modes"], batch.metadata["base_mask_ids"])
            for batch in loader
        ]

    assert collect(0) == collect(2)


@pytest.mark.parametrize("method_name", list(MODEL_CONFIGS))
def test_three_baselines_train_once_and_reuse_one_checkpoint(method_name, tmp_path):
    train = _dataset(split="train", n=4)
    val = _dataset(split="val", n=2)
    train.set_condition_mode(None)
    train_loader = DataLoader(train, batch_size=2, shuffle=False, collate_fn=pde_collate)
    val_loader = DataLoader(val, batch_size=2, shuffle=False, collate_fn=pde_collate)
    spec_batch = _view(train, "both")
    train.set_condition_mode(None)
    model_cls, base_config = MODEL_CONFIGS[method_name]
    config = {
        **base_config,
        "device": "cpu",
        "epochs": 1,
        "max_train_steps": 1,
        "max_val_steps": 1,
        "normalize": True,
    }
    model = model_cls().build(config, build_data_spec(spec_batch))
    history = model.fit(train_loader, val_loader)
    assert len(history["train_loss"]) == 1
    assert set(history["val_loss_by_condition"][0]) == set(CONDITIONS)
    assert all(torch.isfinite(torch.tensor(value)) for value in history["train_loss"])

    checkpoint = tmp_path / f"{method_name}.pt"
    model.save(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    loaded = load_baseline_checkpoint(checkpoint, baseline=method_name)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    for condition in CONDITIONS:
        batch = _view(val, condition)
        prediction = loaded.predict_physical(batch)
        assert prediction.shape == batch.target_fields.shape
        assert torch.isfinite(prediction).all()
    assert loaded.backend_metadata()["adapter_status"] == "multicondition_task_adapter"


def test_burgers_is_rejected_without_fallback():
    registry = build_default_registry()
    raw = registry.synthetic_raw("burger", n=2, resolution=8)
    with pytest.raises(
        ValueError,
        match="sparse_solution_multicondition does not support Burgers trajectory semantics",
    ):
        registry.make_task(raw, "burger", "sparse_solution_multicondition", num_sensors=5)
    with pytest.raises(
        ValueError,
        match="sparse_solution_multicondition does not support Burgers trajectory semantics",
    ):
        run_main(
            [
                "--baseline",
                "recfno",
                "--pde",
                "burger",
                "--task",
                "sparse_solution_multicondition",
            ]
        )


def test_condition_probabilities_are_strictly_validated():
    assert sum(validate_condition_probabilities(None).values()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="non-negative"):
        validate_condition_probabilities({"a_only": -0.1, "u_only": 0.5, "both": 0.6})
    with pytest.raises(ValueError, match="sum to 1"):
        validate_condition_probabilities({"a_only": 0.2, "u_only": 0.2, "both": 0.2})


def test_ablation_matrix_is_12_train_rows_plus_36_eval_views(tmp_path):
    config_path = ROOT / "configs/experiments/sparse_solution_multicondition_ablation.yaml"
    config = load_config(config_path)
    rows, skipped, _summary = build_matrix(
        config,
        tmp_path / "matrix-output",
        "sparse_solution_multicondition_ablation",
        experiment_config_path=config_path,
    )
    train_rows = [row for row in rows if row["execution_mode"] == "train"]
    eval_rows = [row for row in rows if row["execution_mode"] == "eval_only"]
    assert not skipped
    assert len(train_rows) == 12
    assert len(eval_rows) == 36
    assert {row["condition_mode"] for row in train_rows} == {"mixed"}
    assert {row["condition_mode"] for row in eval_rows} == set(CONDITIONS)
    assert all(row["train_only"] for row in train_rows)
    assert all(not row["train_only"] for row in eval_rows)
    assert len({row["run_id"] for row in train_rows}) == 12
    assert len({row["source_train_run_id"] for row in eval_rows}) == 12
    assert {row["pde"] for row in rows} == {"poisson", "helmholtz", "darcy", "nsnonbounded"}
    assert {row["baseline"] for row in rows} == {"recfno", "senseiver", "voronoicnn"}

    main_text = (ROOT / "configs/experiments/main_results.yaml").read_text(encoding="utf-8")
    assert "sparse_solution_multicondition" not in main_text


def test_matrix_commands_separate_training_from_checkpoint_only_evaluation(
    tmp_path, monkeypatch
):
    config_path = ROOT / "configs/experiments/sparse_solution_multicondition_ablation.yaml"
    rows, _skipped, _summary = build_matrix(
        load_config(config_path),
        tmp_path / "matrix-output",
        "sparse_solution_multicondition_ablation",
        experiment_config_path=config_path,
    )
    train_row = next(row for row in rows if row["execution_mode"] == "train")
    eval_row = next(row for row in rows if row["execution_mode"] == "eval_only")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(run_one_module, "_validate_matrix_provenance", lambda *_args, **_kwargs: None)

    train_command = run_one_module.build_command(train_row)
    eval_command = run_one_module.build_command(eval_row)
    assert "--train-only" in train_command
    assert "--eval-only" not in train_command
    assert "--save-checkpoint" in train_command
    assert train_command[train_command.index("--condition-mode") + 1] == "mixed"
    assert "--eval-only" in eval_command
    assert "--train-only" not in eval_command
    assert "--no-save-checkpoint" in eval_command
    assert eval_command[eval_command.index("--checkpoint") + 1] == eval_row["checkpoint_path"]
    assert eval_command[eval_command.index("--condition-mode") + 1] in CONDITIONS


def _fake_eval_summary(condition: str) -> dict[str, object]:
    active_a = condition in {"a_only", "both"}
    active_u = condition in {"u_only", "both"}
    locations = 5
    return {
        "status": "success",
        "execution_mode": "eval_only",
        "task": "sparse_solution_multicondition",
        "pde": "poisson",
        "baseline": "recfno",
        "seed": 1,
        "condition_mode": condition,
        "evaluation_condition_mode": condition,
        "checkpoint_path": "/tmp/shared.pt",
        "checkpoint_sha256": "a" * 64,
        "train_run_fingerprint": "b" * 64,
        "normalization_stats_sha256": "c" * 64,
        "test_sample_set_sha256": "d" * 64,
        "base_mask_manifest_sha256": "e" * 64,
        "test_size": 2,
        "num_sensor_locations": locations,
        "num_scalar_observations_a": locations if active_a else 0,
        "num_scalar_observations_u": locations if active_u else 0,
        "num_scalar_observations_total": locations * (int(active_a) + int(active_u)),
        "rel_l2_a_mean": 0.1,
        "rel_l2_a_median": 0.1,
        "rel_l2_a_p90": 0.1,
        "rel_l2_u_mean": 0.2,
        "rel_l2_u_median": 0.2,
        "rel_l2_u_p90": 0.2,
        "joint_rel_l2_mean": 0.15,
        "joint_rel_l2_median": 0.15,
        "joint_rel_l2_p90": 0.15,
        "observed_mse_a_mean": 0.01 if active_a else float("nan"),
        "observed_mse_u_mean": 0.02 if active_u else float("nan"),
    }


def test_independent_report_validates_provenance_and_writes_all_formats(tmp_path):
    input_root = tmp_path / "runs"
    for condition in CONDITIONS:
        directory = input_root / condition
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(
            json.dumps(_fake_eval_summary(condition)), encoding="utf-8"
        )
    summaries = load_evaluation_summaries(input_root)
    long_rows, wide_rows, audits = validate_and_build_rows(summaries)
    payload = write_report(tmp_path / "report", long_rows, wide_rows, audits)
    assert payload["num_training_groups"] == 1
    assert payload["num_evaluation_results"] == 3
    assert all(Path(path).exists() for path in payload["outputs"].values())

    summaries[0]["checkpoint_sha256"] = "different"
    with pytest.raises(RuntimeError, match="do not share checkpoint_sha256"):
        validate_and_build_rows(summaries)


def test_existing_sparse_solution_contract_remains_joint_and_shared():
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=2, resolution=8, split="train")
    old = registry.make_task(
        raw,
        "poisson",
        "sparse_solution",
        num_sensors=5,
        sensor_mode="fixed",
        experiment_mode="paper",
    )
    assert old.input_fields.shape == old.target_fields.shape == (2, 2, 8, 8)
    assert torch.equal(old.mask[0], old.mask[1])
    assert torch.count_nonzero(old.input_fields[:, 0]) > 0
    assert torch.count_nonzero(old.input_fields[:, 1]) > 0
    assert "sensor_presence" not in old.metadata
    config = yaml.safe_load(
        (ROOT / "configs/experiments/main_results.yaml").read_text(encoding="utf-8")
    )
    assert "sparse_solution_multicondition_train" not in config["task_groups"]
