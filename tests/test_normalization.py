from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from baselines.common.data_adapter import PDEBatch, PDEBatchDataset, build_default_registry, pde_collate
from baselines.common.normalization import NormalizationStats, denormalize_prediction, estimate_normalization_stats, normalize_batch_input_target
from baselines.methods.base import BaselineModel
from baselines.run import _evaluate_full_test_loader, build_data_spec
from baselines.run import main


def test_normalize_batch_keeps_sparse_observation_shapes():
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=3, resolution=8)
    batch = registry.make_task(raw, "poisson", "sparse_solution", num_sensors=5, seed=1)
    loader = DataLoader(PDEBatchDataset(batch), batch_size=2, collate_fn=pde_collate)
    stats = estimate_normalization_stats(loader)
    norm_batch = normalize_batch_input_target(batch, stats)
    restored = denormalize_prediction(norm_batch.target_fields, stats)
    assert tuple(norm_batch.obs_values.shape) == tuple(batch.obs_values.shape)
    assert tuple(norm_batch.mask.shape) == tuple(batch.mask.shape)
    assert torch.isfinite(norm_batch.obs_values).all()
    assert tuple(restored.shape) == tuple(batch.target_fields.shape)


def test_train_loader_channelwise_mean_std_and_denormalize_roundtrip():
    batch = _manual_batch(
        task="forward",
        input_fields=torch.tensor(
            [
                [[[1.0, 3.0], [5.0, 7.0]], [[10.0, 14.0], [18.0, 22.0]]],
                [[[9.0, 11.0], [13.0, 15.0]], [[26.0, 30.0], [34.0, 38.0]]],
            ]
        ),
        target_fields=torch.tensor(
            [
                [[[2.0, 4.0], [6.0, 8.0]], [[20.0, 24.0], [28.0, 32.0]]],
                [[[10.0, 12.0], [14.0, 16.0]], [[36.0, 40.0], [44.0, 48.0]]],
            ]
        ),
    )
    loader = DataLoader(PDEBatchDataset(batch), batch_size=2, collate_fn=pde_collate)
    stats = estimate_normalization_stats(loader)
    norm = normalize_batch_input_target(batch, stats)

    assert stats.input_mean.tolist() == pytest.approx([8.0, 24.0])
    assert stats.target_mean.tolist() == pytest.approx([9.0, 34.0])
    assert norm.input_fields.mean(dim=(0, 2, 3)).tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
    assert norm.target_fields.mean(dim=(0, 2, 3)).tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
    assert norm.input_fields.std(dim=(0, 2, 3), unbiased=False).tolist() == pytest.approx([1.0, 1.0], abs=1e-6)
    assert norm.target_fields.std(dim=(0, 2, 3), unbiased=False).tolist() == pytest.approx([1.0, 1.0], abs=1e-6)
    assert torch.allclose(denormalize_prediction(norm.target_fields, stats), batch.target_fields, atol=1e-6)


def test_constant_channel_std_is_clamped_to_eps():
    batch = _manual_batch(
        task="forward",
        input_fields=torch.ones(2, 1, 2, 2) * 3.0,
        target_fields=torch.ones(2, 1, 2, 2) * 7.0,
    )
    loader = DataLoader(PDEBatchDataset(batch), batch_size=2, collate_fn=pde_collate)
    stats = estimate_normalization_stats(loader, eps=1e-3)
    assert stats.input_std.item() == pytest.approx(1e-3)
    assert stats.target_std.item() == pytest.approx(1e-3)


def test_sparse_observation_values_use_task_side_stats():
    stats = NormalizationStats(
        input_mean=torch.tensor([10.0]),
        input_std=torch.tensor([2.0]),
        target_mean=torch.tensor([100.0]),
        target_std=torch.tensor([5.0]),
        input_shape=(1, 1, 2, 2),
        target_shape=(1, 1, 2, 2),
        num_batches=1,
        num_samples=1,
    )
    mask = torch.ones(1, 2, 2)
    obs_values = torch.tensor([[[14.0], [18.0]]])
    inverse = _manual_batch(
        task="sparse_inverse",
        input_fields=torch.full((1, 1, 2, 2), 14.0),
        target_fields=torch.full((1, 1, 2, 2), 100.0),
        mask=mask,
        obs_values=obs_values,
        metadata={
            "masked_grid": torch.full((1, 1, 2, 2), 14.0),
            "voronoi_grid": torch.full((1, 1, 2, 2), 16.0),
            "observed_solution_fields": torch.full((1, 1, 2, 2), 18.0),
            "observation_source_fields": torch.full((1, 1, 2, 2), 18.0),
        },
    )
    solution = _manual_batch(
        task="sparse_solution",
        input_fields=torch.full((1, 1, 2, 2), 14.0),
        target_fields=torch.full((1, 1, 2, 2), 100.0),
        mask=mask,
        obs_values=obs_values,
        metadata={"masked_grid": torch.full((1, 1, 2, 2), 14.0), "voronoi_grid": torch.full((1, 1, 2, 2), 16.0)},
    )

    norm_inverse = normalize_batch_input_target(inverse, stats)
    norm_solution = normalize_batch_input_target(solution, stats)

    assert norm_inverse.obs_values.flatten().tolist() == pytest.approx([2.0, 4.0])
    assert norm_inverse.metadata["normalization_metadata_sources"]["obs_values"] == "input"
    assert norm_inverse.metadata["normalization_metadata_sources"]["observed_solution_fields"] == "input"
    assert norm_solution.obs_values.flatten().tolist() == pytest.approx([-17.2, -16.4])
    assert norm_solution.metadata["normalization_metadata_sources"]["obs_values"] == "target"
    assert norm_solution.metadata["normalization_metadata_sources"]["masked_grid"] == "target"


def test_evaluation_uses_predict_physical_for_normalized_model(tmp_path: Path):
    target = torch.full((2, 1, 2, 2), 20.0)
    batch = _manual_batch(
        task="forward",
        input_fields=torch.full((2, 1, 2, 2), 3.0),
        target_fields=target,
    )
    dataset = PDEBatchDataset(batch)
    loader = DataLoader(dataset, batch_size=2, collate_fn=pde_collate)
    stats = NormalizationStats(
        input_mean=torch.tensor([1.0]),
        input_std=torch.tensor([2.0]),
        target_mean=torch.tensor([10.0]),
        target_std=torch.tensor([5.0]),
        input_shape=(2, 1, 2, 2),
        target_shape=(2, 1, 2, 2),
        num_batches=1,
        num_samples=2,
    )
    model = _NormalizedTargetEcho().build({}, build_data_spec(batch))
    model.uses_normalization = True
    model.normalization_stats = stats
    args = Namespace(
        run_id="",
        run_name="",
        experiment_kind="",
        ablation_factor="",
        task_group="",
        pde="poisson",
        task="forward",
        baseline="echo",
        seed=1,
        device="cpu",
        train_shards=1,
        data_loading_mode="eager",
        load_full_trajectory=False,
        scalar_param_mode="metadata",
        num_sensors=0,
        sensor_budget_mode="per_time",
        sensor_mode="none",
        noise_level=0.0,
        physics_metric_mode="per_sample",
        experiment_mode="debug",
        dry_run=False,
        synthetic_data=True,
        data_root="",
    )
    backend_info = {
        "backend_used": "test",
        "official_backend": "test",
        "fallback_used": False,
        "backend_warning": "",
        "implementation_mode_requested": "adapted",
        "implementation_mode_effective": "adapted",
        "implementation_source": "test",
        "official_repo": "",
        "official_commit_or_version": "",
        "official_import_path": "",
        "official_vendored_path": "",
        "official_local_modifications": "",
        "official_metadata_note": "",
        "official_import_success": False,
        "official_reimplementation_success": False,
        "official_alignment_level": "",
        "official_alignment_notes": "",
        "adapter_status": "local_adapted",
    }
    capability_info = {
        "support_status": "native",
        "implementation_required": "official",
        "task_family": "full_forward",
        "reason": "",
        "unsupported_reason": "",
        "citation_key": "",
        "source_key": "",
        "notes_for_paper": "",
        "official_architecture_allowed": False,
        "official_aligned_allowed": False,
        "eligible_implementation_modes": [],
        "paper_table_eligible": False,
    }
    rows, _totals = _evaluate_full_test_loader(
        model=model,
        loader=loader,
        args=args,
        out_dir=tmp_path,
        config_snapshot="",
        checkpoint_path="",
        train_time=0.0,
        train_history={},
        backend_info=backend_info,
        capability_info=capability_info,
        train_dataset=dataset,
        spec_dataset=dataset,
        val_dataset=None,
        test_dataset=dataset,
        split_info={
            "effective_train_size": 2,
            "train_requested_size": 2,
            "train_size_loaded_for_fit": 2,
            "train_size_loaded_for_spec": 2,
            "val_requested_size": 0,
            "val_split_source": "none",
            "val_from_train_offset": None,
            "train_size_requested": 2,
            "train_size_loaded_in_memory": 2,
        },
        method_budget_fields={"steps": 0, "refine_steps": 0, "particles": 0, "method_budget_label": ""},
        normalization_fields={"normalize": True, "uses_normalization": True, "normalization_stats_path": "", "input_mean": "", "input_std": "", "target_mean": "", "target_std": ""},
        config_hash="",
    )
    assert rows[0]["mse"] == pytest.approx(0.0)
    assert rows[0]["mae"] == pytest.approx(0.0)
    assert json.loads(rows[0]["relative_l2_solution_values"]) == pytest.approx([0.0, 0.0])


def test_normalized_fno_tiny_run_saves_stats_and_best_val(tmp_path: Path):
    config = tmp_path / "norm.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "method": {
                    "implementation_mode": "adapted",
                    "official_backend": "local",
                    "normalize": True,
                    "width": 8,
                    "modes1": 4,
                    "modes2": 4,
                    "layers": 1,
                    "max_steps": 1,
                    "max_val_steps": 1,
                },
                "epochs": 1,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    main(
        [
            "--baseline",
            "fno",
            "--pde",
            "poisson",
            "--task",
            "forward",
            "--experiment-mode",
            "debug",
            "--synthetic-data",
            "--synthetic-resolution",
            "8",
            "--config",
            str(config),
            "--train-size",
            "4",
            "--val-size",
            "2",
            "--test-size",
            "2",
            "--batch-size",
            "2",
            "--epochs",
            "1",
            "--save-checkpoint",
            "--output-dir",
            str(out),
        ]
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["normalize"] is True
    assert summary["best_val_loss"] is not None
    assert Path(summary["normalization_stats_path"]).exists()
    ckpt = torch.load(summary["checkpoint_path"], map_location="cpu")
    assert ckpt["uses_normalization"] is True
    assert ckpt["normalization_stats"]["input_mean"].numel() == 1


class _NormalizedTargetEcho(BaselineModel):
    name = "normalized_target_echo"

    def predict(self, batch: PDEBatch):
        return batch.target_fields


def _manual_batch(
    *,
    task: str,
    input_fields: torch.Tensor,
    target_fields: torch.Tensor,
    mask: torch.Tensor | None = None,
    obs_values: torch.Tensor | None = None,
    metadata: dict | None = None,
) -> PDEBatch:
    coords = torch.stack(torch.meshgrid(torch.linspace(0, 1, input_fields.shape[-2]), torch.linspace(0, 1, input_fields.shape[-1]), indexing="ij"), dim=-1)
    coords = coords.reshape(1, -1, 2).repeat(input_fields.shape[0], 1, 1)
    return PDEBatch(
        pde_name="poisson",
        task=task,
        full_tensor=torch.cat([input_fields[:, :1], target_fields[:, :1]], dim=1),
        input_fields=input_fields.float(),
        target_fields=target_fields.float(),
        coords=coords.float(),
        mask=mask,
        obs_values=obs_values,
        obs_coords=None,
        channel_names=["input", "target"],
        input_channel_names=[f"x{i}" for i in range(input_fields.shape[1])],
        target_channel_names=[f"y{i}" for i in range(target_fields.shape[1])],
        metadata=dict(metadata or {}),
        pde_params={},
        split="train",
        sample_indices=torch.arange(input_fields.shape[0]),
        global_sample_ids=[str(i) for i in range(input_fields.shape[0])],
        file_paths=[],
    )
