from __future__ import annotations

import json
from argparse import Namespace

import torch
from torch.utils.data import DataLoader
import pytest

from baselines.capabilities import paper_table_eligible, resolve_capability
from baselines.common.data_adapter import PDEBatchDataset, build_default_registry, pde_collate
from baselines.common.normalization import NormalizationStats
from baselines.methods.official import OfficialImportError
from baselines.methods.vivid import VIVIDBaseline
from baselines.run import _backend_info, _evaluate_full_test_loader, build_data_spec


def _time_varying_batch(pde: str = "reaction_diffusion"):
    registry = build_default_registry()
    raw = registry.synthetic_raw(pde, n=2, resolution=8)
    return registry.make_task(
        raw,
        pde,
        "sparse_solution",
        num_sensors=4,
        sensor_mode="time_varying",
        seed=1,
        experiment_mode="debug",
    )


def test_vivid_official_aligned_time_varying_da_trains_and_predicts():
    batch = _time_varying_batch("reaction_diffusion")
    cfg = {
        "implementation_mode": "official_aligned",
        "official_backend": "vivid",
        "uses_official_inverse_observation_operator": True,
        "epochs": 1,
        "max_steps": 1,
        "refine_steps": 1,
        "inverse_width": 8,
        "inverse_depth": 1,
    }
    model = VIVIDBaseline().build(cfg, build_data_spec(batch))
    loader = DataLoader(PDEBatchDataset(batch), batch_size=1, collate_fn=pde_collate)
    history = model.fit(loader)
    pred = model.predict(batch)
    backend = _backend_info(model, model.config)
    cap = resolve_capability(
        "vivid",
        "reaction_diffusion",
        "sparse_solution",
        "time_varying",
        "time_varying",
        load_full_trajectory=True,
        train_inverse_operator=True,
        uses_official_inverse_observation_operator=True,
    )
    assert tuple(pred.shape) == tuple(batch.target_fields.shape)
    assert batch.metadata["assimilation_mode"] == "full_trajectory"
    assert batch.metadata["inverse_observation_operator_used"] is True
    assert history["inverse_operator_loss"]
    assert "style" not in backend["adapter_status"]
    assert "local" not in backend["adapter_status"]
    assert paper_table_eligible(cap, backend_info=backend) is True


def test_vivid_style_adapted_path_is_supplement_only():
    batch = _time_varying_batch("shallow_water")
    cfg = {"implementation_mode": "adapted", "official_backend": "local", "train_inverse_operator": True, "epochs": 1}
    model = VIVIDBaseline().build(cfg, build_data_spec(batch))
    backend = _backend_info(model, model.config)
    cap = resolve_capability(
        "vivid",
        "shallow_water",
        "sparse_solution",
        "time_varying",
        "time_varying",
        load_full_trajectory=True,
        train_inverse_operator=True,
        uses_official_inverse_observation_operator=False,
    )
    assert backend["implementation_mode_effective"] == "adapted"
    assert paper_table_eligible(cap, backend_info=backend) is False


def test_vivid_strict_official_does_not_fallback_to_aligned():
    batch = _time_varying_batch("reaction_diffusion")
    cfg = {
        "implementation_mode": "official",
        "official_backend": "vivid",
        "uses_official_inverse_observation_operator": True,
    }
    with pytest.raises(OfficialImportError):
        VIVIDBaseline().build(cfg, build_data_spec(batch))


def test_vivid_predict_physical_avoids_double_denormalization(tmp_path):
    batch = _time_varying_batch("reaction_diffusion")
    physical_target = torch.full_like(batch.target_fields, 20.0)
    batch.target_fields = physical_target.clone()
    dataset = PDEBatchDataset(batch)
    loader = DataLoader(dataset, batch_size=2, collate_fn=pde_collate)
    model = _PhysicalReturningVIVID(physical_target)
    model.uses_normalization = True
    model.normalization_stats = NormalizationStats(
        input_mean=torch.tensor([1.0, 1.0]),
        input_std=torch.tensor([2.0, 2.0]),
        target_mean=torch.tensor([10.0, 10.0]),
        target_std=torch.tensor([5.0, 5.0]),
        input_shape=tuple(batch.input_fields.shape),
        target_shape=tuple(batch.target_fields.shape),
        num_batches=1,
        num_samples=int(batch.target_fields.shape[0]),
    )
    args = Namespace(
        run_id="",
        run_name="",
        experiment_kind="",
        ablation_factor="",
        task_group="time_varying_da_main",
        pde="reaction_diffusion",
        task="sparse_solution",
        baseline="vivid",
        seed=1,
        device="cpu",
        train_shards=1,
        data_loading_mode="eager",
        load_full_trajectory=True,
        scalar_param_mode="metadata",
        num_sensors=4,
        sensor_budget_mode="per_time",
        sensor_mode="time_varying",
        noise_level=0.0,
        physics_metric_mode="per_sample",
        experiment_mode="debug",
        dry_run=False,
        synthetic_data=True,
        data_root="",
    )
    backend_info = _backend_info(model, model.config)
    capability = resolve_capability(
        "vivid",
        "reaction_diffusion",
        "sparse_solution",
        "time_varying",
        "time_varying_da_main",
        load_full_trajectory=True,
        train_inverse_operator=True,
        uses_official_inverse_observation_operator=True,
    )
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
        capability_info={**capability.to_row(), "paper_table_eligible": False},
        train_dataset=dataset,
        spec_dataset=dataset,
        val_dataset=None,
        test_dataset=dataset,
        split_info={
            "effective_train_size": 2,
            "train_requested_size": 2,
            "train_size_loaded_for_fit": 0,
            "train_size_loaded_for_spec": 2,
            "val_requested_size": 0,
            "val_split_source": "none",
            "val_from_train_offset": None,
            "train_size_requested": 2,
            "train_size_loaded_in_memory": 2,
        },
        method_budget_fields={"steps": 0, "refine_steps": 0, "particles": 0, "method_budget_label": ""},
        normalization_fields={
            "normalize": True,
            "uses_normalization": True,
            "normalization_stats_path": "",
            "input_mean": "",
            "input_std": "",
            "target_mean": "",
            "target_std": "",
        },
        config_hash="",
    )
    assert rows[0]["mse"] == pytest.approx(0.0)
    assert rows[0]["mae"] == pytest.approx(0.0)
    assert json.loads(rows[0]["relative_l2_solution_values"]) == pytest.approx([0.0, 0.0])


class _PhysicalReturningVIVID(VIVIDBaseline):
    def __init__(self, physical: torch.Tensor) -> None:
        super().__init__()
        self.physical = physical
        self.config = {}
        self.data_spec = {}
        self.mark_canonical_math("vivid_test", adapter_status="canonical_math")

    def predict(self, batch):
        return self.physical[: batch.target_fields.shape[0]].to(batch.target_fields.device, batch.target_fields.dtype)

    def parameter_count(self) -> int:
        return 0
