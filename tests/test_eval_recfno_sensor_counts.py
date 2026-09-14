import json

import numpy as np
import pytest

from baselines.common.data_adapter import PDEBatchDataset, build_default_registry
from baselines.methods.recfno import RecFNOBaseline
from baselines.run import build_data_spec
from scripts.eval_recfno_sensor_counts import evaluate_cell


def test_evaluation_covers_samples_and_checks_resumed_identity(tmp_path):
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=5, resolution=8, split="test", seed=3)
    batch = registry.make_task(raw, "poisson", "sparse_solution_multicondition", num_sensors=3,
                              sensor_mode="random_per_sample", sensor_budget_mode="total", condition_mode="mixed")
    dataset = PDEBatchDataset(batch)
    model = RecFNOBaseline().build(dict(implementation_mode="adapted", official_backend="local", width=4,
                                       modes1=2, modes2=2), build_data_spec(dataset[0]))
    identity = dict(condition="a_only", test_size=5)
    first = evaluate_cell(model, dataset, device="cpu", batch_size=2, output=tmp_path, identity=identity)
    with np.load(tmp_path / "per_sample.npz") as values:
        assert len(values["sample_ids"]) == 5
        assert first["rel_l2_u"] == pytest.approx(values["rel_l2_u"].mean())
    assert first == evaluate_cell(model, dataset, device="cpu", batch_size=2, output=tmp_path, identity=identity)
    with pytest.raises(ValueError, match="different identity"):
        evaluate_cell(model, dataset, device="cpu", batch_size=2, output=tmp_path,
                      identity={**identity, "condition": "both"})
    (tmp_path / "per_sample.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        evaluate_cell(model, dataset, device="cpu", batch_size=2, output=tmp_path, identity=identity)
