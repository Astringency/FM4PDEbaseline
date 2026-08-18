from __future__ import annotations

import pytest
import torch

from baselines.common.data_adapter import build_default_registry
from baselines.methods.fno import FNOBaseline
from baselines.run import build_data_spec, load_baseline_checkpoint


def _model():
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=1, resolution=8)
    batch = registry.make_task(raw, "poisson", "forward")
    return FNOBaseline().build(
        {
            "implementation_mode": "adapted",
            "official_backend": "local",
            "width": 4,
            "modes1": 2,
            "modes2": 2,
            "layers": 1,
        },
        build_data_spec(batch),
    )


def test_strict_checkpoint_load_rejects_task_mismatch(tmp_path):
    model = _model()
    model.provenance = {
        "run_id": "train-1",
        "run_fingerprint": "abc",
        "pde": "poisson",
        "task": "forward",
        "config_content_sha256": "def",
        "task_protocol_version": "2",
        "sensor_protocol_version": "2",
    }
    path = tmp_path / "model.pt"
    model.save(path)

    with pytest.raises(ValueError, match="task"):
        load_baseline_checkpoint(
            path,
            expected={"baseline": "fno", "pde": "poisson", "task": "inverse"},
            require_provenance=True,
        )


def test_strict_checkpoint_load_rejects_legacy_payload(tmp_path):
    model = _model()
    path = tmp_path / "legacy.pt"
    model.save(path)
    payload = torch.load(path, weights_only=False)
    payload.pop("provenance", None)
    torch.save(payload, path)

    with pytest.raises(ValueError, match="no provenance"):
        load_baseline_checkpoint(path, require_provenance=True)


def test_strict_checkpoint_load_rejects_repository_revision_mismatch(tmp_path):
    model = _model()
    model.provenance = {
        "run_id": "train-1",
        "run_fingerprint": "abc",
        "pde": "poisson",
        "task": "forward",
        "commit_hash": "revision-a",
    }
    path = tmp_path / "model.pt"
    model.save(path)

    with pytest.raises(ValueError, match="commit_hash"):
        load_baseline_checkpoint(
            path,
            expected={"commit_hash": "revision-b"},
            require_provenance=True,
        )
