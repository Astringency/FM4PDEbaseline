from __future__ import annotations

import json
from pathlib import Path

import torch

from baselines.common.data_adapter import build_default_registry
from baselines.common.sample_artifacts import (
    EvaluationArtifactWriter,
    load_evaluation_sample,
    render_sample_manifest_pdf,
)


def test_evaluation_artifacts_round_trip_every_sample_and_render_pdf(tmp_path: Path):
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=2, resolution=4, split="test", seed=9)
    batch = registry.make_task(
        raw,
        "poisson",
        "sparse_forward",
        num_sensors=3,
        sensor_mode="fixed",
        seed=4,
    )
    prediction = batch.target_fields + 0.25
    predictive_std = torch.full_like(prediction, 0.1)
    posterior_samples = torch.stack([prediction - 0.1, prediction + 0.1], dim=1)

    writer = EvaluationArtifactWriter(
        tmp_path / "samples",
        run_metadata={"run_id": "run-1", "baseline": "pc_bnn", "seed": 1},
        pdf_path=tmp_path / "samples.pdf",
    )
    with writer:
        records = writer.write_batch(
            batch,
            prediction,
            predictive_std=predictive_std,
            posterior_samples=posterior_samples,
            metrics=[{"relative_l2_solution": 0.1}, {"relative_l2_solution": 0.2}],
            batch_index=0,
        )

    assert len(records) == 2
    assert writer.summary["sample_artifact_count"] == 2
    assert writer.manifest_path.exists()
    assert (tmp_path / "samples.pdf").read_bytes().startswith(b"%PDF")

    manifest = [json.loads(line) for line in writer.manifest_path.read_text(encoding="utf-8").splitlines()]
    assert [row["sample_ordinal"] for row in manifest] == [0, 1]
    loaded = load_evaluation_sample(manifest[0]["artifact_path"])
    assert loaded["schema_version"] == "fm4pde-evaluation-sample-v1"
    assert loaded["global_sample_id"] == batch.global_sample_ids[0]
    assert torch.equal(loaded["prediction"], prediction[0])
    assert torch.equal(loaded["predictive_std"], predictive_std[0])
    assert loaded["posterior_samples"].shape == (2, *prediction.shape[1:])
    assert loaded["metrics"]["relative_l2_solution"] == 0.1

    regenerated = render_sample_manifest_pdf(writer.manifest_path, tmp_path / "regenerated.pdf")
    assert regenerated.read_bytes().startswith(b"%PDF")
