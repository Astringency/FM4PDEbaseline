from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from baselines.common.data_adapter import build_default_registry
from baselines.common.sample_artifacts import EvaluationArtifactWriter, load_evaluation_sample
from scripts.recompute_metrics import _prediction_errors, recompute_run, select_run_dirs


@pytest.fixture
def saved_run(tmp_path: Path) -> Path:
    directory = tmp_path / "run=cached-run"
    directory.mkdir()
    registry = build_default_registry()
    batch = registry.make_task(
        registry.synthetic_raw("burger", n=3, resolution=4, split="test"),
        "burger", "sparse_solution", num_sensors=3, sensor_mode="fixed",
    )
    batch.target_fields.fill_(1.0)
    prediction = batch.target_fields.clone()
    prediction[0, :, 0] += 1.0
    prediction[1, :, 1] += 2.0
    prediction[2, :, -1] += 3.0
    identity = {"run_id": "cached-run", "baseline": "var4d", "pde": "burger", "task": "sparse_solution", "seed": 1}
    writer = EvaluationArtifactWriter(directory / "cached-run_samples", run_metadata=identity)
    writer.write_batch(batch, prediction, metrics=[{"relative_l2_solution": 99.0}] * 3, batch_index=0)
    writer.finalize()
    summary = {
        **identity, **writer.summary, "status": "success", "test_size": 3,
        "relative_l2_solution_mean": 99.0, "relative_l2_solution_std": 99.0,
        "train_time": 123.0, "inference_time_total": 456.0, "pde_residual_mean": 7.0,
    }
    raw = {
        **identity, "batch_index": 0, "sample_count": 3,
        "global_sample_ids": json.dumps(batch.global_sample_ids),
        "pred_shape": json.dumps(list(prediction.shape)),
        "target_shape": json.dumps(list(batch.target_fields.shape)),
        "relative_l2_solution_values": "[99.0, 99.0, 99.0]", "pde_residual": 7.0,
    }
    (directory / "results_raw.jsonl").write_text(json.dumps(raw) + "\n")
    for name in ("summary.json", "cached-run_summary.json"):
        (directory / name).write_text(json.dumps(summary))
    other = {**summary, "run_id": "other-run", "relative_l2_solution_mean": 88.0}
    (directory / "results_summary.jsonl").write_text(json.dumps(other) + "\n" + json.dumps(summary) + "\n")
    for name in ("results_summary.csv", "cached-run_results_summary_latest.csv"):
        with (directory / name).open("w", newline="") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=list(summary))
            csv_writer.writeheader()
            csv_writer.writerows([other, summary])
    return directory


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def test_recompute_updates_all_metric_outputs_and_keeps_saved_tensors(saved_run: Path):
    samples_before = _snapshot(saved_run / "cached-run_samples")
    result = recompute_run(saved_run)
    assert result["samples"] == 3
    assert result["relative_l2_solution_mean_before"] == 99.0
    assert result["relative_l2_solution_mean_after"] == pytest.approx(1.0)
    summary = json.loads((saved_run / "summary.json").read_text())
    assert summary["relative_l2_solution_mean"] == pytest.approx(1.0)
    assert summary["relative_l2_solution_std"] == pytest.approx(0.5)
    assert summary["relative_l2_solution_sem"] == pytest.approx(0.5 / math.sqrt(3))
    assert summary["relative_l2_solution_ci95"] == pytest.approx(1.96 * 0.5 / math.sqrt(3))
    assert summary["relative_l2_solution_n"] == 3
    assert summary["relative_l2_solution_scope"] == "full_trajectory"
    assert summary["relative_l2_input_or_coeff_mean"] == pytest.approx(1 / 3)
    assert summary["mse_mean"] == pytest.approx(3.5 / 3)
    assert summary["mae_mean"] == pytest.approx(0.5)
    assert summary["pde_residual_mean"] == 7.0
    assert summary["train_time"] == 123.0
    assert summary["inference_time_total"] == 456.0
    assert json.loads((saved_run / "cached-run_summary.json").read_text()) == summary
    raw = json.loads((saved_run / "results_raw.jsonl").read_text())
    assert json.loads(raw["relative_l2_solution_values"]) == pytest.approx([0.5, 1.0, 1.5])
    assert raw["relative_l2_solution"] == pytest.approx(1.0)
    assert raw["relative_l2_solution_scope"] == "full_trajectory"
    assert raw["pde_residual"] == 7.0
    rows = [json.loads(line) for line in (saved_run / "results_summary.jsonl").read_text().splitlines()]
    assert rows[0]["relative_l2_solution_mean"] == 88.0
    assert rows[1] == summary
    for name in ("results_summary.csv", "cached-run_results_summary_latest.csv"):
        with (saved_run / name).open() as handle:
            csv_rows = list(csv.DictReader(handle))
        assert csv_rows[0]["relative_l2_solution_mean"] == "88.0"
        assert float(csv_rows[1]["relative_l2_solution_std"]) == pytest.approx(0.5)
        assert csv_rows[1]["relative_l2_solution_scope"] == "full_trajectory"
    assert _snapshot(saved_run / "cached-run_samples") == samples_before
    once = _snapshot(saved_run)
    recompute_run(saved_run, workers=1)
    assert _snapshot(saved_run) == once


def test_recompute_cli_dry_run_reads_predictions_without_writing(saved_run: Path):
    before = _snapshot(saved_run)
    command = [sys.executable, "-B", "scripts/recompute_metrics.py", "--run-dir", str(saved_run), "--dry-run"]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '"relative_l2_solution_mean_after": 1.0' in result.stdout
    assert _snapshot(saved_run) == before


def test_recompute_accepts_artifacts_moved_from_another_host(saved_run: Path):
    summary_path = saved_run / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["sample_artifact_dir"] = "/remote/old/run=cached-run/cached-run_samples"
    summary["sample_manifest_path"] = summary["sample_artifact_dir"] + "/manifest.jsonl"
    summary_path.write_text(json.dumps(summary))
    manifest_path = saved_run / "cached-run_samples/manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    for record in records:
        record["artifact_path"] = summary["sample_artifact_dir"] + "/" + Path(record["artifact_path"]).name
    manifest_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    assert recompute_run(saved_run, dry_run=True)["relative_l2_solution_mean_after"] == pytest.approx(1.0)


def test_nonfinite_predictions_retain_nan_counts_and_resumable_arrays(saved_run: Path):
    path = saved_run / "cached-run_samples/sample_000002.pt"
    sample = load_evaluation_sample(path)
    sample["prediction"].fill_(float("nan"))
    torch.save(sample, path)
    manifest_path = path.parent / "manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    records[-1]["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    recompute_run(saved_run)
    summary = json.loads((saved_run / "summary.json").read_text())
    assert summary["relative_l2_solution_n"] == 2
    assert summary["relative_l2_solution_nan_count"] == 1
    assert summary["relative_l2_solution_mean"] == pytest.approx(0.75)
    values = json.loads(json.loads((saved_run / "results_raw.jsonl").read_text())["relative_l2_solution_values"])
    assert math.isnan(values[-1])


@pytest.mark.parametrize("failure", ["missing", "checksum", "duplicate", "shape", "run_identity", "partial_raw"])
def test_invalid_samples_never_replace_any_results(saved_run: Path, failure: str):
    manifest_path = saved_run / "cached-run_samples/manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    last_file = saved_run / "cached-run_samples/sample_000002.pt"
    if failure == "missing":
        last_file.unlink()
    elif failure == "checksum":
        last_file.write_bytes(last_file.read_bytes() + b"damaged")
    elif failure == "duplicate":
        records[-1]["global_sample_id"] = records[0]["global_sample_id"]
    elif failure in {"shape", "run_identity"}:
        sample = load_evaluation_sample(last_file)
        if failure == "shape":
            sample["prediction"] = sample["prediction"][:, :2]
            sample["target_fields"] = sample["target_fields"][:, :2]
        else:
            sample["run_metadata"]["run_id"] = "different-run"
        torch.save(sample, last_file)
        records[-1]["artifact_sha256"] = hashlib.sha256(last_file.read_bytes()).hexdigest()
    else:
        raw_path = saved_run / "results_raw.jsonl"
        raw = json.loads(raw_path.read_text())
        raw["sample_count"] = 2
        raw["global_sample_ids"] = json.dumps(json.loads(raw["global_sample_ids"])[:2])
        raw_path.write_text(json.dumps(raw))
    manifest_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    before = _snapshot(saved_run)
    with pytest.raises((OSError, ValueError)):
        recompute_run(saved_run)
    assert _snapshot(saved_run) == before


def test_offline_metrics_preserve_inverse_and_multicondition_definitions():
    sample = {
        "pde_name": "poisson", "task": "inverse", "metadata": {},
        "target_fields": torch.ones(1, 2, 2), "prediction": torch.full((1, 2, 2), 3.0),
    }
    inverse = _prediction_errors(sample)
    assert math.isnan(inverse["relative_l2_solution"])
    assert inverse["relative_l2_input_or_coeff"] == 2.0
    sample.update({
        "task": "sparse_solution_multicondition", "metadata": {"joint_input_channels": 1},
        "target_fields": torch.ones(2, 2, 2),
        "prediction": torch.stack([torch.full((2, 2), 2.0), torch.full((2, 2), 4.0)]),
        "mask": torch.ones(2, 2, 2, dtype=torch.bool),
    })
    multi = _prediction_errors(sample)
    assert multi["rel_l2_a"] == 1.0
    assert multi["rel_l2_u"] == 3.0
    assert multi["joint_rel_l2"] == pytest.approx(math.sqrt(5))
    assert multi["observed_mse_a"] == 1.0
    assert multi["observed_mse_u"] == 9.0


def test_matrix_selection_filters_methods_pdes_and_distributions(tmp_path: Path):
    rows = [
        {"run_id": "a", "pde": "burger", "baseline": "recfno", "task_group": "sparse", "seed": 1},
        {"run_id": "b", "pde": "burger", "baseline": "vivid", "task_group": "sparse", "seed": 1},
        {"run_id": "c", "pde": "darcy", "baseline": "recfno", "task_group": "sparse", "seed": 1},
    ]
    matrix_path = tmp_path / "matrices/main_results.jsonl"
    matrix_path.parent.mkdir()
    matrix_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    args = SimpleNamespace(run_dir=[], output_root=str(tmp_path), matrix=None, pde=["burger"], baselines="recfno", distributions="main,id")
    selected = select_run_dirs(args)
    assert selected == [
        tmp_path / "runs/main_results/task_group=sparse/pde=burger/baseline=recfno/seed=1/run=a",
        tmp_path / "runs/evaluations/id/task_group=sparse/pde=burger/baseline=recfno/seed=1/run=eval_id_a",
    ]
