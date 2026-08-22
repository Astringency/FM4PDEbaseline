from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from baselines.configuration import resolve_method_config
from scripts.experiments import run_one
from scripts.experiments.provenance import (
    DEFAULT_SENSOR_PROTOCOL_VERSION,
    MATRIX_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    baseline_code_sha256,
    baseline_config_sha256,
    canonical_sha256,
    repository_revision,
    run_fingerprint,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def _row(tmp_path: Path, run_id: str = "progress_run") -> dict:
    config_path = ROOT / "baselines/configs/paper.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    effective_method_config = resolve_method_config(
        config,
        baseline="recfno",
        pde="poisson",
        epochs=1,
        device="cpu",
        seed=1,
    )
    row = {
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "run_id": "",
        "run_fingerprint": "",
        "run_name": run_id,
        "execution_mode": "train",
        "comparison_track": "unified_adapted",
        "experiment_kind": "main",
        "ablation_factor": "",
        "task_group": "sparse_solution_main_amortized",
        "task": "sparse_solution",
        "task_protocol_version": "progress-test-v1",
        "sensor_protocol_version": DEFAULT_SENSOR_PROTOCOL_VERSION,
        "data_manifest_sha256": "",
        "data_manifest_path": "",
        "pde": "poisson",
        "baseline": "recfno",
        "seed": 1,
        "sensor_seed": 1,
        "train_size": 8,
        "val_size": 0,
        "test_size": 4,
        "train_shards": 1,
        "num_sensors": 16,
        "sensor_mode": "random_per_sample",
        "sensor_budget_mode": "per_time",
        "noise_level": 0.0,
        "scalar_param_mode": "metadata",
        "physics_metric_mode": "per_sample",
        "data_loading_mode": "eager",
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "load_full_trajectory": False,
        "batch_size": 1,
        "epochs": 1,
        "steps": 0,
        "refine_steps": 0,
        "particles": 0,
        "device": "cpu",
        "commit_hash": repository_revision(ROOT),
        "config": "baselines/configs/paper.yaml",
        "config_content_sha256": sha256_file("baselines/configs/paper.yaml", root=ROOT),
        "baseline_config_sha256": baseline_config_sha256(effective_method_config),
        "baseline_code_sha256": baseline_code_sha256("recfno", root=ROOT),
        "data_content_sha256": canonical_sha256(
            {"verification": "not_bound_to_full_data_manifest"}
        ),
        "experiment_config_sha256": "",
        "output_dir": str(tmp_path / run_id),
        "log_dir": str(tmp_path / "logs" / run_id),
        "status_file": str(tmp_path / run_id / "run.status.json"),
        "skip_reason": "",
    }
    row["run_fingerprint"] = run_fingerprint(row)
    row["run_id"] = run_id
    # These tests use a human-readable run id; recompute the fingerprint after
    # setting it is unnecessary because run_id is intentionally not recursive.
    return row


def _summary(row: dict) -> dict:
    return {
        "status": "success",
        "matrix_schema_version": row["matrix_schema_version"],
        "summary_schema_version": row["summary_schema_version"],
        "run_id": row["run_id"],
        "run_fingerprint": row["run_fingerprint"],
        "baseline_config_sha256": row["baseline_config_sha256"],
        "baseline_code_sha256": row["baseline_code_sha256"],
        "data_content_sha256": row["data_content_sha256"],
        "execution_mode": row["execution_mode"],
        "eval_only": False,
        "comparison_track": row["comparison_track"],
        "config_content_sha256": row["config_content_sha256"],
        "config_hash": "resolved-config-sha1",
        "commit_hash": row["commit_hash"],
        "task_protocol_version": row["task_protocol_version"],
        "sensor_protocol_version": row["sensor_protocol_version"],
        "sensor_seed": row["sensor_seed"],
        "task_group": row["task_group"],
        "task": row["task"],
        "pde": row["pde"],
        "baseline": row["baseline"],
        "seed": row["seed"],
        "train_requested_size": row["train_size"],
        "train_size_requested": row["train_size"],
        "val_requested_size": row["val_size"],
        "val_size": row["val_size"],
        "test_requested_size": row["test_size"],
        "test_size": row["test_size"],
        "train_shards": row["train_shards"],
        "batch_size": row["batch_size"],
        "epochs": row["epochs"],
        "device": row["device"],
        "num_sensors": row["num_sensors"],
        "requested_sensor_mode": row["sensor_mode"],
        "sensor_budget_mode_requested": row["sensor_budget_mode"],
        "noise_level": row["noise_level"],
        "steps": row["steps"],
        "refine_steps": row["refine_steps"],
        "particles": row["particles"],
        "scalar_param_mode_requested": row["scalar_param_mode"],
        "metric_granularity": row["physics_metric_mode"],
        "data_loading_mode_requested": row["data_loading_mode"],
        "num_workers": row["num_workers"],
        "pin_memory_requested": row["pin_memory"],
        "persistent_workers_requested": row["persistent_workers"],
        "prefetch_factor_requested": row["prefetch_factor"],
        "load_full_trajectory": row["load_full_trajectory"],
    }


def test_progress_can_be_disabled(monkeypatch, capsys):
    monkeypatch.setenv("RUN_PROGRESS", "0")

    run_one.progress("[run start] index=1/1 run_id=hidden output_dir=/tmp/hidden")

    captured = capsys.readouterr()
    assert captured.err == ""


def test_default_progress_reports_skip_context(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv("RUN_PROGRESS", raising=False)
    row = _row(tmp_path, "skip_context")
    output_dir = Path(row["output_dir"])
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(json.dumps(_summary(row)), encoding="utf-8")
    (output_dir / "run.done").write_text("{}", encoding="utf-8")

    rc = run_one.run_one(row, [sys.executable, "-c", "raise SystemExit(99)"], index=2, total=5)

    assert rc == 0
    captured = capsys.readouterr()
    assert "[run skip]" in captured.err
    assert "index=3/5" in captured.err
    assert "run_id=skip_context" in captured.err
    assert f"output_dir={output_dir}" in captured.err


def test_heartbeat_uses_short_interval_without_slow_test(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv("RUN_PROGRESS", raising=False)
    monkeypatch.delenv("RUN_TAIL_LOGS", raising=False)
    monkeypatch.setenv("PROGRESS_INTERVAL_SECONDS", "0.01")
    row = _row(tmp_path, "heartbeat_context")
    summary_path = Path(row["output_dir"]) / "summary.json"
    summary_json = json.dumps(_summary(row))
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys, time; from pathlib import Path; "
            "print('child stdout'); print('child stderr', file=sys.stderr); time.sleep(0.08); "
            f"Path({str(summary_path)!r}).write_text({summary_json!r}, encoding='utf-8')"
        ),
    ]

    rc = run_one.run_one(row, cmd, index=0, total=1)

    assert rc == 0
    captured = capsys.readouterr()
    assert "[run start] index=1/1 run_id=heartbeat_context" in captured.err
    assert "[run heartbeat] index=1/1 run_id=heartbeat_context" in captured.err
    assert "stdout=" in captured.err
    assert "stderr=" in captured.err
    assert "child stdout" in (Path(row["output_dir"]) / "stdout.log").read_text(encoding="utf-8")
    assert "child stderr" in (Path(row["output_dir"]) / "stderr.log").read_text(encoding="utf-8")
