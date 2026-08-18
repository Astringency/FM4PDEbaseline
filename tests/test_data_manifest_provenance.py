from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from baselines.run import _validate_checkpoint_payload, main as run_baseline
from baselines.run import parse_args as parse_baseline_args
from scripts.experiments.build_matrix import build_matrix, load_data_manifest_binding
from scripts.experiments.build_matrix import parse_args as parse_matrix_args
from scripts.experiments.provenance import (
    DEFAULT_TASK_PROTOCOL_VERSION,
    summary_validation_reasons,
)
from scripts.experiments.run_one import build_command


def _write_manifest(path: Path, *, status: str = "pass", mode: str = "full", revision: int = 1) -> Path:
    path.write_text(
        json.dumps({"status": status, "mode": mode, "revision": revision}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _config(method_config: Path) -> dict:
    return {
        "name": "manifest_test",
        "experiment_kind": "main",
        "comparison_track": "unified_adapted",
        "task_protocol_version": DEFAULT_TASK_PROTOCOL_VERSION,
        "config": str(method_config),
        "pdes": ["darcy"],
        "seeds": [1],
        "train_size": 4,
        "val_size": 0,
        "test_size": 2,
        "train_shards": 1,
        "device": "cpu",
        "task_groups": ["full_forward_main"],
        "task_group_overrides": {
            "full_forward_main": {
                "task": "forward",
                "pdes": ["darcy"],
                "baselines": ["fno"],
                "batch_size": 1,
                "epochs": 1,
            }
        },
    }


def _summary(row: dict) -> dict:
    return {
        "status": "success",
        "summary_schema_version": row["summary_schema_version"],
        "run_id": row["run_id"],
        "run_fingerprint": row["run_fingerprint"],
        "execution_mode": row["execution_mode"],
        "comparison_track": row["comparison_track"],
        "config_content_sha256": row["config_content_sha256"],
        "config_hash": "resolved-config-sha1",
        "data_manifest_sha256": row["data_manifest_sha256"],
        "data_manifest_path": row["data_manifest_path"],
        "commit_hash": row["commit_hash"],
        "task_protocol_version": row["task_protocol_version"],
        "sensor_protocol_version": row["sensor_protocol_version"],
        "sensor_seed": row["sensor_seed"],
        "task_group": row["task_group"],
        "task": row["task"],
        "pde": row["pde"],
        "baseline": row["baseline"],
        "seed": row["seed"],
    }


def test_matrix_accepts_only_passing_full_manifest_and_binds_its_content(tmp_path: Path):
    method_config = tmp_path / "method.yaml"
    method_config.write_text("method:\n  width: 4\n", encoding="utf-8")
    cfg = _config(method_config)

    bounded = _write_manifest(tmp_path / "bounded.json", mode="bounded")
    failed = _write_manifest(tmp_path / "failed.json", status="fail")
    with pytest.raises(ValueError, match="mode='full'"):
        load_data_manifest_binding(bounded)
    with pytest.raises(ValueError, match="status='pass'"):
        load_data_manifest_binding(failed)

    manifest = _write_manifest(tmp_path / "data_protocol_report.json")
    parsed = parse_matrix_args(["--config", str(method_config), "--data-manifest", str(manifest)])
    assert parsed.data_manifest == str(manifest)
    rows, skipped, summary = build_matrix(
        cfg,
        tmp_path / "bound-output",
        "manifest_test",
        data_manifest=manifest,
    )
    assert not skipped
    row = rows[0]
    expected_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert row["data_manifest_path"] == str(manifest.resolve())
    assert row["data_manifest_sha256"] == expected_sha256
    assert summary["data_manifest_sha256"] == expected_sha256

    previous_fingerprint = row["run_fingerprint"]
    _write_manifest(manifest, revision=2)
    changed_rows, _, _ = build_matrix(
        cfg,
        tmp_path / "changed-output",
        "manifest_test",
        data_manifest=manifest,
    )
    assert changed_rows[0]["run_fingerprint"] != previous_fingerprint

    design_rows, _, _ = build_matrix(cfg, tmp_path / "design-output", "manifest_test")
    assert design_rows[0]["data_manifest_sha256"] == ""
    assert design_rows[0]["data_manifest_path"] == ""


def test_run_one_forwards_bound_manifest_and_rejects_manifest_drift(tmp_path: Path, monkeypatch):
    method_config = tmp_path / "method.yaml"
    method_config.write_text("method:\n  width: 4\n", encoding="utf-8")
    manifest = _write_manifest(tmp_path / "data_protocol_report.json")
    rows, _, _ = build_matrix(
        _config(method_config),
        tmp_path / "output",
        "manifest_test",
        data_manifest=manifest,
    )
    row = rows[0]
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "PDEdata"))

    command = build_command(row)
    assert command[command.index("--data-manifest-sha256") + 1] == row["data_manifest_sha256"]
    assert command[command.index("--data-manifest-path") + 1] == row["data_manifest_path"]
    parsed = parse_baseline_args(command[3:])
    assert parsed.data_manifest_sha256 == row["data_manifest_sha256"]
    assert parsed.data_manifest_path == row["data_manifest_path"]

    _write_manifest(manifest, revision=2)
    with pytest.raises(RuntimeError, match="data manifest SHA-256"):
        build_command(row)


def test_formal_summary_and_checkpoint_validation_require_matching_manifest(tmp_path: Path):
    method_config = tmp_path / "method.yaml"
    method_config.write_text("method:\n  width: 4\n", encoding="utf-8")
    manifest = _write_manifest(tmp_path / "data_protocol_report.json")
    rows, _, _ = build_matrix(
        _config(method_config),
        tmp_path / "output",
        "manifest_test",
        data_manifest=manifest,
    )
    row = rows[0]
    summary = _summary(row)
    assert summary_validation_reasons(row, summary) == []

    missing = dict(summary)
    missing.pop("data_manifest_sha256")
    assert "summary_missing:data_manifest_sha256" in summary_validation_reasons(row, missing)
    mismatch = dict(summary, data_manifest_sha256="0" * 64)
    assert "summary_mismatch:data_manifest_sha256" in summary_validation_reasons(row, mismatch)

    design_rows, _, _ = build_matrix(_config(method_config), tmp_path / "design", "manifest_test")
    design_reasons = summary_validation_reasons(design_rows[0], _summary(design_rows[0]))
    assert "legacy_matrix_missing:data_manifest_sha256" in design_reasons

    payload = {
        "baseline": "fno",
        "data_spec": {},
        "provenance": {
            "data_manifest_sha256": row["data_manifest_sha256"],
            "data_manifest_path": row["data_manifest_path"],
        },
    }
    _validate_checkpoint_payload(
        payload,
        {
            "data_manifest_sha256": row["data_manifest_sha256"],
            "data_manifest_path": row["data_manifest_path"],
        },
        require_provenance=True,
    )
    with pytest.raises(ValueError, match="data_manifest_sha256"):
        _validate_checkpoint_payload(
            payload,
            {"data_manifest_sha256": "0" * 64},
            require_provenance=True,
        )


def test_formal_paper_run_fails_before_data_loading_without_manifest():
    with pytest.raises(ValueError, match="require --data-manifest-sha256"):
        run_baseline(
            [
                "--baseline",
                "fno",
                "--pde",
                "poisson",
                "--experiment-mode",
                "paper",
                "--task-protocol-version",
                DEFAULT_TASK_PROTOCOL_VERSION,
            ]
        )


def test_bound_manifest_is_written_to_checkpoint_raw_rows_and_summary(tmp_path: Path):
    manifest = _write_manifest(tmp_path / "data_protocol_report.json")
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    config = tmp_path / "debug.yaml"
    config.write_text(
        json.dumps(
            {
                "epochs": 1,
                "learning_rate": 0.001,
                "method": {
                    "implementation_mode": "adapted",
                    "official_backend": "local",
                    "hidden": 8,
                    "basis": 4,
                    "max_steps": 1,
                    "max_val_steps": 1,
                    "normalize": False,
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"
    run_baseline(
        [
            "--baseline",
            "deeponet",
            "--pde",
            "poisson",
            "--task",
            "forward",
            "--experiment-mode",
            "debug",
            "--synthetic-data",
            "--synthetic-resolution",
            "8",
            "--dry-run",
            "--config",
            str(config),
            "--train-size",
            "2",
            "--val-size",
            "1",
            "--test-size",
            "1",
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--task-protocol-version",
            DEFAULT_TASK_PROTOCOL_VERSION,
            "--data-manifest-sha256",
            manifest_sha256,
            "--data-manifest-path",
            str(manifest),
            "--output-dir",
            str(output_dir),
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    raw = json.loads((output_dir / "results_raw.jsonl").read_text(encoding="utf-8").splitlines()[0])
    checkpoint = torch.load(next(output_dir.glob("*.pt")), map_location="cpu", weights_only=False)
    for payload in (summary, raw, checkpoint["provenance"]):
        assert payload["data_manifest_sha256"] == manifest_sha256
        assert payload["data_manifest_path"] == str(manifest.resolve())
