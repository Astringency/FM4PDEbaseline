from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from baselines.run import parse_args as parse_baseline_args
from scripts.experiments.provenance import (
    DATA_MANIFEST_CONTENT_HASH_CONTRACT,
    DATA_MANIFEST_REPORT_SCHEMA_VERSION,
)
from scripts.experiments.run_one import build_command
from scripts.experiments.sensor_generalization_common import (
    make_provenance_row,
    require_source_identity,
)


def _template(config: Path) -> dict:
    return {
        "baseline": "recfno",
        "pde": "poisson",
        "task": "sparse_forward",
        "task_group": "sparse_forward_main_amortized",
        "experiment_kind": "main",
        "comparison_track": "unified_adapted",
        "config": str(config),
        "train_size": 8,
        "val_size": 1,
        "test_size": 2,
        "train_shards": 1,
        "batch_size": 2,
        "epochs": 1,
        "device": "cpu",
        "data_loading_mode": "eager",
        "num_workers": 0,
        "prefetch_factor": 2,
        "scalar_param_mode": "metadata",
        "num_sensors": 4,
        "sensor_mode": "random_per_sample",
        "sensor_budget_mode": "per_time",
        "noise_level": 0.0,
        "pin_memory": False,
        "persistent_workers": False,
        "load_full_trajectory": False,
    }


def _write_full_manifest(path: Path, data_root: Path, config: Path) -> Path:
    source = data_root / "poisson" / "fixture.bin"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"sensor-generalization-fixture")
    stat = source.stat()
    signature = {
        "path": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }
    verifier = Path(__file__).resolve().parents[1] / "scripts" / "verify_data_protocol.py"
    sample_row = {
        "pde": "poisson",
        "split": "test",
        "source_kind": "fixture",
        "ordinal": 0,
        "sample_index": 0,
        "global_sample_id": "poisson:test:0",
        "field_sha256": hashlib.sha256(b"sensor-field").hexdigest(),
        "content_sha256": hashlib.sha256(b"sensor-content").hexdigest(),
    }
    sample_content = (
        json.dumps(sample_row, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    path.with_name("sample_manifest.jsonl").write_bytes(sample_content)
    payload = {
        "report_schema_version": DATA_MANIFEST_REPORT_SCHEMA_VERSION,
        "content_hash_contract": DATA_MANIFEST_CONTENT_HASH_CONTRACT,
        "status": "pass",
        "mode": "full",
        "content_hashes": True,
        "errors": [],
        "experiment_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "verifier_sha256": hashlib.sha256(verifier.read_bytes()).hexdigest(),
        "data_root": str(data_root.resolve()),
        "pdes": ["poisson"],
        "sample_manifest_record_count": 1,
        "sample_manifest_sha256": hashlib.sha256(sample_content).hexdigest(),
        "results": [
            {
                "pde": "poisson",
                "status": "pass",
                "issues": [],
                "global_id_overlaps": [],
                "field_hash_overlaps": [],
                "content_hash_overlaps": [],
                "splits": [
                    {
                        "pde": "poisson",
                        "split": "test",
                        "requested_count": 1,
                        "loaded_count": 1,
                        "coverage_complete": True,
                        "scan_target_complete": True,
                        "content_hashes": True,
                        "global_ids_complete": True,
                        "duplicate_global_id_count": 0,
                        "source_files": [signature],
                    }
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_eval_command_has_independent_identity_and_explicit_source(tmp_path: Path, monkeypatch):
    config = tmp_path / "paper.yaml"
    config.write_text("method:\n  width: 8\n", encoding="utf-8")
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"small test checkpoint")
    data_root = tmp_path / "PDEdata"
    manifest = _write_full_manifest(
        tmp_path / "data_protocol_report.json", data_root, config
    )
    template = _template(config) | {
        "experiment_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "data_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "data_manifest_path": str(manifest),
    }
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("PYTHON", "python")
    monkeypatch.setenv("SAVE_CHECKPOINT", "off")

    train = make_provenance_row(
        template,
        execution_mode="train",
        output_dir=tmp_path / "train",
        run_label="train",
        seed=1,
        sensor_seed=1,
    )
    evaluation = make_provenance_row(
        train,
        execution_mode="eval_only",
        output_dir=tmp_path / "eval",
        run_label="eval_testseed2",
        seed=1,
        sensor_seed=2,
        source_train_run_id=train["run_id"],
        source_train_run_fingerprint=train["run_fingerprint"],
        source_train_seed=1,
        checkpoint_path=checkpoint,
    )
    command = build_command(evaluation)
    parsed = parse_baseline_args(command[3:])

    assert evaluation["execution_mode"] == "eval_only"
    assert evaluation["run_id"] != train["run_id"]
    assert evaluation["run_fingerprint"] != train["run_fingerprint"]
    assert parsed.eval_only is True
    assert parsed.execution_mode == "eval_only"
    assert parsed.seed == 1
    assert parsed.sensor_seed == 2
    assert parsed.source_train_seed == 1
    assert parsed.source_train_run_id == train["run_id"]
    assert parsed.source_train_run_fingerprint == train["run_fingerprint"]
    assert evaluation["data_manifest_sha256"] == train["data_manifest_sha256"]
    assert evaluation["data_manifest_path"] == train["data_manifest_path"]
    assert parsed.data_manifest_sha256 == train["data_manifest_sha256"]
    assert parsed.data_manifest_path == train["data_manifest_path"]
    assert parsed.checkpoint == str(checkpoint)
    assert "--no-save-checkpoint" in command
    assert command.count("--eval-only") == 1


def test_eval_fingerprint_changes_with_checkpoint_or_sensor_seed(tmp_path: Path):
    config = tmp_path / "paper.yaml"
    config.write_text("method: {}\n", encoding="utf-8")
    first_checkpoint = tmp_path / "first.pt"
    second_checkpoint = tmp_path / "second.pt"
    first_checkpoint.write_bytes(b"one")
    second_checkpoint.write_bytes(b"two")
    train = make_provenance_row(
        _template(config),
        execution_mode="train",
        output_dir=tmp_path / "train",
        run_label="train",
        seed=1,
        sensor_seed=1,
    )

    def evaluation(checkpoint: Path, sensor_seed: int) -> dict:
        return make_provenance_row(
            train,
            execution_mode="eval_only",
            output_dir=tmp_path / f"eval-{sensor_seed}",
            run_label="eval",
            seed=1,
            sensor_seed=sensor_seed,
            source_train_run_id=train["run_id"],
            source_train_run_fingerprint=train["run_fingerprint"],
            source_train_seed=1,
            checkpoint_path=checkpoint,
        )

    first = evaluation(first_checkpoint, 2)
    assert evaluation(first_checkpoint, 3)["run_fingerprint"] != first["run_fingerprint"]
    assert evaluation(second_checkpoint, 2)["run_fingerprint"] != first["run_fingerprint"]


def test_legacy_source_identity_is_rejected():
    with pytest.raises(ValueError, match="legacy/unverifiable"):
        require_source_identity({"run_id": "legacy", "seed": 1})
