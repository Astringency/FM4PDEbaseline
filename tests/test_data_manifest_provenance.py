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
    DATA_MANIFEST_CONTENT_HASH_CONTRACT,
    DATA_MANIFEST_REPORT_SCHEMA_VERSION,
    DEFAULT_TASK_PROTOCOL_VERSION,
    run_fingerprint,
    sha256_file,
    summary_validation_reasons,
    validate_full_data_manifest,
)
from scripts.experiments.run_one import build_command


def _write_manifest(
    path: Path,
    *,
    status: str = "pass",
    mode: str = "full",
    revision: int = 1,
    experiment_config: Path | None = None,
    data_root: Path | None = None,
    pde: str = "darcy",
) -> Path:
    if status != "pass" or mode != "full":
        path.write_text(
            json.dumps({"status": status, "mode": mode, "revision": revision}) + "\n",
            encoding="utf-8",
        )
        return path
    if experiment_config is None or data_root is None:
        raise ValueError("passing manifest fixture requires experiment_config and data_root")
    source = data_root / pde / "fixture.bin"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"verified-pde-fixture")
    stat = source.stat()
    signature = {
        "path": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }
    splits = [
        {
            "pde": pde,
            "split": split,
            "requested_count": count,
            "loaded_count": count,
            "coverage_complete": True,
            "scan_target_complete": True,
            "content_hashes": True,
            "global_ids_complete": True,
            "duplicate_global_id_count": 0,
            "source_files": [signature],
        }
        for split, count in (("train", 4), ("test", 2))
    ]
    sample_rows = []
    for split, count in (("train", 4), ("test", 2)):
        for ordinal in range(count):
            identity = f"{pde}:{split}:{ordinal}"
            sample_rows.append(
                {
                    "pde": pde,
                    "split": split,
                    "source_kind": "fixture",
                    "ordinal": ordinal,
                    "sample_index": ordinal,
                    "global_sample_id": identity,
                    "field_sha256": hashlib.sha256(
                        f"field:{identity}".encode()
                    ).hexdigest(),
                    "content_sha256": hashlib.sha256(
                        f"content:{identity}".encode()
                    ).hexdigest(),
                }
            )
    sample_content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in sample_rows
    ).encode()
    path.with_name("sample_manifest.jsonl").write_bytes(sample_content)
    verifier = Path(__file__).resolve().parents[1] / "scripts" / "verify_data_protocol.py"
    payload = {
        "report_schema_version": DATA_MANIFEST_REPORT_SCHEMA_VERSION,
        "content_hash_contract": DATA_MANIFEST_CONTENT_HASH_CONTRACT,
        "status": status,
        "mode": mode,
        "revision": revision,
        "content_hashes": True,
        "errors": [],
        "experiment_config_sha256": sha256_file(experiment_config),
        "verifier_sha256": sha256_file(verifier),
        "data_root": str(data_root.resolve()),
        "pdes": [pde],
        "sample_manifest_record_count": len(sample_rows),
        "sample_manifest_sha256": hashlib.sha256(sample_content).hexdigest(),
        "results": [
            {
                "pde": pde,
                "status": "pass",
                "issues": [],
                "splits": splits,
                "global_id_overlaps": [],
                "field_hash_overlaps": [],
                "content_hash_overlaps": [],
            }
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
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


def _write_experiment_config(tmp_path: Path, method_config: Path) -> tuple[dict, Path]:
    cfg = _config(method_config)
    path = tmp_path / "experiment.yaml"
    path.write_text(json.dumps(cfg, sort_keys=True), encoding="utf-8")
    return cfg, path


def _summary(row: dict) -> dict:
    checkpoint = Path(row["output_dir"]) / "model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"formal-training-checkpoint")
    data_root = ""
    if row.get("data_manifest_path"):
        data_root = json.loads(
            Path(row["data_manifest_path"]).read_text(encoding="utf-8")
        )["data_root"]
    return {
        "status": "success",
        "matrix_schema_version": row["matrix_schema_version"],
        "summary_schema_version": row["summary_schema_version"],
        "run_id": row["run_id"],
        "run_fingerprint": row["run_fingerprint"],
        "execution_mode": row["execution_mode"],
        "eval_only": row["execution_mode"] == "eval_only",
        "comparison_track": row["comparison_track"],
        "config_content_sha256": row["config_content_sha256"],
        "experiment_config_sha256": row["experiment_config_sha256"],
        "config_hash": "resolved-config-sha1",
        "data_manifest_sha256": row["data_manifest_sha256"],
        "data_manifest_path": row["data_manifest_path"],
        "data_root": data_root,
        "commit_hash": row["commit_hash"],
        "task_protocol_version": row["task_protocol_version"],
        "sensor_protocol_version": row["sensor_protocol_version"],
        "sensor_seed": row["sensor_seed"],
        "task_group": row["task_group"],
        "task": row["task"],
        "pde": row["pde"],
        "baseline": row["baseline"],
        "seed": row["seed"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
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
        "data_loading_mode_requested": row["data_loading_mode"],
        "num_workers": row["num_workers"],
        "pin_memory_requested": row["pin_memory"],
        "persistent_workers_requested": row["persistent_workers"],
        "prefetch_factor_requested": row["prefetch_factor"],
        "load_full_trajectory": row["load_full_trajectory"],
    }


def test_matrix_accepts_only_passing_full_manifest_and_binds_its_content(tmp_path: Path):
    method_config = tmp_path / "method.yaml"
    method_config.write_text("method:\n  width: 4\n", encoding="utf-8")
    cfg, experiment_config = _write_experiment_config(tmp_path, method_config)
    data_root = tmp_path / "PDEdata"

    bounded = _write_manifest(tmp_path / "bounded.json", mode="bounded")
    failed = _write_manifest(tmp_path / "failed.json", status="fail")
    with pytest.raises(ValueError, match="mode='full'"):
        load_data_manifest_binding(bounded)
    with pytest.raises(ValueError, match="status='pass'"):
        load_data_manifest_binding(failed)

    manifest = _write_manifest(
        tmp_path / "data_protocol_report.json",
        experiment_config=experiment_config,
        data_root=data_root,
    )
    parsed = parse_matrix_args(["--config", str(method_config), "--data-manifest", str(manifest)])
    assert parsed.data_manifest == str(manifest)
    rows, skipped, summary = build_matrix(
        cfg,
        tmp_path / "bound-output",
        "manifest_test",
        data_manifest=manifest,
        experiment_config_path=experiment_config,
    )
    assert not skipped
    row = rows[0]
    expected_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert row["data_manifest_path"] == str(manifest.resolve())
    assert row["data_manifest_sha256"] == expected_sha256
    assert row["experiment_config_sha256"] == sha256_file(experiment_config)
    assert summary["data_manifest_sha256"] == expected_sha256

    previous_fingerprint = row["run_fingerprint"]
    _write_manifest(
        manifest,
        revision=2,
        experiment_config=experiment_config,
        data_root=data_root,
    )
    changed_rows, _, _ = build_matrix(
        cfg,
        tmp_path / "changed-output",
        "manifest_test",
        data_manifest=manifest,
        experiment_config_path=experiment_config,
    )
    assert changed_rows[0]["run_fingerprint"] != previous_fingerprint

    design_rows, _, _ = build_matrix(cfg, tmp_path / "design-output", "manifest_test")
    assert design_rows[0]["data_manifest_sha256"] == ""
    assert design_rows[0]["data_manifest_path"] == ""


def test_formal_matrix_rejects_shell_design_overrides(tmp_path: Path, monkeypatch):
    method_config = tmp_path / "method.yaml"
    method_config.write_text("method:\n  width: 4\n", encoding="utf-8")
    cfg, experiment_config = _write_experiment_config(tmp_path, method_config)
    manifest = _write_manifest(
        tmp_path / "data_protocol_report.json",
        experiment_config=experiment_config,
        data_root=tmp_path / "PDEdata",
    )
    monkeypatch.setenv("SEEDS", "99")

    with pytest.raises(ValueError, match="forbids design environment overrides"):
        build_matrix(
            cfg,
            tmp_path / "output",
            "manifest_test",
            data_manifest=manifest,
            experiment_config_path=experiment_config,
        )


def test_run_one_forwards_bound_manifest_and_rejects_manifest_drift(tmp_path: Path, monkeypatch):
    method_config = tmp_path / "method.yaml"
    method_config.write_text("method:\n  width: 4\n", encoding="utf-8")
    cfg, experiment_config = _write_experiment_config(tmp_path, method_config)
    data_root = tmp_path / "PDEdata"
    manifest = _write_manifest(
        tmp_path / "data_protocol_report.json",
        experiment_config=experiment_config,
        data_root=data_root,
    )
    rows, _, _ = build_matrix(
        cfg,
        tmp_path / "output",
        "manifest_test",
        data_manifest=manifest,
        experiment_config_path=experiment_config,
    )
    row = rows[0]
    monkeypatch.setenv("DATA_ROOT", str(data_root))

    command = build_command(row)
    assert command[command.index("--data-manifest-sha256") + 1] == row["data_manifest_sha256"]
    assert command[command.index("--data-manifest-path") + 1] == row["data_manifest_path"]
    assert command[command.index("--experiment-config-sha256") + 1] == row[
        "experiment_config_sha256"
    ]
    parsed = parse_baseline_args(command[3:])
    assert parsed.data_manifest_sha256 == row["data_manifest_sha256"]
    assert parsed.data_manifest_path == row["data_manifest_path"]
    assert parsed.experiment_config_sha256 == row["experiment_config_sha256"]

    wrong_design_row = dict(row, experiment_config_sha256="0" * 64)
    wrong_design_row["run_fingerprint"] = run_fingerprint(wrong_design_row)
    with pytest.raises(RuntimeError, match="experiment_config_sha256"):
        build_command(wrong_design_row)

    _write_manifest(
        manifest,
        revision=2,
        experiment_config=experiment_config,
        data_root=data_root,
    )
    with pytest.raises(RuntimeError, match="data manifest SHA-256"):
        build_command(row)


def test_manifest_binding_rejects_wrong_design_pde_root_and_stale_source(
    tmp_path: Path, monkeypatch
):
    method_config = tmp_path / "method.yaml"
    method_config.write_text("method:\n  width: 4\n", encoding="utf-8")
    cfg, experiment_config = _write_experiment_config(tmp_path, method_config)
    data_root = tmp_path / "PDEdata"
    manifest = _write_manifest(
        tmp_path / "data_protocol_report.json",
        experiment_config=experiment_config,
        data_root=data_root,
    )

    wrong_design = tmp_path / "wrong-experiment.yaml"
    wrong_design.write_text("pdes: [darcy]\ntrain_size: 999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="experiment_config_sha256"):
        load_data_manifest_binding(
            manifest,
            experiment_config_path=wrong_design,
            expected_pdes=["darcy"],
        )
    with pytest.raises(ValueError, match="PDE cohort"):
        load_data_manifest_binding(
            manifest,
            experiment_config_path=experiment_config,
            expected_pdes=["poisson"],
        )
    with pytest.raises(ValueError, match="runtime data_root"):
        validate_full_data_manifest(
            manifest,
            expected_data_root=tmp_path / "different-data",
        )

    rows, _, _ = build_matrix(
        cfg,
        tmp_path / "output",
        "manifest_test",
        data_manifest=manifest,
        experiment_config_path=experiment_config,
    )
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "different-data"))
    with pytest.raises(RuntimeError, match="runtime data_root"):
        build_command(rows[0])

    source = data_root / "darcy" / "fixture.bin"
    source.write_bytes(b"changed-after-audit")
    with pytest.raises(ValueError, match="changed after verification"):
        validate_full_data_manifest(
            manifest,
            expected_data_root=data_root,
            verify_source_signatures=True,
        )


def test_manifest_binding_rejects_missing_tampered_and_cross_split_sample_evidence(
    tmp_path: Path,
):
    method_config = tmp_path / "method.yaml"
    method_config.write_text("method:\n  width: 4\n", encoding="utf-8")
    _cfg, experiment_config = _write_experiment_config(tmp_path, method_config)
    data_root = tmp_path / "PDEdata"
    manifest = _write_manifest(
        tmp_path / "data_protocol_report.json",
        experiment_config=experiment_config,
        data_root=data_root,
    )
    sample_path = manifest.with_name("sample_manifest.jsonl")
    original_sample = sample_path.read_bytes()

    sample_path.unlink()
    with pytest.raises(ValueError, match="sample evidence is missing"):
        validate_full_data_manifest(manifest)

    sample_path.write_bytes(original_sample + b"\n")
    with pytest.raises(ValueError, match="sample manifest SHA-256 mismatch"):
        validate_full_data_manifest(manifest)

    rows = [json.loads(line) for line in original_sample.splitlines()]
    rows[-1]["global_sample_id"] = rows[0]["global_sample_id"]
    malicious_sample = b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        for row in rows
    )
    sample_path.write_bytes(malicious_sample)
    report = json.loads(manifest.read_text(encoding="utf-8"))
    report["sample_manifest_sha256"] = hashlib.sha256(malicious_sample).hexdigest()
    manifest.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cross-split global_sample_id overlap"):
        validate_full_data_manifest(manifest)


def test_formal_summary_and_checkpoint_validation_require_matching_manifest(tmp_path: Path):
    method_config = tmp_path / "method.yaml"
    method_config.write_text("method:\n  width: 4\n", encoding="utf-8")
    cfg, experiment_config = _write_experiment_config(tmp_path, method_config)
    data_root = tmp_path / "PDEdata"
    manifest = _write_manifest(
        tmp_path / "data_protocol_report.json",
        experiment_config=experiment_config,
        data_root=data_root,
    )
    rows, _, _ = build_matrix(
        cfg,
        tmp_path / "output",
        "manifest_test",
        data_manifest=manifest,
        experiment_config_path=experiment_config,
    )
    row = rows[0]
    summary = _summary(row)
    assert summary_validation_reasons(row, summary) == []

    checkpoint = Path(summary["checkpoint_path"])
    checkpoint.write_bytes(b"tampered-checkpoint")
    assert "checkpoint_hash_mismatch" in summary_validation_reasons(row, summary)
    checkpoint.write_bytes(b"formal-training-checkpoint")
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

    with pytest.raises(ValueError, match="--data-manifest-path"):
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
                "--data-manifest-sha256",
                "0" * 64,
            ]
        )


def test_core_runner_rejects_a_forged_method_config_hash(tmp_path: Path):
    config = tmp_path / "method.yaml"
    config.write_text("method:\n  width: 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the loaded configuration"):
        run_baseline(
            [
                "--baseline",
                "fno",
                "--pde",
                "poisson",
                "--config",
                str(config),
                "--config-content-sha256",
                "0" * 64,
                "--output-dir",
                str(tmp_path / "run"),
            ]
        )

def test_bound_manifest_is_written_to_checkpoint_raw_rows_and_summary(tmp_path: Path):
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
    data_root = tmp_path / "PDEdata"
    manifest = _write_manifest(
        tmp_path / "data_protocol_report.json",
        experiment_config=config,
        data_root=data_root,
        pde="poisson",
    )
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
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
            "--data-root",
            str(data_root),
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
