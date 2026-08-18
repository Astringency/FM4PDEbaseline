from __future__ import annotations

import hashlib
import json
from pathlib import Path

import scipy.io

from baselines.common.data_adapter import build_default_registry
from scripts.verify_data_protocol import (
    ScanOptions,
    SplitRequest,
    configured_pdes,
    run_audit,
    scan_split,
    write_outputs,
)


def _config() -> dict:
    return {
        "name": "tiny_protocol",
        "pdes": ["poisson"],
        "train_size": 3,
        "val_size": 1,
        "test_size": 2,
        "train_shards": 1,
    }


def _config_file(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_configured_pdes_is_stable_union():
    config = {
        "pdes": ["poisson", "darcy"],
        "task_group_overrides": {"one": {"pdes": ["darcy", "burger"]}},
    }
    assert configured_pdes(config) == ["poisson", "darcy", "burger"]


def test_bounded_audit_mirrors_train_tail_validation_and_is_disjoint(tiny_data_root, tmp_path: Path):
    config = _config()
    config_path = _config_file(tmp_path, config)
    report, samples = run_audit(
        config,
        config_path=config_path,
        data_root=tiny_data_root,
        output_dir=tmp_path / "out",
        options=ScanOptions(sample_limit=2, chunk_size=1, content_hashes=True),
    )

    assert report["status"] == "pass"
    result = report["results"][0]
    assert result["split_selection"]["val_source"] == "deterministic_train_subset"
    assert result["split_selection"]["val_from_train_offset"] == 2
    assert not result["global_id_overlaps"]
    assert not result["content_hash_overlaps"]
    by_split = {row["split"]: row for row in result["splits"]}
    assert by_split["train"]["loaded_count"] == 2
    assert by_split["val"]["loaded_count"] == 1
    assert by_split["test"]["loaded_count"] == 2
    assert all(sample["content_sha256"] for sample in samples)


def test_independent_validation_still_counts_inside_the_configured_training_budget(tiny_data_root, tmp_path: Path):
    train_path = tiny_data_root / "poisson/poisson_10000-128-128_1.mat"
    train = scipy.io.loadmat(train_path)
    scipy.io.savemat(
        tiny_data_root / "poisson/poisson_val_10000-128-128.mat",
        {"f_data": train["f_data"][-1:], "phi_data": train["phi_data"][-1:]},
    )
    config = _config()
    report, _ = run_audit(
        config,
        config_path=_config_file(tmp_path, config),
        data_root=tiny_data_root,
        output_dir=tmp_path / "out",
        options=ScanOptions(sample_limit=3, chunk_size=1, content_hashes=False),
    )

    selection = report["results"][0]["split_selection"]
    assert selection["val_source"] == "independent_val"
    assert selection["effective_train_count"] == 2


def test_content_hash_detects_same_sample_under_different_global_ids(tiny_data_root, tmp_path: Path):
    train_path = tiny_data_root / "poisson/poisson_10000-128-128_1.mat"
    test_path = tiny_data_root / "poisson/poisson_test_10000-128-128.mat"
    train = scipy.io.loadmat(train_path)
    test = scipy.io.loadmat(test_path)
    test["f_data"][0] = train["f_data"][0]
    test["phi_data"][0] = train["phi_data"][0]
    scipy.io.savemat(test_path, {"f_data": test["f_data"], "phi_data": test["phi_data"]})

    config = _config()
    config_path = _config_file(tmp_path, config)
    report, _ = run_audit(
        config,
        config_path=config_path,
        data_root=tiny_data_root,
        output_dir=tmp_path / "out",
        options=ScanOptions(sample_limit=2, chunk_size=2, content_hashes=True, use_cache=False),
    )

    result = report["results"][0]
    assert report["status"] == "fail"
    assert "cross_split_complete_content_overlap" in result["issues"]
    assert not result["global_id_overlaps"]


def test_split_manifest_cache_reuses_unchanged_sources(tiny_data_root, tmp_path: Path, monkeypatch):
    registry = build_default_registry()
    request = SplitRequest(
        pde="poisson",
        split="test",
        requested_count=2,
        scan_count=2,
        train_shards=1,
    )
    options = ScanOptions(sample_limit=2, chunk_size=1, content_hashes=True)
    first, first_samples = scan_split(registry, request, tiny_data_root, options, tmp_path / "cache")
    assert first["cache_hit"] is False

    def unexpected_load(*args, **kwargs):
        raise AssertionError("valid cache should avoid loading the dataset")

    monkeypatch.setattr(registry, "load_raw", unexpected_load)
    second, second_samples = scan_split(registry, request, tiny_data_root, options, tmp_path / "cache")
    assert second["cache_hit"] is True
    assert second_samples == first_samples


def test_write_outputs_emits_json_and_csv_provenance(tiny_data_root, tmp_path: Path):
    config = _config()
    config_path = _config_file(tmp_path, config)
    report, samples = run_audit(
        config,
        config_path=config_path,
        data_root=tiny_data_root,
        output_dir=tmp_path / "audit",
        options=ScanOptions(sample_limit=1, chunk_size=1),
    )
    paths = write_outputs(tmp_path / "audit", report, samples)

    assert all(Path(path).is_file() for path in paths.values())
    saved = json.loads(Path(paths["report_json"]).read_text(encoding="utf-8"))
    assert saved["report_schema_version"] == "fm4pde-data-protocol-report-v1"
    sample_manifest = Path(paths["manifest_jsonl"])
    assert saved["sample_manifest_record_count"] == len(
        sample_manifest.read_text(encoding="utf-8").splitlines()
    )
    assert saved["sample_manifest_sha256"] == hashlib.sha256(
        sample_manifest.read_bytes()
    ).hexdigest()
