#!/usr/bin/env python
"""Verify FM4PDE train/validation/test split provenance without changing data.

The default scan is intentionally bounded.  ``--full`` is the fail-closed
paper-validation mode: it scans every configured sample, computes exact sample
content hashes, and requires complete coverage.  Hash manifests are cached by
scan parameters and source-file size/mtime/ctime so an unchanged full scan can be
reused safely.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.common.data_adapter import PDEDataRegistry, build_default_registry
from baselines.common.data_files import files_for_pde, load_data_files_from_config


REPORT_SCHEMA_VERSION = "fm4pde-data-protocol-report-v1"
MANIFEST_SCHEMA_VERSION = "fm4pde-sample-manifest-v1"
CONTENT_HASH_CONTRACT = "full_tensor+sample_indexed_physical_metadata-v1"
DEFAULT_SAMPLE_LIMIT = 32
DEFAULT_CHUNK_SIZE = 10_000
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitRequest:
    pde: str
    split: str
    requested_count: int
    scan_count: int
    sample_offset: int = 0
    val_from_train_offset: int | None = None
    source_kind: str = "native"
    train_shards: int = 5
    scalar_param_mode: str = "metadata"
    load_full_trajectory: bool = True
    data_files: dict[str, list[str]] | None = None

    def cache_identity(self, data_root: Path, *, content_hashes: bool) -> dict[str, Any]:
        return {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "data_root": str(data_root.resolve()),
            **asdict(self),
            "content_hashes": bool(content_hashes),
            "content_hash_contract": CONTENT_HASH_CONTRACT,
        }


@dataclass(frozen=True)
class ScanOptions:
    full: bool = False
    sample_limit: int = DEFAULT_SAMPLE_LIMIT
    chunk_size: int = DEFAULT_CHUNK_SIZE
    content_hashes: bool = False
    use_cache: bool = True

    @property
    def hashes_enabled(self) -> bool:
        return bool(self.full or self.content_hashes)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Verify configured FM4PDE data splits and optionally hash every sample."
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/main_results.yaml",
        help="Experiment YAML whose PDEs and split sizes define the audit cohort.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT", ""),
        help="Dataset root override. Defaults to DATA_ROOT, experiment YAML, then baseline config.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to outputs/data_protocol/<experiment-name>.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=DEFAULT_SAMPLE_LIMIT,
        help="Maximum samples per split in the default safe scan.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum samples materialized per loader call.",
    )
    parser.add_argument(
        "--content-hashes",
        action="store_true",
        help="Hash all samples reached by the bounded scan as well as checking IDs.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Paper validation: scan configured counts, hash every sample, and require complete coverage.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Ignore and replace reusable split manifests.")
    parser.add_argument(
        "--pde",
        action="append",
        dest="pdes",
        help="Restrict the audit to one or more PDEs (repeatable).",
    )
    return parser.parse_args(argv)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return value


def configured_pdes(config: Mapping[str, Any]) -> list[str]:
    """Return the stable union of top-level and task-group PDE declarations."""
    pdes: list[str] = []

    def add(values: Any) -> None:
        if values is None or (isinstance(values, str) and values.lower() == "all"):
            return
        if isinstance(values, str):
            values = [values]
        for value in values:
            name = str(value)
            if name not in pdes:
                pdes.append(name)

    add(config.get("pdes"))
    overrides = config.get("task_group_overrides", {}) or {}
    if isinstance(overrides, Mapping):
        for group in overrides.values():
            if isinstance(group, Mapping):
                add(group.get("pdes"))
    if not pdes:
        raise ValueError("experiment config must declare an explicit PDE cohort")
    return pdes


def resolve_data_root(
    experiment_config: Mapping[str, Any],
    experiment_path: str | Path,
    override: str = "",
) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    if experiment_config.get("data_root"):
        return Path(str(experiment_config["data_root"])).expanduser().resolve()
    baseline_config_value = str(experiment_config.get("config", "") or "")
    if baseline_config_value:
        experiment_path = Path(experiment_path).resolve()
        candidates = [Path(baseline_config_value)]
        if not Path(baseline_config_value).is_absolute():
            candidates.extend((ROOT / baseline_config_value, experiment_path.parent / baseline_config_value))
        baseline_path = next((path for path in candidates if path.is_file()), None)
        if baseline_path is not None:
            baseline_config = load_yaml(baseline_path)
            if baseline_config.get("data_root"):
                return Path(str(baseline_config["data_root"])).expanduser().resolve()
    raise ValueError("data root is not configured; pass --data-root or set DATA_ROOT")


def _scan_count(requested_count: int, options: ScanOptions) -> int:
    if requested_count <= 0:
        return 0
    return requested_count if options.full else min(requested_count, options.sample_limit)


def _probe_validation_source(
    registry: PDEDataRegistry,
    *,
    pde: str,
    data_root: Path,
    train_size: int,
    val_size: int,
    train_shards: int,
    scalar_param_mode: str,
    load_full_trajectory: bool,
    data_files: dict[str, list[str]] | None,
) -> tuple[str, int | None, int]:
    """Mirror baselines.run: prefer native val, otherwise reserve train tail."""
    if val_size <= 0:
        return "none", None, train_size
    try:
        registry.load_raw(
            pde,
            data_root,
            split="val",
            max_samples=1,
            train_shards=train_shards,
            scalar_param_mode=scalar_param_mode,
            load_full_trajectory=load_full_trajectory,
            strict_size=False,
            data_files=data_files,
        )
        if train_size <= val_size:
            raise ValueError(
                f"{pde}: cannot include val_size={val_size} inside train_size={train_size}"
            )
        return "independent_val", None, train_size - val_size
    except FileNotFoundError:
        if train_size <= val_size:
            raise ValueError(
                f"{pde}: cannot reserve val_size={val_size} from train_size={train_size}"
            )
        offset = train_size - val_size
        # Probe the exact tail location now, rather than accepting a nominal
        # offset that the configured shards cannot satisfy.
        registry.load_raw(
            pde,
            data_root,
            split="val",
            max_samples=1,
            train_shards=train_shards,
            val_from_train_offset=offset,
            scalar_param_mode=scalar_param_mode,
            load_full_trajectory=load_full_trajectory,
            strict_size=True,
            data_files=data_files,
        )
        return "deterministic_train_subset", offset, offset


def build_requests_for_pde(
    registry: PDEDataRegistry,
    pde: str,
    data_root: Path,
    config: Mapping[str, Any],
    options: ScanOptions,
) -> tuple[list[SplitRequest], dict[str, Any]]:
    train_size = int(config.get("train_size", 50_000))
    val_size = int(config.get("val_size", 0))
    test_size = int(config.get("test_size", 10_000))
    train_shards = int(config.get("train_shards", 5))
    scalar_param_mode = str(config.get("scalar_param_mode", "metadata") or "metadata")
    # Content identity should cover the whole physical sample, not only the
    # endpoint representation chosen by a particular method.
    load_full_trajectory = bool(options.hashes_enabled or config.get("load_full_trajectory", False))
    configured_files, _config_path, _config_sha256 = load_data_files_from_config(
        config, repository_root=ROOT
    )
    data_files = files_for_pde(configured_files, pde)
    if configured_files and data_files is None:
        raise ValueError(f"data_files_config does not define required PDE {pde!r}")
    val_source, val_offset, effective_train_size = _probe_validation_source(
        registry,
        pde=pde,
        data_root=data_root,
        train_size=train_size,
        val_size=val_size,
        train_shards=train_shards,
        scalar_param_mode=scalar_param_mode,
        load_full_trajectory=load_full_trajectory,
        data_files=data_files,
    )
    requests = [
        SplitRequest(
            pde=pde,
            split="train",
            requested_count=effective_train_size,
            scan_count=_scan_count(effective_train_size, options),
            train_shards=train_shards,
            scalar_param_mode=scalar_param_mode,
            load_full_trajectory=load_full_trajectory,
            data_files=data_files,
        )
    ]
    if val_size > 0:
        requests.append(
            SplitRequest(
                pde=pde,
                split="val",
                requested_count=val_size,
                scan_count=_scan_count(val_size, options),
                val_from_train_offset=val_offset,
                source_kind=val_source,
                train_shards=train_shards,
                scalar_param_mode=scalar_param_mode,
                load_full_trajectory=load_full_trajectory,
                data_files=data_files,
            )
        )
    if test_size > 0:
        requests.append(
            SplitRequest(
                pde=pde,
                split="test",
                requested_count=test_size,
                scan_count=_scan_count(test_size, options),
                train_shards=train_shards,
                scalar_param_mode=scalar_param_mode,
                load_full_trajectory=load_full_trajectory,
                data_files=data_files,
            )
        )
    return requests, {
        "train_requested_count": train_size,
        "effective_train_count": effective_train_size,
        "val_requested_count": val_size,
        "test_requested_count": test_size,
        "val_source": val_source,
        "val_from_train_offset": val_offset,
    }


def _tensor_bytes(tensor: torch.Tensor | np.ndarray | Any) -> tuple[str, tuple[int, ...], bytes]:
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(np.asarray(tensor))
    return str(array.dtype), tuple(int(v) for v in array.shape), array.tobytes(order="C")


def _update_hash_with_value(digest: Any, name: str, value: Any) -> None:
    dtype, shape, raw = _tensor_bytes(value)
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(dtype.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(raw)


def sample_hashes(raw: Mapping[str, Any], index: int) -> tuple[str, str]:
    """Return (field hash, complete sample hash) for one loaded raw sample."""
    full = raw["full_tensor"]
    field_digest = hashlib.sha256()
    _update_hash_with_value(field_digest, "full_tensor", full[index])

    content_digest = hashlib.sha256()
    _update_hash_with_value(content_digest, "full_tensor", full[index])
    n = int(full.shape[0])
    seen: set[str] = set()
    containers = (
        ("pde_params", raw.get("pde_params", {})),
        ("metadata", raw.get("metadata", {})),
    )
    ignored = {
        "full_tensor",
        "full_trajectory",
        "sample_indices",
        "global_sample_ids",
        "files",
    }
    for prefix, container in containers:
        if not isinstance(container, Mapping):
            continue
        for key in sorted(container):
            if key in ignored:
                continue
            value = container[key]
            if not isinstance(value, (torch.Tensor, np.ndarray)) or value.ndim == 0 or int(value.shape[0]) != n:
                continue
            qualified = f"{prefix}.{key}"
            # pde_params are commonly mirrored into metadata; hash once.
            semantic_key = str(key)
            if semantic_key in seen:
                continue
            seen.add(semantic_key)
            _update_hash_with_value(content_digest, qualified, value[index])
    return field_digest.hexdigest(), content_digest.hexdigest()


def _source_signatures(paths: Iterable[str]) -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = []
    for raw_path in sorted(set(str(path) for path in paths)):
        path = Path(raw_path).resolve()
        stat = path.stat()
        signatures.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "ctime_ns": int(stat.st_ctime_ns),
            }
        )
    return signatures


def _signatures_still_valid(signatures: Sequence[Mapping[str, Any]]) -> bool:
    if not signatures:
        return False
    for signature in signatures:
        try:
            stat = Path(str(signature["path"])).stat()
        except (FileNotFoundError, KeyError):
            return False
        if int(signature.get("size", -1)) != int(stat.st_size):
            return False
        if int(signature.get("mtime_ns", -1)) != int(stat.st_mtime_ns):
            return False
        if int(signature.get("ctime_ns", -1)) != int(stat.st_ctime_ns):
            return False
    return True


def _cache_path(cache_dir: Path, identity: Mapping[str, Any]) -> Path:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:20]
    return cache_dir / f"{identity['pde']}_{identity['split']}_{digest}.jsonl"


def _read_cached_manifest(
    path: Path,
    identity: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        if not first:
            return None
        header = json.loads(first)
        if header.get("record_type") != "header" or header.get("identity") != dict(identity):
            return None
        if not _signatures_still_valid(header.get("source_files", [])):
            return None
        samples = [json.loads(line) for line in handle if line.strip()]
    if len(samples) != int(header.get("loaded_count", -1)):
        return None
    return header, samples


def _write_cached_manifest(path: Path, header: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(header), sort_keys=True) + "\n")
        for sample in samples:
            handle.write(json.dumps(dict(sample), sort_keys=True) + "\n")
    temporary.replace(path)


def _load_request_chunk(
    registry: PDEDataRegistry,
    request: SplitRequest,
    data_root: Path,
    relative_offset: int,
    count: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "split": request.split,
        "max_samples": count,
        "train_shards": request.train_shards,
        "scalar_param_mode": request.scalar_param_mode,
        "load_full_trajectory": request.load_full_trajectory,
        "strict_size": False,
        "data_files": request.data_files,
    }
    if request.source_kind == "deterministic_train_subset":
        assert request.val_from_train_offset is not None
        kwargs["val_from_train_offset"] = request.val_from_train_offset + relative_offset
    else:
        kwargs["sample_offset"] = request.sample_offset + relative_offset
    return registry.load_raw(request.pde, data_root, **kwargs)


def scan_split(
    registry: PDEDataRegistry,
    request: SplitRequest,
    data_root: Path,
    options: ScanOptions,
    cache_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identity = request.cache_identity(data_root, content_hashes=options.hashes_enabled)
    cache_path = _cache_path(cache_dir, identity)
    if options.use_cache:
        cached = _read_cached_manifest(cache_path, identity)
        if cached is not None:
            header, samples = cached
            summary = dict(header["summary"])
            summary["cache_hit"] = True
            summary["cache_path"] = str(cache_path)
            return summary, samples

    samples: list[dict[str, Any]] = []
    source_paths: set[str] = set()
    tensor_shape: list[int] | None = None
    tensor_dtype = ""
    relative_offset = 0
    while relative_offset < request.scan_count:
        count = min(options.chunk_size, request.scan_count - relative_offset)
        raw = _load_request_chunk(registry, request, data_root, relative_offset, count)
        full = raw["full_tensor"]
        loaded = int(full.shape[0])
        if loaded <= 0:
            break
        tensor_shape = [int(v) for v in full.shape[1:]]
        tensor_dtype = str(full.dtype)
        source_paths.update(str(path) for path in raw.get("file_paths", []))
        global_ids = list(raw.get("global_sample_ids", []))
        indices = raw.get("sample_indices")
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().reshape(-1).tolist()
        elif indices is None:
            indices = []
        else:
            indices = list(indices)
        for index in range(loaded):
            global_id = str(global_ids[index]) if index < len(global_ids) else ""
            sample_index = int(indices[index]) if index < len(indices) else request.sample_offset + relative_offset + index
            field_sha256 = content_sha256 = ""
            if options.hashes_enabled:
                field_sha256, content_sha256 = sample_hashes(raw, index)
            samples.append(
                {
                    "pde": request.pde,
                    "split": request.split,
                    "source_kind": request.source_kind,
                    "ordinal": relative_offset + index,
                    "sample_index": sample_index,
                    "global_sample_id": global_id,
                    "field_sha256": field_sha256,
                    "content_sha256": content_sha256,
                }
            )
        relative_offset += loaded
        if loaded < count:
            break

    ids = [sample["global_sample_id"] for sample in samples if sample["global_sample_id"]]
    missing_id_count = len(samples) - len(ids)
    duplicate_id_count = len(ids) - len(set(ids))
    summary = {
        "pde": request.pde,
        "split": request.split,
        "source_kind": request.source_kind,
        "requested_count": request.requested_count,
        "scan_target_count": request.scan_count,
        "loaded_count": len(samples),
        "coverage_complete": len(samples) == request.requested_count,
        "scan_target_complete": len(samples) == request.scan_count,
        "global_ids_complete": missing_id_count == 0,
        "missing_global_id_count": missing_id_count,
        "duplicate_global_id_count": duplicate_id_count,
        "content_hashes": options.hashes_enabled,
        "tensor_shape": tensor_shape or [],
        "tensor_dtype": tensor_dtype,
        "source_files": _source_signatures(source_paths),
        "cache_hit": False,
        "cache_path": str(cache_path),
    }
    header = {
        "record_type": "header",
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "identity": identity,
        "source_files": summary["source_files"],
        "loaded_count": len(samples),
        "summary": summary,
    }
    if options.use_cache and summary["source_files"]:
        _write_cached_manifest(cache_path, header, samples)
    return summary, samples


def _pairwise_overlap(
    samples_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    field: str,
) -> list[dict[str, Any]]:
    overlaps: list[dict[str, Any]] = []
    for left_index, left in enumerate(SPLITS):
        left_values: dict[str, list[str]] = {}
        for row in samples_by_split.get(left, []):
            value = str(row.get(field, "") or "")
            if value:
                left_values.setdefault(value, []).append(str(row.get("global_sample_id", "")))
        for right in SPLITS[left_index + 1 :]:
            right_values: dict[str, list[str]] = {}
            for row in samples_by_split.get(right, []):
                value = str(row.get(field, "") or "")
                if value:
                    right_values.setdefault(value, []).append(str(row.get("global_sample_id", "")))
            common = sorted(set(left_values) & set(right_values))
            if common:
                overlaps.append(
                    {
                        "left_split": left,
                        "right_split": right,
                        "field": field,
                        "overlap_count": len(common),
                        "examples": [
                            {
                                "value": value,
                                "left_ids": left_values[value][:3],
                                "right_ids": right_values[value][:3],
                            }
                            for value in common[:10]
                        ],
                    }
                )
    return overlaps


def audit_pde(
    registry: PDEDataRegistry,
    pde: str,
    data_root: Path,
    config: Mapping[str, Any],
    options: ScanOptions,
    cache_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    requests, split_selection = build_requests_for_pde(registry, pde, data_root, config, options)
    summaries: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    samples_by_split: dict[str, list[dict[str, Any]]] = {}
    for request in requests:
        summary, samples = scan_split(registry, request, data_root, options, cache_dir)
        summaries.append(summary)
        all_samples.extend(samples)
        samples_by_split[request.split] = samples

    id_overlaps = _pairwise_overlap(samples_by_split, "global_sample_id")
    field_overlaps = _pairwise_overlap(samples_by_split, "field_sha256") if options.hashes_enabled else []
    content_overlaps = _pairwise_overlap(samples_by_split, "content_sha256") if options.hashes_enabled else []
    issues: list[str] = []
    for summary in summaries:
        split = summary["split"]
        if not summary["scan_target_complete"]:
            issues.append(f"{split}:scan_target_incomplete")
        if options.full and not summary["coverage_complete"]:
            issues.append(f"{split}:configured_coverage_incomplete")
        if not summary["global_ids_complete"]:
            issues.append(f"{split}:missing_global_ids")
        if summary["duplicate_global_id_count"]:
            issues.append(f"{split}:duplicate_global_ids")
    if id_overlaps:
        issues.append("cross_split_global_id_overlap")
    if field_overlaps:
        issues.append("cross_split_field_content_overlap")
    if content_overlaps:
        issues.append("cross_split_complete_content_overlap")
    result = {
        "pde": pde,
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "split_selection": split_selection,
        "splits": summaries,
        "global_id_overlaps": id_overlaps,
        "field_hash_overlaps": field_overlaps,
        "content_hash_overlaps": content_overlaps,
    }
    return result, all_samples


def run_audit(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    data_root: Path,
    output_dir: Path,
    options: ScanOptions,
    pdes: Sequence[str] | None = None,
    registry: PDEDataRegistry | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry = registry or build_default_registry()
    selected_pdes = list(pdes or configured_pdes(config))
    cache_dir = output_dir / "cache"
    pde_results: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for pde in selected_pdes:
        try:
            result, pde_samples = audit_pde(
                registry, pde, data_root, config, options, cache_dir
            )
        except Exception as exc:  # fail closed while preserving other PDE evidence
            result = {
                "pde": pde,
                "status": "error",
                "issues": ["load_or_validation_error"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "splits": [],
            }
            pde_samples = []
            errors.append({"pde": pde, "error_type": type(exc).__name__, "error": str(exc)})
        pde_results.append(result)
        samples.extend(pde_samples)

    config_path = Path(config_path).resolve()
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    configured_files, data_files_config_path, data_files_config_sha256 = load_data_files_from_config(
        config, repository_root=ROOT
    )
    ordered_samples = _ordered_sample_rows(samples)
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "content_hash_contract": CONTENT_HASH_CONTRACT,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "experiment_name": str(config.get("name", config_path.stem)),
        "experiment_config": str(config_path),
        "experiment_config_sha256": config_sha256,
        "data_files_config": data_files_config_path,
        "data_files_config_sha256": data_files_config_sha256,
        "data_files": configured_files,
        "data_root": str(data_root.resolve()),
        "mode": "full" if options.full else "bounded",
        "content_hashes": options.hashes_enabled,
        "sample_limit": None if options.full else options.sample_limit,
        "chunk_size": options.chunk_size,
        "cache_enabled": options.use_cache,
        "pdes": selected_pdes,
        "sample_manifest_record_count": len(ordered_samples),
        "sample_manifest_sha256": _sample_manifest_sha256(ordered_samples),
        "status": "pass" if all(item["status"] == "pass" for item in pde_results) else "fail",
        "errors": errors,
        "results": pde_results,
    }
    return report, samples


def _ordered_sample_rows(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "pde",
        "split",
        "source_kind",
        "ordinal",
        "sample_index",
        "global_sample_id",
        "field_sha256",
        "content_sha256",
    )
    rows = [{field: sample.get(field, "") for field in fields} for sample in samples]
    return sorted(
        rows,
        key=lambda row: (
            str(row["pde"]),
            str(row["split"]),
            int(row["ordinal"]),
            str(row["global_sample_id"]),
        ),
    )


def _sample_manifest_sha256(samples: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in _ordered_sample_rows(samples):
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_outputs(output_dir: Path, report: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "data_protocol_report.json"
    split_csv_path = output_dir / "data_protocol_splits.csv"
    sample_jsonl_path = output_dir / "sample_manifest.jsonl"
    sample_csv_path = output_dir / "sample_manifest.csv"
    report_path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    split_fields = [
        "pde",
        "split",
        "source_kind",
        "requested_count",
        "scan_target_count",
        "loaded_count",
        "coverage_complete",
        "scan_target_complete",
        "global_ids_complete",
        "missing_global_id_count",
        "duplicate_global_id_count",
        "content_hashes",
        "tensor_shape",
        "tensor_dtype",
        "cache_hit",
        "cache_path",
    ]
    with split_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=split_fields, extrasaction="ignore")
        writer.writeheader()
        for pde_result in report.get("results", []):
            for split in pde_result.get("splits", []):
                row = dict(split)
                row["tensor_shape"] = json.dumps(row.get("tensor_shape", []), separators=(",", ":"))
                writer.writerow(row)

    sample_fields = [
        "pde",
        "split",
        "source_kind",
        "ordinal",
        "sample_index",
        "global_sample_id",
        "field_sha256",
        "content_sha256",
    ]
    with sample_jsonl_path.open("w", encoding="utf-8") as jsonl, sample_csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=sample_fields, extrasaction="ignore")
        writer.writeheader()
        for row in _ordered_sample_rows(samples):
            jsonl.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            writer.writerow(row)
    return {
        "report_json": str(report_path),
        "splits_csv": str(split_csv_path),
        "manifest_jsonl": str(sample_jsonl_path),
        "manifest_csv": str(sample_csv_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sample_limit <= 0:
        raise SystemExit("--sample-limit must be positive")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    data_root = resolve_data_root(config, config_path, args.data_root)
    experiment_name = str(config.get("name", config_path.stem))
    mode_namespace = "full" if args.full else ("bounded_hashes" if args.content_hashes else "bounded_ids")
    output_dir = Path(
        args.output_dir or ROOT / "outputs" / "data_protocol" / experiment_name / mode_namespace
    ).resolve()
    options = ScanOptions(
        full=bool(args.full),
        sample_limit=int(args.sample_limit),
        chunk_size=int(args.chunk_size),
        content_hashes=bool(args.content_hashes),
        use_cache=not bool(args.no_cache),
    )
    report, samples = run_audit(
        config,
        config_path=config_path,
        data_root=data_root,
        output_dir=output_dir,
        options=options,
        pdes=args.pdes,
    )
    paths = write_outputs(output_dir, report, samples)
    print(json.dumps({"status": report["status"], **paths}, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
