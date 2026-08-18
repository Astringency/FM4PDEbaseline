"""Provenance contracts shared by matrix, runner, and result exporters.

Versioned, content-addressed metadata is deliberately strict here: a legacy or
partially matching artifact is useful audit evidence, but it is not a completed
run and must not enter a publishable result table.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


MATRIX_SCHEMA_VERSION = 2
SUMMARY_SCHEMA_VERSION = 2
DEFAULT_TASK_PROTOCOL_VERSION = "fm4pde-task-contract-v3"
DEFAULT_SENSOR_PROTOCOL_VERSION = "fm4pde-sensor-contract-v3"
FORMAL_TASK_PROTOCOL_VERSIONS = {"fm4pde-task-contract-v2", DEFAULT_TASK_PROTOCOL_VERSION}
DATA_MANIFEST_REPORT_SCHEMA_VERSION = "fm4pde-data-protocol-report-v1"
DATA_MANIFEST_CONTENT_HASH_CONTRACT = "full_tensor+sample_indexed_physical_metadata-v1"

# Every field that may change the command, inputs, training, or evaluation is
# part of the fingerprint.  Keep this explicit so adding a matrix column cannot
# silently alter result identity.
FINGERPRINT_FIELDS = (
    "matrix_schema_version",
    "summary_schema_version",
    "execution_mode",
    "comparison_track",
    "experiment_kind",
    "ablation_factor",
    "task_group",
    "task",
    "task_protocol_version",
    "sensor_protocol_version",
    "pde",
    "baseline",
    "seed",
    "sensor_seed",
    "train_size",
    "val_size",
    "test_size",
    "train_shards",
    "batch_size",
    "epochs",
    "device",
    "commit_hash",
    "config_content_sha256",
    "experiment_config_sha256",
    "data_manifest_sha256",
    "num_sensors",
    "sensor_mode",
    "sensor_budget_mode",
    "noise_level",
    "steps",
    "refine_steps",
    "particles",
    "scalar_param_mode",
    "data_loading_mode",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "prefetch_factor",
    "load_full_trajectory",
)
EVAL_FINGERPRINT_FIELDS = (
    "source_train_run_id",
    "source_train_run_fingerprint",
    "source_train_seed",
    "checkpoint_sha256",
)

SUMMARY_IDENTITY_FIELDS = ("task_group", "task", "pde", "baseline", "seed")
AMORTIZED_CHECKPOINT_BASELINES = {
    "fno",
    "deeponet",
    "ifno",
    "recfno",
    "senseiver",
    "voronoicnn",
}
COHORT_FIELDS = (
    "summary_schema_version",
    "execution_mode",
    "comparison_track",
    "task_protocol_version",
    "sensor_protocol_version",
    "config_content_sha256",
    "experiment_config_sha256",
    "data_manifest_sha256",
    "commit_hash",
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_EXPERIMENT_NAMESPACE = "experiment_plan_v2"
IMMUTABLE_HISTORICAL_WORKBOOK = (
    REPOSITORY_ROOT / "outputs" / "experiment_plan_v2_summary.xlsx"
).resolve()
_FULL_DATA_MANIFEST_VALIDATION_CACHE: set[tuple[Any, ...]] = set()


class HistoricalExperimentError(RuntimeError):
    """Raised before a mutating command can touch historical audit evidence."""


def path_contains_namespace(value: str | Path, namespace: str) -> bool:
    """Return whether a normalized path has ``namespace`` as one component."""
    path = Path(value).expanduser()
    try:
        normalized = path.resolve(strict=False)
    except OSError:
        normalized = Path(os.path.abspath(os.path.normpath(path)))
    return namespace in normalized.parts


def reject_historical_experiment_path(
    value: str | Path, *, field: str = "path"
) -> None:
    """Refuse a path that targets the immutable experiment_plan_v2 cohort."""
    path = Path(value).expanduser()
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        resolved = Path(os.path.abspath(os.path.normpath(path)))
    if (
        path.name == f"{HISTORICAL_EXPERIMENT_NAMESPACE}.jsonl"
        or path_contains_namespace(resolved, HISTORICAL_EXPERIMENT_NAMESPACE)
        or resolved == IMMUTABLE_HISTORICAL_WORKBOOK
    ):
        raise HistoricalExperimentError(
            f"refusing historical {HISTORICAL_EXPERIMENT_NAMESPACE} {field}: {path}; "
            "use a corrected namespace and leave historical audit evidence untouched"
        )


def reject_historical_experiment_row(row: Mapping[str, Any]) -> None:
    """Validate every path a row runner may create, move, or overwrite."""
    for field in ("output_dir", "log_dir", "status_file"):
        value = row.get(field)
        if isinstance(value, (str, Path)) and str(value):
            reject_historical_experiment_path(value, field=f"row {field}")

# Matrix fields describe requested execution.  Some summary names predate the
# v2 contract, so keep the mapping explicit and fail closed when either side is
# absent.  This prevents an opaque copied fingerprint from blessing a run that
# actually used a different sample count, sensor layout, budget, or runtime.
SUMMARY_DESIGN_FIELD_MAP = {
    "train_size": "train_requested_size",
    "val_size": "val_requested_size",
    "test_size": "test_requested_size",
    "train_shards": "train_shards",
    "batch_size": "batch_size",
    "epochs": "epochs",
    "device": "device",
    "num_sensors": "num_sensors",
    "sensor_mode": "requested_sensor_mode",
    "sensor_budget_mode": "sensor_budget_mode_requested",
    "noise_level": "noise_level",
    "steps": "steps",
    "refine_steps": "refine_steps",
    "particles": "particles",
    "scalar_param_mode": "scalar_param_mode_requested",
    "data_loading_mode": "data_loading_mode_requested",
    "num_workers": "num_workers",
    "pin_memory": "pin_memory_requested",
    "persistent_workers": "persistent_workers_requested",
    "prefetch_factor": "prefetch_factor_requested",
    "load_full_trajectory": "load_full_trajectory",
}


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path, *, root: str | Path | None = None) -> str:
    resolved = Path(path)
    if not resolved.is_absolute() and root is not None:
        resolved = Path(root) / resolved
    if not resolved.is_file():
        raise FileNotFoundError(f"configuration file not found for provenance hashing: {resolved}")
    return sha256_bytes(resolved.read_bytes())


def validate_full_data_manifest(
    path: str | Path,
    *,
    expected_sha256: str = "",
    expected_experiment_config_sha256: str = "",
    expected_data_root: str | Path | None = None,
    expected_pdes: Iterable[str] | None = None,
    expected_verifier_sha256: str = "",
    verify_source_signatures: bool = False,
) -> tuple[dict[str, Any], str, str]:
    """Validate that a full data report describes the data actually in scope."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"data manifest not found: {manifest_path}")
    observed_sha256 = sha256_file(manifest_path)
    if expected_sha256 and observed_sha256 != str(expected_sha256).lower():
        raise ValueError(
            f"data manifest SHA-256 mismatch: observed={observed_sha256}, "
            f"expected={expected_sha256}, path={manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"data manifest is not readable JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"data manifest must be a JSON object: {manifest_path}")
    if manifest.get("status") != "pass":
        raise ValueError(
            f"data manifest must have status='pass', got {manifest.get('status')!r}: {manifest_path}"
        )
    if manifest.get("mode") != "full":
        raise ValueError(
            f"data manifest must have mode='full', got {manifest.get('mode')!r}: {manifest_path}"
        )
    if manifest.get("report_schema_version") != DATA_MANIFEST_REPORT_SCHEMA_VERSION:
        raise ValueError(
            "data manifest has an unsupported report_schema_version: "
            f"{manifest.get('report_schema_version')!r}"
        )
    if manifest.get("content_hash_contract") != DATA_MANIFEST_CONTENT_HASH_CONTRACT:
        raise ValueError(
            "data manifest has an unsupported content_hash_contract: "
            f"{manifest.get('content_hash_contract')!r}"
        )
    if manifest.get("content_hashes") is not True:
        raise ValueError("full data manifest must record content_hashes=true")
    if manifest.get("errors") not in ([], None):
        raise ValueError("passing data manifest must not contain verifier errors")
    sample_manifest_sha256 = str(manifest.get("sample_manifest_sha256", "") or "").lower()
    _require_sha256(sample_manifest_sha256, "sample_manifest_sha256")
    if int(manifest.get("sample_manifest_record_count", 0) or 0) <= 0:
        raise ValueError("full data manifest must bind a non-empty sample manifest")

    experiment_sha256 = str(manifest.get("experiment_config_sha256", "") or "").lower()
    _require_sha256(experiment_sha256, "experiment_config_sha256")
    if expected_experiment_config_sha256 and experiment_sha256 != str(expected_experiment_config_sha256).lower():
        raise ValueError(
            "data manifest experiment_config_sha256 does not match the matrix design: "
            f"manifest={experiment_sha256}, expected={expected_experiment_config_sha256}"
        )
    verifier_sha256 = str(manifest.get("verifier_sha256", "") or "").lower()
    _require_sha256(verifier_sha256, "verifier_sha256")
    if expected_verifier_sha256 and verifier_sha256 != str(expected_verifier_sha256).lower():
        raise ValueError(
            "data manifest was generated by a different verifier revision: "
            f"manifest={verifier_sha256}, expected={expected_verifier_sha256}"
        )

    manifest_root_text = str(manifest.get("data_root", "") or "")
    if not manifest_root_text:
        raise ValueError("data manifest is missing data_root")
    manifest_root = Path(manifest_root_text).expanduser().resolve()
    if expected_data_root is not None:
        actual_root = Path(expected_data_root).expanduser().resolve()
        if actual_root != manifest_root:
            raise ValueError(
                f"data manifest data_root {manifest_root} does not match runtime data_root {actual_root}"
            )

    manifest_pdes = [str(value) for value in manifest.get("pdes", [])]
    if not manifest_pdes:
        raise ValueError("data manifest must declare a non-empty PDE cohort")
    if len(manifest_pdes) != len(set(manifest_pdes)):
        raise ValueError("data manifest PDE cohort contains duplicates")
    expected_pde_set = None if expected_pdes is None else {str(value) for value in expected_pdes}
    if expected_pde_set is not None and set(manifest_pdes) != expected_pde_set:
        raise ValueError(
            f"data manifest PDE cohort {sorted(manifest_pdes)} does not match matrix "
            f"{sorted(expected_pde_set)}"
        )
    results = manifest.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("data manifest must contain per-PDE results")
    result_pdes = {str(result.get("pde", "")) for result in results if isinstance(result, dict)}
    if result_pdes != set(manifest_pdes):
        raise ValueError(
            f"data manifest result PDEs {sorted(result_pdes)} do not match declared PDEs {sorted(manifest_pdes)}"
        )
    sample_manifest_path = manifest_path.with_name("sample_manifest.jsonl")
    try:
        sample_stat = sample_manifest_path.stat()
    except FileNotFoundError as exc:
        raise ValueError(
            f"full data manifest sample evidence is missing: {sample_manifest_path}"
        ) from exc
    source_stat_key = (
        _current_manifest_source_stat_key(results)
        if verify_source_signatures
        else ()
    )
    cache_key = (
        str(manifest_path),
        observed_sha256,
        int(sample_stat.st_size),
        int(sample_stat.st_mtime_ns),
        int(sample_stat.st_ctime_ns),
        bool(verify_source_signatures),
        source_stat_key,
    )
    if cache_key in _FULL_DATA_MANIFEST_VALIDATION_CACHE:
        return manifest, str(manifest_path), observed_sha256
    loaded_record_count = 0
    expected_sample_counts: dict[tuple[str, str], int] = {}
    for result in results:
        loaded_record_count += _validate_manifest_pde_result(
            result, manifest_root, verify_source_signatures
        )
        pde_name = str(result["pde"])
        for split in result["splits"]:
            expected_sample_counts[(pde_name, str(split["split"]))] = int(
                split["loaded_count"]
            )
    if loaded_record_count != int(manifest.get("sample_manifest_record_count", 0)):
        raise ValueError(
            "data manifest sample count does not equal the per-split evidence: "
            f"manifest={manifest.get('sample_manifest_record_count')}, splits={loaded_record_count}"
        )
    _validate_sample_manifest_file(
        sample_manifest_path,
        expected_sha256=sample_manifest_sha256,
        expected_record_count=loaded_record_count,
        expected_pdes=set(manifest_pdes),
        expected_split_counts=expected_sample_counts,
    )
    _FULL_DATA_MANIFEST_VALIDATION_CACHE.add(cache_key)
    return manifest, str(manifest_path), observed_sha256


def _current_manifest_source_stat_key(results: list[Any]) -> tuple[Any, ...]:
    """Cheap live-state key used to invalidate cached full-manifest checks."""
    states: list[tuple[str, int, int, int]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        for split in result.get("splits", []):
            if not isinstance(split, dict):
                continue
            for signature in split.get("source_files", []):
                if not isinstance(signature, dict):
                    continue
                path = Path(str(signature.get("path", ""))).expanduser().resolve()
                try:
                    stat = path.stat()
                except FileNotFoundError as exc:
                    raise ValueError(f"data manifest source file is missing: {path}") from exc
                states.append(
                    (
                        str(path),
                        int(stat.st_size),
                        int(stat.st_mtime_ns),
                        int(stat.st_ctime_ns),
                    )
                )
    return tuple(sorted(set(states)))


def _validate_manifest_pde_result(
    result: Any,
    data_root: Path,
    verify_source_signatures: bool,
) -> int:
    if not isinstance(result, dict) or result.get("status") != "pass":
        raise ValueError("every data manifest PDE result must have status='pass'")
    if result.get("issues") not in ([], None):
        raise ValueError(f"passing PDE result {result.get('pde')!r} contains issues")
    for overlap_field in ("global_id_overlaps", "field_hash_overlaps", "content_hash_overlaps"):
        if result.get(overlap_field) not in ([], None):
            raise ValueError(f"passing PDE result {result.get('pde')!r} contains {overlap_field}")
    splits = result.get("splits")
    if not isinstance(splits, list) or not splits:
        raise ValueError(f"PDE result {result.get('pde')!r} has no split evidence")
    seen_splits: set[str] = set()
    loaded_total = 0
    for split in splits:
        if not isinstance(split, dict):
            raise ValueError("data manifest split evidence must be an object")
        split_name = str(split.get("split", ""))
        if not split_name or split_name in seen_splits:
            raise ValueError(
                f"PDE result {result.get('pde')!r} has a missing or duplicate split name"
            )
        seen_splits.add(split_name)
        requested = int(split.get("requested_count", -1))
        loaded = int(split.get("loaded_count", -2))
        if requested < 0 or loaded != requested:
            raise ValueError(
                f"data manifest split {result.get('pde')}/{split.get('split')} is incomplete: "
                f"loaded={loaded}, requested={requested}"
            )
        if split.get("coverage_complete") is not True or split.get("scan_target_complete") is not True:
            raise ValueError(f"data manifest split {result.get('pde')}/{split.get('split')} lacks full coverage")
        if split.get("content_hashes") is not True or split.get("global_ids_complete") is not True:
            raise ValueError(f"data manifest split {result.get('pde')}/{split.get('split')} lacks hashes or IDs")
        if int(split.get("duplicate_global_id_count", 0) or 0) != 0:
            raise ValueError(f"data manifest split {result.get('pde')}/{split.get('split')} has duplicate IDs")
        signatures = split.get("source_files")
        if not isinstance(signatures, list) or not signatures:
            raise ValueError(f"data manifest split {result.get('pde')}/{split.get('split')} has no source signatures")
        if verify_source_signatures:
            _validate_source_signatures(signatures, data_root)
        loaded_total += loaded
    return loaded_total


def _validate_source_signatures(signatures: list[Any], data_root: Path) -> None:
    for signature in signatures:
        if not isinstance(signature, dict):
            raise ValueError("data manifest source signature must be an object")
        source_path = Path(str(signature.get("path", ""))).expanduser().resolve()
        if not source_path.is_relative_to(data_root):
            raise ValueError(f"data manifest source file is outside data_root: {source_path}")
        try:
            stat = source_path.stat()
        except FileNotFoundError as exc:
            raise ValueError(f"data manifest source file is missing: {source_path}") from exc
        observed = (int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns))
        expected = (
            int(signature.get("size", -1)),
            int(signature.get("mtime_ns", -1)),
            int(signature.get("ctime_ns", -1)),
        )
        if observed != expected:
            raise ValueError(
                f"data manifest source file changed after verification: {source_path}; "
                f"observed={observed}, expected={expected}"
            )


def _validate_sample_manifest_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_record_count: int,
    expected_pdes: set[str],
    expected_split_counts: Mapping[tuple[str, str], int],
) -> None:
    """Re-hash and structurally validate the report's per-sample evidence."""
    if not path.is_file():
        raise ValueError(f"full data manifest sample evidence is missing: {path}")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read full data sample manifest: {path}") from exc
    observed_sha256 = sha256_bytes(content)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "sample manifest SHA-256 mismatch: "
            f"observed={observed_sha256}, expected={expected_sha256}, path={path}"
        )
    lines = [line for line in content.splitlines() if line.strip()]
    if len(lines) != expected_record_count:
        raise ValueError(
            "sample manifest record count mismatch: "
            f"observed={len(lines)}, expected={expected_record_count}, path={path}"
        )
    identities: set[tuple[str, str, str]] = set()
    split_counts: dict[tuple[str, str], int] = {}
    global_id_splits: dict[tuple[str, str], str] = {}
    field_hash_splits: dict[tuple[str, str], str] = {}
    content_hash_splits: dict[tuple[str, str], str] = {}
    for index, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"sample manifest line {index} is not valid JSON: {path}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"sample manifest line {index} must be an object: {path}")
        pde = str(row.get("pde", ""))
        split = str(row.get("split", ""))
        global_id = str(row.get("global_sample_id", ""))
        if pde not in expected_pdes or not split or not global_id:
            raise ValueError(
                f"sample manifest line {index} has invalid PDE/split/global ID evidence: {path}"
            )
        identity = (pde, split, global_id)
        if identity in identities:
            raise ValueError(f"sample manifest contains duplicate identity {identity}: {path}")
        identities.add(identity)
        split_counts[(pde, split)] = split_counts.get((pde, split), 0) + 1
        field_hash = str(row.get("field_sha256", "")).lower()
        content_hash = str(row.get("content_sha256", "")).lower()
        _require_sha256(field_hash, "field_sha256")
        _require_sha256(content_hash, "content_sha256")
        _reject_cross_split_sample_value(
            global_id_splits, (pde, global_id), split, "global_sample_id", path
        )
        _reject_cross_split_sample_value(
            field_hash_splits, (pde, field_hash), split, "field_sha256", path
        )
        _reject_cross_split_sample_value(
            content_hash_splits, (pde, content_hash), split, "content_sha256", path
        )
    if split_counts != dict(expected_split_counts):
        raise ValueError(
            "sample manifest per-split counts do not match the data report: "
            f"observed={split_counts}, expected={dict(expected_split_counts)}, path={path}"
        )


def _reject_cross_split_sample_value(
    seen: dict[tuple[str, str], str],
    key: tuple[str, str],
    split: str,
    field: str,
    path: Path,
) -> None:
    previous_split = seen.get(key)
    if previous_split is not None and previous_split != split:
        raise ValueError(
            f"sample manifest contains cross-split {field} overlap for PDE {key[0]!r}: "
            f"{previous_split!r} and {split!r}, path={path}"
        )
    seen[key] = split


def _require_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"data manifest {field} must be a 64-character hexadecimal SHA-256 digest")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256_bytes(encoded)


def repository_revision(root: str | Path | None = None) -> str:
    """Return HEAD plus a deterministic digest when the Git worktree is dirty.

    A bare commit SHA is insufficient for experiment identity while local code
    changes are present.  The dirty digest covers the complete tracked diff
    against HEAD (staged and unstaged) and every non-ignored untracked file's
    path, kind, and content.  Ignored runtime outputs do not affect identity.
    """
    requested_root = Path(root) if root is not None else Path(__file__).resolve().parents[2]

    def git_bytes(*args: str) -> bytes:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=requested_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = ""
            if isinstance(exc, subprocess.CalledProcessError):
                detail = exc.stderr.decode("utf-8", errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"cannot resolve repository revision from {requested_root}{suffix}") from exc
        return result.stdout

    repository_root = Path(os.fsdecode(git_bytes("rev-parse", "--show-toplevel").rstrip(b"\n")))
    head = git_bytes("rev-parse", "HEAD").decode("ascii").strip()
    tracked_diff = git_bytes("diff", "--binary", "--no-ext-diff", "HEAD", "--")
    untracked = sorted(
        path
        for path in git_bytes("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        if path
    )
    if not tracked_diff and not untracked:
        return head

    digest = hashlib.sha256()
    _update_revision_digest(digest, b"tracked-diff", tracked_diff)
    for relative_path in untracked:
        path = repository_root / os.fsdecode(relative_path)
        if path.is_symlink():
            kind = b"symlink"
            content = os.fsencode(os.readlink(path))
        elif path.is_file():
            kind = b"file"
            content = path.read_bytes()
        else:
            raise RuntimeError(f"unsupported untracked path while resolving repository revision: {path}")
        _update_revision_digest(digest, b"untracked-path", relative_path)
        _update_revision_digest(digest, b"untracked-kind", kind)
        _update_revision_digest(digest, b"untracked-content", content)
    return f"{head}-dirty-{digest.hexdigest()}"


def _update_revision_digest(digest: Any, label: bytes, content: bytes) -> None:
    """Add an unambiguous length-delimited item to a repository digest."""
    digest.update(len(label).to_bytes(8, "big"))
    digest.update(label)
    digest.update(len(content).to_bytes(8, "big"))
    digest.update(content)


def run_fingerprint(row: Mapping[str, Any]) -> str:
    fields = list(FINGERPRINT_FIELDS)
    if row.get("execution_mode") == "eval_only":
        fields.extend(EVAL_FINGERPRINT_FIELDS)
    missing = [field for field in FINGERPRINT_FIELDS if field not in row]
    if row.get("execution_mode") == "eval_only":
        missing.extend(field for field in EVAL_FINGERPRINT_FIELDS if row.get(field) in {None, ""})
    if missing:
        raise ValueError(f"cannot fingerprint matrix row; missing fields: {missing}")
    return canonical_sha256({field: row[field] for field in fields})


def requires_full_data_manifest(task_protocol_version: Any) -> bool:
    """Return whether a task protocol is a formal paper contract."""
    return str(task_protocol_version) in FORMAL_TASK_PROTOCOL_VERSIONS


def summary_validation_reasons(row: Mapping[str, Any], summary: Mapping[str, Any]) -> list[str]:
    """Return stable reason codes; an empty list means publishable/complete."""
    reasons: list[str] = []
    required_matrix_fields = (
        "matrix_schema_version",
        "run_id",
        "run_fingerprint",
        "summary_schema_version",
        "execution_mode",
        "comparison_track",
        "config_content_sha256",
        "task_protocol_version",
        "sensor_protocol_version",
        "sensor_seed",
        "commit_hash",
    )
    formal_data_contract = requires_full_data_manifest(row.get("task_protocol_version"))
    if formal_data_contract:
        required_matrix_fields = (
            *required_matrix_fields,
            "experiment_config_sha256",
            "data_manifest_sha256",
            "data_manifest_path",
        )
    for field in required_matrix_fields:
        if row.get(field) in {None, ""}:
            reasons.append(f"legacy_matrix_missing:{field}")

    try:
        expected_matrix_fingerprint = run_fingerprint(row)
    except (KeyError, TypeError, ValueError):
        # Legacy rows are retained as audit inputs, but their identity cannot
        # be trusted merely because a summary repeats the same opaque string.
        reasons.append("matrix_unverifiable:run_fingerprint")
    else:
        if row.get("run_fingerprint") != expected_matrix_fingerprint:
            reasons.append("matrix_mismatch:run_fingerprint")

    if summary.get("status") != "success":
        reasons.append("summary_not_success")

    compared_fields = (
        "matrix_schema_version",
        "summary_schema_version",
        "run_id",
        "run_fingerprint",
        "execution_mode",
        "comparison_track",
        "config_content_sha256",
        "task_protocol_version",
        "sensor_protocol_version",
        "sensor_seed",
        "commit_hash",
        *SUMMARY_IDENTITY_FIELDS,
    )
    if formal_data_contract:
        compared_fields = (
            *compared_fields,
            "experiment_config_sha256",
            "data_manifest_sha256",
            "data_manifest_path",
        )
    for field in compared_fields:
        expected = row.get(field)
        observed = summary.get(field)
        if observed in {None, ""}:
            reasons.append(f"summary_missing:{field}")
        elif expected not in {None, ""} and observed != expected:
            reasons.append(f"summary_mismatch:{field}")

    if summary.get("config_hash") in {None, ""}:
        reasons.append("summary_missing:config_hash")
    if formal_data_contract and summary.get("data_root") in {None, ""}:
        reasons.append("summary_missing:data_root")

    for matrix_field, summary_field in SUMMARY_DESIGN_FIELD_MAP.items():
        expected = row.get(matrix_field)
        observed = summary.get(summary_field)
        if expected in {None, ""}:
            reasons.append(f"legacy_matrix_missing:{matrix_field}")
        elif observed in {None, ""}:
            reasons.append(f"summary_missing:{summary_field}")
        elif observed != expected:
            reasons.append(f"summary_mismatch:{summary_field}")

    # Requested and observed split sizes are both recorded.  Formal execution
    # uses --strict-size, so a short data split can never be silently published.
    for requested_field, observed_field in (
        ("train_size", "train_size_requested"),
        ("val_size", "val_size"),
        ("test_size", "test_size"),
    ):
        expected = row.get(requested_field)
        observed = summary.get(observed_field)
        if expected not in {None, ""}:
            if observed in {None, ""}:
                reasons.append(f"summary_missing:{observed_field}")
            elif observed != expected:
                reasons.append(f"summary_mismatch:{observed_field}")

    execution_mode = summary.get("execution_mode")
    if execution_mode not in {"train", "eval_only"}:
        reasons.append("summary_invalid:execution_mode")
    expected_eval_only = execution_mode == "eval_only"
    if summary.get("eval_only") is not expected_eval_only:
        reasons.append("summary_mismatch:eval_only")

    if formal_data_contract and row.get("data_manifest_path") not in {None, ""}:
        try:
            manifest, _path, _sha = validate_full_data_manifest(
                str(row["data_manifest_path"]),
                expected_sha256=str(row.get("data_manifest_sha256", "")),
                expected_experiment_config_sha256=str(
                    row.get("experiment_config_sha256", "")
                ),
                expected_data_root=summary.get("data_root") or None,
                expected_verifier_sha256=sha256_file(
                    REPOSITORY_ROOT / "scripts" / "verify_data_protocol.py"
                ),
                verify_source_signatures=True,
            )
            if str(row.get("pde", "")) not in {
                str(value) for value in manifest.get("pdes", [])
            }:
                reasons.append("data_manifest_missing_pde")
        except (FileNotFoundError, OSError, TypeError, ValueError):
            reasons.append("data_manifest_invalid")

    if execution_mode == "eval_only":
        for field in (*EVAL_FINGERPRINT_FIELDS, "checkpoint_path"):
            if summary.get(field) in {None, ""}:
                reasons.append(f"summary_missing:{field}")
            elif row.get(field) not in {None, ""} and summary.get(field) != row.get(field):
                reasons.append(f"summary_mismatch:{field}")
    requires_training_checkpoint = bool(
        formal_data_contract
        and execution_mode == "train"
        and str(row.get("baseline", "")) in AMORTIZED_CHECKPOINT_BASELINES
    )
    if requires_training_checkpoint:
        for field in ("checkpoint_path", "checkpoint_sha256"):
            if summary.get(field) in {None, ""}:
                reasons.append(f"summary_missing:{field}")

    if execution_mode == "eval_only" or requires_training_checkpoint:
        checkpoint_path = str(summary.get("checkpoint_path", "") or "")
        expected_checkpoint_sha256 = str(summary.get("checkpoint_sha256", "") or "")
        if checkpoint_path:
            try:
                observed_checkpoint_sha256 = sha256_file(
                    checkpoint_path, root=REPOSITORY_ROOT
                )
            except (FileNotFoundError, OSError):
                reasons.append("checkpoint_missing")
            else:
                if observed_checkpoint_sha256 != expected_checkpoint_sha256:
                    reasons.append("checkpoint_hash_mismatch")

    # Keep reason order deterministic and avoid duplicated missing/mismatch
    # diagnostics when malformed input repeats the same issue.
    return list(dict.fromkeys(reasons))


def cohort_signature(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {field: summary.get(field) for field in COHORT_FIELDS}


def cohort_id(summary: Mapping[str, Any]) -> str:
    return canonical_sha256(cohort_signature(summary))[:16]


def quarantine_output_artifacts(
    row: Mapping[str, Any],
    reasons: list[str],
    *,
    exclude_names: tuple[str, ...] = ("quarantine",),
) -> Path | None:
    """Move pre-existing run artifacts aside without deleting audit evidence."""
    reject_historical_experiment_row(row)
    output_dir = Path(str(row["output_dir"]))
    if not output_dir.exists():
        return None
    candidates = [path for path in output_dir.iterdir() if path.name not in set(exclude_names)]
    if not candidates:
        return None
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    destination = output_dir / "quarantine" / f"invalid_{stamp}_{time.time_ns()}"
    destination.mkdir(parents=True, exist_ok=False)
    moved: list[str] = []
    for path in candidates:
        shutil.move(str(path), str(destination / path.name))
        moved.append(path.name)
    manifest = {
        "run_id": row.get("run_id"),
        "run_fingerprint": row.get("run_fingerprint"),
        "validation_reasons": reasons,
        "moved_artifacts": sorted(moved),
    }
    (destination / "quarantine_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return destination
