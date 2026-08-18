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
from typing import Any, Mapping


MATRIX_SCHEMA_VERSION = 2
SUMMARY_SCHEMA_VERSION = 2
DEFAULT_TASK_PROTOCOL_VERSION = "fm4pde-task-contract-v2"
DEFAULT_SENSOR_PROTOCOL_VERSION = "fm4pde-sensor-contract-v2"

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
    "data_manifest_sha256",
    "num_sensors",
    "sensor_mode",
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
COHORT_FIELDS = (
    "summary_schema_version",
    "execution_mode",
    "comparison_track",
    "task_protocol_version",
    "sensor_protocol_version",
    "config_content_sha256",
    "data_manifest_sha256",
    "commit_hash",
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path, *, root: str | Path | None = None) -> str:
    resolved = Path(path)
    if not resolved.is_absolute() and root is not None:
        resolved = Path(root) / resolved
    if not resolved.is_file():
        raise FileNotFoundError(f"configuration file not found for provenance hashing: {resolved}")
    return sha256_bytes(resolved.read_bytes())


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
    """Return whether a task protocol is the formal v2 paper contract."""
    return str(task_protocol_version) == DEFAULT_TASK_PROTOCOL_VERSION


def summary_validation_reasons(row: Mapping[str, Any], summary: Mapping[str, Any]) -> list[str]:
    """Return stable reason codes; an empty list means publishable/complete."""
    reasons: list[str] = []
    required_matrix_fields = (
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
    if requires_full_data_manifest(row.get("task_protocol_version")):
        required_matrix_fields = (*required_matrix_fields, "data_manifest_sha256")
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
    if requires_full_data_manifest(row.get("task_protocol_version")):
        compared_fields = (*compared_fields, "data_manifest_sha256")
    for field in compared_fields:
        expected = row.get(field)
        observed = summary.get(field)
        if observed in {None, ""}:
            reasons.append(f"summary_missing:{field}")
        elif expected not in {None, ""} and observed != expected:
            reasons.append(f"summary_mismatch:{field}")

    if summary.get("config_hash") in {None, ""}:
        reasons.append("summary_missing:config_hash")
    execution_mode = summary.get("execution_mode")
    if execution_mode not in {"train", "eval_only"}:
        reasons.append("summary_invalid:execution_mode")
    if execution_mode == "eval_only":
        for field in (*EVAL_FINGERPRINT_FIELDS, "checkpoint_path"):
            if summary.get(field) in {None, ""}:
                reasons.append(f"summary_missing:{field}")
            elif row.get(field) not in {None, ""} and summary.get(field) != row.get(field):
                reasons.append(f"summary_mismatch:{field}")

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
