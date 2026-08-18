"""Provenance-safe row construction for sensor generalization evaluations."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from scripts.experiments.provenance import (
    DEFAULT_SENSOR_PROTOCOL_VERSION,
    DEFAULT_TASK_PROTOCOL_VERSION,
    MATRIX_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    repository_revision,
    run_fingerprint,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]


def _int_value(value: Any, default: int) -> int:
    return int(default if value is None or value == "" else value)


def _bool_value(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def checkpoint_sha256(path: str | Path) -> str:
    checkpoint = Path(path)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(template: Mapping[str, Any]) -> str:
    # A newly constructed train/eval request is bound to the config content
    # currently on disk. Reusing a recorded hash from an older source row
    # would make retraining claim the old configuration and then fail only
    # after launch.
    return sha256_file(str(template["config"]), root=ROOT)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def make_provenance_row(
    template: Mapping[str, Any],
    *,
    execution_mode: str,
    output_dir: str | Path,
    run_label: str,
    seed: int,
    sensor_seed: int,
    sensor_mode: str | None = None,
    source_train_run_id: str = "",
    source_train_run_fingerprint: str = "",
    source_train_seed: int | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    if execution_mode not in {"train", "eval_only"}:
        raise ValueError(f"unsupported execution_mode={execution_mode!r}")
    row: dict[str, Any] = {
        "matrix_schema_version": _int_value(template.get("matrix_schema_version"), MATRIX_SCHEMA_VERSION),
        "summary_schema_version": _int_value(template.get("summary_schema_version"), SUMMARY_SCHEMA_VERSION),
        "run_id": "",
        "run_fingerprint": "",
        "run_name": run_label,
        "execution_mode": execution_mode,
        "comparison_track": str(template.get("comparison_track", "unified_adapted")),
        "experiment_kind": str(template.get("experiment_kind", "main")),
        "ablation_factor": str(template.get("ablation_factor", "")),
        "task_group": str(template.get("task_group", "sparse_solution_main_amortized")),
        "task": str(template["task"]),
        "task_protocol_version": str(
            template.get("task_protocol_version", DEFAULT_TASK_PROTOCOL_VERSION)
        ),
        "sensor_protocol_version": str(
            template.get("sensor_protocol_version", DEFAULT_SENSOR_PROTOCOL_VERSION)
        ),
        "pde": str(template["pde"]),
        "baseline": str(template["baseline"]),
        "seed": int(seed),
        "sensor_seed": int(sensor_seed),
        "source_train_seed": "" if source_train_seed is None else int(source_train_seed),
        "train_size": _int_value(template.get("train_size"), 50_000),
        "val_size": _int_value(template.get("val_size"), 0),
        "test_size": _int_value(template.get("test_size"), 10_000),
        "train_shards": _int_value(template.get("train_shards"), 5),
        "batch_size": _int_value(template.get("batch_size"), 16),
        "epochs": _int_value(template.get("epochs"), 200),
        "device": str(template.get("device", "cuda")),
        "commit_hash": repository_revision(ROOT),
        "config": str(template["config"]),
        "config_content_sha256": _config_hash(template),
        "experiment_config_sha256": str(
            template.get("experiment_config_sha256", "") or ""
        ),
        "data_manifest_sha256": str(template.get("data_manifest_sha256", "") or ""),
        "data_manifest_path": str(template.get("data_manifest_path", "") or ""),
        "num_sensors": _int_value(template.get("num_sensors"), 500),
        "sensor_mode": str(sensor_mode or template.get("sensor_mode", "random_per_sample")),
        "sensor_budget_mode": str(template.get("sensor_budget_mode", "per_time")),
        "noise_level": float(template.get("noise_level", 0.0)),
        "steps": int(template.get("steps") or 0),
        "refine_steps": int(template.get("refine_steps") or 0),
        "particles": int(template.get("particles") or 0),
        "scalar_param_mode": str(template.get("scalar_param_mode", "metadata")),
        "data_loading_mode": str(template.get("data_loading_mode", "eager")),
        "num_workers": _int_value(template.get("num_workers"), 0),
        "pin_memory": _bool_value(template.get("pin_memory"), True),
        "persistent_workers": _bool_value(template.get("persistent_workers"), False),
        "prefetch_factor": _int_value(template.get("prefetch_factor"), 2),
        "load_full_trajectory": _bool_value(template.get("load_full_trajectory"), False),
        "source_train_run_id": str(source_train_run_id),
        "source_train_run_fingerprint": str(source_train_run_fingerprint),
        "checkpoint_path": str(checkpoint_path or ""),
        "checkpoint_sha256": "",
        "output_dir": str(output_dir),
        "log_dir": str(Path(output_dir) / "logs"),
        "status_file": str(Path(output_dir) / "run.status.json"),
        "skip_reason": "",
    }
    if execution_mode == "eval_only":
        if not source_train_run_id or not source_train_run_fingerprint or source_train_seed is None:
            raise ValueError(
                "eval-only row requires source_train_run_id, source_train_run_fingerprint, and source_train_seed"
            )
        if not checkpoint_path:
            raise ValueError("eval-only row requires checkpoint_path")
        row["checkpoint_sha256"] = checkpoint_sha256(checkpoint_path)
    row["run_fingerprint"] = run_fingerprint(row)
    row["run_id"] = _safe_name(f"{run_label}_{row['run_fingerprint'][:12]}")
    return row


def require_source_identity(template: Mapping[str, Any], *, fallback_run_id: str = "") -> tuple[str, str, int]:
    run_id = str(template.get("run_id", "") or fallback_run_id)
    fingerprint = str(template.get("run_fingerprint", "") or "")
    seed = int(template.get("seed", 1))
    missing = []
    if not run_id:
        missing.append("run_id")
    if not fingerprint:
        missing.append("run_fingerprint")
    if missing:
        raise ValueError(
            "source checkpoint is legacy/unverifiable (missing "
            + ", ".join(missing)
            + "); retrain it under the v2 provenance contract before eval-only use"
        )
    return run_id, fingerprint, seed


def write_eval_request(output_dir: str | Path, row: Mapping[str, Any], command: list[str]) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "eval_request.json"
    path.write_text(
        json.dumps(
            {
                "run_id": row["run_id"],
                "run_fingerprint": row["run_fingerprint"],
                "execution_mode": row["execution_mode"],
                "seed": row["seed"],
                "sensor_seed": row["sensor_seed"],
                "source_train_run_id": row["source_train_run_id"],
                "source_train_run_fingerprint": row["source_train_run_fingerprint"],
                "source_train_seed": row["source_train_seed"],
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "experiment_config_sha256": row["experiment_config_sha256"],
                "data_manifest_sha256": row["data_manifest_sha256"],
                "data_manifest_path": row["data_manifest_path"],
                "command": command,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
