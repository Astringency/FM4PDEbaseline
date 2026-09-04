#!/usr/bin/env python
"""Evaluate experiment-matrix rows on replacement test distributions.

Checkpoint-backed rows use eval-only inference. Rows without reusable state
(PINN-Sparse, PDE-Opt, PC-BNN, Var4D, and VIVID) rerun their original
per-instance/training procedure against the replacement test file. Matrix rows
may come from main results or an ablation, provided their source summaries are
complete.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_NAME = "main_results"
DEFAULT_TEST_SIZE = 1000
DEFAULT_TEST_FILE = "poisson/poisson_test_10000-128-128-2.mat"

DISTRIBUTION_TEST_FILES: dict[str, dict[str, str]] = {
    "id": {
        "poisson": "poisson/poisson_test_10000-128-128_id.mat",
        "helmholtz": "helmholtz/helmholtz_test_10000-128-128_id.mat",
        "darcy": "darcy/darcy_test_10000-128-128_id.mat",
        "nsnonbounded": "nsnonbounded/nsnonbounded_test_10000-128-128-10_id.mat",
        "burger": "burgers/burger_test_10000-128-128_id.mat",
    },
    "smooth": {
        "poisson": "poisson/poisson_test_10000-128-128_smooth.mat",
        "helmholtz": "helmholtz/helmholtz_test_10000-128-128_smooth.mat",
        "darcy": "darcy/darcy_test_10000-128-128_smooth.mat",
        "nsnonbounded": "nsnonbounded/nsnonbounded_test_10000-128-128-10_smooth.mat",
        "burger": "burgers/burger_test_10000-128-128_smooth.mat",
    },
    "rough": {
        "poisson": "poisson/poisson_test_10000-128-128_rough.mat",
        "helmholtz": "helmholtz/helmholtz_test_10000-128-128_rough.mat",
        "darcy": "darcy/darcy_test_10000-128-128_rough.mat",
        "nsnonbounded": "nsnonbounded/nsnonbounded_test_10000-128-128-10_rough.mat",
        "burger": "burgers/burger_test_10000-128-128_rough.mat",
    },
}


def log(message: str) -> None:
    print(f"[run_eval] {message}", flush=True)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw!r}")


def split_selection(value: str | Iterable[str] | None) -> list[str]:
    """Parse comma/space-separated environment or CLI selections."""
    if value is None:
        return []
    if isinstance(value, str):
        values = re.split(r"[\s,]+", value.strip()) if value.strip() else []
    else:
        values = []
        for item in value:
            values.extend(re.split(r"[\s,]+", str(item).strip()))
    output: list[str] = []
    for item in values:
        normalized = item.strip().lower()
        if normalized and normalized not in output:
            output.append(normalized)
    return output


def detect_output_root() -> Path:
    candidates = (
        Path.home() / "share/outputs/FM4PDEbaseline",
        Path.home() / "share/zhangxfA100/large_storage/outputs/FM4PDEbaseline",
    )
    return next((path.resolve() for path in candidates if path.is_dir()), candidates[0])


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def sanitize_tag(value: str) -> str:
    tag = re.sub(r"[^a-zA-Z0-9_.-]", "_", value)
    if not tag:
        raise ValueError("evaluation tag resolves to an empty value")
    return tag


def load_matrix(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"matrix not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"matrix row {line_number} is not a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"matrix has no rows: {path}")
    return rows


def stable_values(rows: Sequence[Mapping[str, Any]], field: str) -> list[str]:
    values: list[str] = []
    for row in rows:
        value = str(row.get(field, "") or "").lower()
        if value and value not in values:
            values.append(value)
    return values


def filter_rows(
    rows: Sequence[dict[str, Any]],
    *,
    pdes: Sequence[str],
    baselines: Sequence[str] = (),
    tasks: Sequence[str] = (),
    task_groups: Sequence[str] = (),
) -> list[dict[str, Any]]:
    pde_set = set(pdes)
    baseline_set = set(baselines)
    task_set = set(tasks)
    task_group_set = set(task_groups)
    return [
        row
        for row in rows
        if str(row.get("pde", "")).lower() in pde_set
        and (not baseline_set or str(row.get("baseline", "")).lower() in baseline_set)
        and (not task_set or str(row.get("task", "")).lower() in task_set)
        and (not task_group_set or str(row.get("task_group", "")).lower() in task_group_set)
        and not row.get("skip_reason")
    ]


def _translate_runs_path(path: str | Path, output_root: Path) -> Path:
    """Translate a recorded run path onto the active output mount."""
    candidate = Path(path).expanduser()
    try:
        if candidate.exists():
            return candidate.resolve()
    except OSError:
        # A path recorded on another host can have an inaccessible parent on
        # the active machine. Fall through to mount-relative translation.
        pass
    configured = str(candidate)
    marker = "/runs/"
    if marker in configured:
        return output_root / "runs" / configured.split(marker, 1)[1]
    return candidate


def source_run_dir(
    row: Mapping[str, Any], train_root: Path, output_root: Path | None = None
) -> Path:
    configured = str(row.get("output_dir", "") or "")
    if configured and output_root is not None:
        return _translate_runs_path(configured, output_root)
    marker = "/runs/main_results/"
    if marker in configured:
        return train_root / configured.split(marker, 1)[1]
    return (
        train_root
        / f"task_group={row['task_group']}"
        / f"pde={row['pde']}"
        / f"baseline={row['baseline']}"
        / f"seed={row['seed']}"
        / f"run={row['run_id']}"
    )


def translate_run_path(
    path: str | Path, train_root: Path, output_root: Path | None = None
) -> Path:
    candidate = Path(path).expanduser()
    if output_root is not None:
        return _translate_runs_path(candidate, output_root)
    configured = str(candidate)
    marker = "/runs/main_results/"
    if marker in configured:
        return train_root / configured.split(marker, 1)[1]
    if candidate.is_file():
        return candidate.resolve()
    return candidate


def load_source_summary(
    row: Mapping[str, Any], train_root: Path, output_root: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    run_dir = source_run_dir(row, train_root, output_root)
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"source summary is not complete: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "success":
        raise ValueError(f"source summary is not successful: {summary_path}")
    return run_dir, summary


def replacement_data_files(
    row: Mapping[str, Any], summary: Mapping[str, Any], test_file: str
) -> dict[str, list[str]]:
    value = row.get("data_files")
    if not isinstance(value, Mapping):
        try:
            value = json.loads(str(summary.get("data_files_json", "")))
        except json.JSONDecodeError as exc:
            raise ValueError(f"row {row.get('run_id')} has no valid data_files mapping") from exc
    if not isinstance(value, Mapping) or not value.get("train"):
        raise ValueError(f"row {row.get('run_id')} has no configured training files")
    data_files = {str(split): [str(path) for path in paths] for split, paths in value.items()}
    data_files["test"] = [test_file]
    return data_files


def as_bool(value: Any, default: bool = False) -> bool:
    if value in {None, ""}:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def first_value(*values: Any, default: Any = "") -> Any:
    for value in values:
        if value not in {None, ""}:
            return value
    return default


@dataclass(frozen=True)
class EvaluationRun:
    row: dict[str, Any]
    source_summary: dict[str, Any]
    source_run_dir: Path
    test_file: str
    output_dir: Path
    command: list[str]
    uses_checkpoint: bool


def build_evaluation_run(
    row: dict[str, Any],
    summary: dict[str, Any],
    run_dir: Path,
    *,
    test_file: str,
    test_size: int,
    eval_root: Path,
    eval_tag: str,
    data_root: Path,
    config: Path,
    python_bin: str,
    device: str,
    save_samples: bool,
    train_root: Path,
    output_root: Path | None = None,
    resume: bool = True,
) -> EvaluationRun:
    baseline = str(row["baseline"])
    pde = str(row["pde"])
    task = str(row["task"])
    task_group = str(row["task_group"])
    seed = int(row.get("seed", 1))
    source_row_id = str(row["run_id"])
    eval_run_id = f"eval_{eval_tag}_{source_row_id}"
    output_dir = (
        eval_root
        / f"task_group={task_group}"
        / f"pde={pde}"
        / f"baseline={baseline}"
        / f"seed={seed}"
        / f"run={eval_run_id}"
    )
    data_files = replacement_data_files(row, summary, test_file)

    checkpoint_value = str(
        summary.get("checkpoint_path", "") or row.get("checkpoint_path", "") or ""
    )
    checkpoint = (
        translate_run_path(checkpoint_value, train_root, output_root)
        if checkpoint_value
        else None
    )
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint is missing: {checkpoint}")
    uses_checkpoint = checkpoint is not None
    execution_mode = "eval_only" if uses_checkpoint else "train"

    train_size = int(first_value(row.get("train_size"), summary.get("train_requested_size"), default=50_000))
    val_size = int(first_value(row.get("val_size"), summary.get("val_requested_size"), default=5_000))
    train_shards = int(first_value(row.get("train_shards"), summary.get("train_shards"), default=5))
    batch_size = int(first_value(row.get("batch_size"), summary.get("batch_size"), default=16))
    epochs = int(first_value(row.get("epochs"), summary.get("epochs"), default=1))
    sensor_seed = int(first_value(row.get("sensor_seed"), summary.get("sensor_seed"), default=seed))
    data_loading_mode = str(
        first_value(row.get("data_loading_mode"), summary.get("data_loading_mode_requested"), default="eager")
    )
    num_workers = int(first_value(row.get("num_workers"), summary.get("dataloader_num_workers"), default=4))
    prefetch_factor = int(
        first_value(row.get("prefetch_factor"), summary.get("prefetch_factor_requested"), default=2)
    )
    scalar_param_mode = str(
        first_value(row.get("scalar_param_mode"), summary.get("scalar_param_mode_requested"), default="metadata")
    )
    physics_metric_mode = str(
        first_value(row.get("physics_metric_mode"), summary.get("physics_metric_mode"), default="per_sample")
    )
    comparison_track = str(
        first_value(row.get("comparison_track"), summary.get("comparison_track"), default="unified_adapted")
    )
    task_protocol_version = str(
        first_value(
            row.get("task_protocol_version"),
            summary.get("task_protocol_version"),
            default="fm4pde-task-contract-v3",
        )
    )
    sensor_protocol_version = str(
        first_value(
            row.get("sensor_protocol_version"),
            summary.get("sensor_protocol_version"),
            default="fm4pde-sensor-contract-v3",
        )
    )

    command = [
        python_bin,
        "-m",
        "baselines.run",
        "--baseline",
        baseline,
        "--pde",
        pde,
        "--task",
        task,
        "--task-group",
        task_group,
        "--config",
        str(config),
        "--experiment-mode",
        "debug",
        "--execution-mode",
        execution_mode,
        "--comparison-track",
        comparison_track,
        "--task-protocol-version",
        task_protocol_version,
        "--sensor-protocol-version",
        sensor_protocol_version,
        "--data-root",
        str(data_root),
        "--data-files-json",
        json.dumps(data_files, sort_keys=True, separators=(",", ":")),
        "--train-size",
        str(train_size),
        "--val-size",
        str(val_size),
        "--test-size",
        str(test_size),
        "--train-shards",
        str(train_shards),
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--sensor-seed",
        str(sensor_seed),
        "--device",
        device,
        "--data-loading-mode",
        data_loading_mode,
        "--num-workers",
        str(num_workers),
        "--prefetch-factor",
        str(prefetch_factor),
        "--scalar-param-mode",
        scalar_param_mode,
        "--physics-metric-mode",
        physics_metric_mode,
        "--strict-size",
        "--output-dir",
        str(output_dir),
        "--run-id",
        eval_run_id,
        "--run-name",
        f"{eval_tag}/{task_group}/{baseline}/{pde}/seed={seed}",
        "--experiment-kind",
        "evaluation",
        "--ablation-factor",
        "replacement_test",
        "--no-save-checkpoint",
        "--resume-eval" if resume else "--no-resume-eval",
    ]

    if task.startswith("sparse"):
        command.extend(
            [
                "--num-sensors",
                str(first_value(row.get("num_sensors"), summary.get("num_sensors"), default=500)),
                "--sensor-mode",
                str(first_value(row.get("sensor_mode"), summary.get("sensor_mode"), default="random_per_sample")),
                "--sensor-budget-mode",
                str(
                    first_value(
                        row.get("sensor_budget_mode"),
                        summary.get("sensor_budget_mode_requested"),
                        default="per_time",
                    )
                ),
                "--noise-level",
                str(first_value(row.get("noise_level"), summary.get("noise_level"), default=0.0)),
            ]
        )

    if task == "sparse_solution_multicondition":
        condition_mode = str(
            first_value(
                row.get("condition_mode"),
                summary.get("evaluation_condition_mode"),
                summary.get("condition_mode"),
                default="mixed",
            )
        )
        probabilities: Any = row.get("condition_probabilities")
        if not isinstance(probabilities, Mapping):
            probabilities = summary.get("condition_probabilities")
        if isinstance(probabilities, str):
            try:
                probabilities = json.loads(probabilities)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"row {source_row_id} has invalid condition_probabilities JSON"
                ) from exc
        if not isinstance(probabilities, Mapping):
            probabilities = {
                "a_only": 0.3333333333,
                "u_only": 0.3333333333,
                "both": 0.3333333334,
            }
        command.extend(
            [
                "--condition-mode",
                condition_mode,
                "--condition-probabilities-json",
                json.dumps(dict(probabilities), sort_keys=True, separators=(",", ":")),
            ]
        )

    if as_bool(first_value(row.get("load_full_trajectory"), summary.get("load_full_trajectory"), default=False)):
        command.append("--load-full-trajectory")
    command.append(
        "--pin-memory"
        if as_bool(first_value(row.get("pin_memory"), summary.get("pin_memory_requested"), default=True))
        else "--no-pin-memory"
    )
    command.append(
        "--persistent-workers"
        if as_bool(
            first_value(row.get("persistent_workers"), summary.get("persistent_workers_requested"), default=True)
        )
        else "--no-persistent-workers"
    )
    command.append("--save-sample-artifacts" if save_samples else "--no-save-sample-artifacts")

    for flag, field in (("--steps", "steps"), ("--refine-steps", "refine_steps"), ("--particles", "particles")):
        value = int(first_value(row.get(field), summary.get(field), default=0))
        if value > 0:
            command.extend([flag, str(value)])

    if uses_checkpoint:
        if str(row.get("execution_mode", "train")) == "eval_only":
            source_train_run_id = str(
                first_value(row.get("source_train_run_id"), summary.get("source_train_run_id"))
            )
            source_train_fingerprint = str(
                first_value(
                    row.get("source_train_run_fingerprint"),
                    summary.get("source_train_run_fingerprint"),
                )
            )
            source_train_seed = int(
                first_value(row.get("source_train_seed"), summary.get("source_train_seed"), default=seed)
            )
            source_train_task = str(
                first_value(row.get("source_train_task"), summary.get("source_train_task"), default=task)
            )
        else:
            source_train_run_id = source_row_id
            source_train_fingerprint = str(
                row.get("run_fingerprint", "") or summary.get("run_fingerprint", "")
            )
            source_train_seed = seed
            source_train_task = task
        if not source_train_run_id or not source_train_fingerprint:
            raise ValueError(f"checkpoint row {source_row_id} has incomplete source-training provenance")
        command.extend(
            [
                "--eval-only",
                "--checkpoint",
                str(checkpoint),
                "--source-train-run-id",
                source_train_run_id,
                "--source-train-run-fingerprint",
                source_train_fingerprint,
                "--source-train-seed",
                str(source_train_seed),
                "--source-train-task",
                source_train_task,
            ]
        )
        checkpoint_sha256 = str(
            first_value(summary.get("checkpoint_sha256"), row.get("checkpoint_sha256"))
        )
        if checkpoint_sha256:
            command.extend(["--checkpoint-sha256", checkpoint_sha256])
        source_code_sha256 = str(
            first_value(
                row.get("source_train_baseline_code_sha256"),
                summary.get("source_train_baseline_code_sha256"),
                row.get("baseline_code_sha256"),
                summary.get("baseline_code_sha256"),
            )
        )
        if source_code_sha256:
            command.extend(["--source-train-baseline-code-sha256", source_code_sha256])

    return EvaluationRun(
        row=row,
        source_summary=summary,
        source_run_dir=run_dir,
        test_file=test_file,
        output_dir=output_dir,
        command=command,
        uses_checkpoint=uses_checkpoint,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate selected rows from an experiment matrix on replacement test data. "
            "BASELINE_LIST (or legacy BASELINES) accepts comma/space-separated method names."
        )
    )
    parser.add_argument("--test-file", default=os.environ.get("TEST_FILE", ""))
    parser.add_argument(
        "--distribution",
        choices=sorted(DISTRIBUTION_TEST_FILES),
        default=os.environ.get("DISTRIBUTION", "") or None,
    )
    parser.add_argument("--pde", action="append", dest="pde_values", default=[])
    parser.add_argument("--pdes", default=os.environ.get("PDE_LIST", os.environ.get("PDE", "")))
    parser.add_argument(
        "--baselines",
        "--baseline-list",
        dest="baselines",
        default=os.environ.get("BASELINE_LIST", os.environ.get("BASELINES", "")),
        help="Comma/space-separated baseline filter; empty selects all methods.",
    )
    parser.add_argument("--tasks", default=os.environ.get("TASKS", ""), help="Optional task filter.")
    parser.add_argument(
        "--task-groups",
        default=os.environ.get("TASK_GROUPS", ""),
        help="Optional task-group filter.",
    )
    parser.add_argument(
        "--test-size", type=int, default=int(os.environ.get("TEST_SIZE", DEFAULT_TEST_SIZE))
    )
    parser.add_argument(
        "--data-root", default=os.environ.get("DATA_ROOT", str(Path.home() / "share/PDEdata"))
    )
    parser.add_argument("--output-root", default=os.environ.get("FM_OUTPUT_ROOT", ""))
    parser.add_argument("--train-root", default=os.environ.get("TRAIN_ROOT", ""))
    parser.add_argument("--matrix", default=os.environ.get("MATRIX", ""))
    parser.add_argument("--eval-root", default=os.environ.get("EVAL_ROOT", ""))
    parser.add_argument("--eval-tag", default=os.environ.get("EVAL_TAG", ""))
    parser.add_argument("--config", default=os.environ.get("CONFIG", "baselines/configs/paper.yaml"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--python", dest="python_bin", default=os.environ.get("PYTHON", sys.executable))
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument("--save-samples", dest="save_samples", action="store_true")
    sample_group.add_argument("--no-save-samples", dest="save_samples", action="store_false")
    parser.set_defaults(save_samples=env_bool("SAVE_SAMPLES", True))
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=env_bool("RESUME", True))
    parser.add_argument("--dry-run", action="store_true", default=env_bool("DRY_RUN", False))
    return parser.parse_args(argv)


def command_value(command: Sequence[str], flag: str) -> str:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"evaluation command is missing {flag}") from exc


def successful_summary(path: Path, evaluation: EvaluationRun) -> bool:
    if not path.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(summary, dict) or summary.get("status") != "success":
        return False
    command = evaluation.command
    expected = {
        "baseline": command_value(command, "--baseline"),
        "pde": command_value(command, "--pde"),
        "task": command_value(command, "--task"),
        "run_id": command_value(command, "--run-id"),
        "seed": int(command_value(command, "--seed")),
        "test_size": int(command_value(command, "--test-size")),
        "test_requested_size": int(command_value(command, "--test-size")),
        "batch_size": int(command_value(command, "--batch-size")),
        "metric_granularity": command_value(command, "--physics-metric-mode"),
        "execution_mode": command_value(command, "--execution-mode"),
        "eval_only": "--eval-only" in command,
    }
    if "--condition-mode" in command:
        expected["condition_mode"] = command_value(command, "--condition-mode")
    if any(summary.get(field) != value for field, value in expected.items()):
        return False
    try:
        stored_data_files = json.loads(str(summary.get("data_files_json", "")))
        requested_data_files = json.loads(command_value(command, "--data-files-json"))
    except json.JSONDecodeError:
        return False
    if stored_data_files != requested_data_files:
        return False
    if "--save-sample-artifacts" in command:
        manifest_path = Path(str(summary.get("sample_manifest_path", "")))
        if int(summary.get("sample_artifact_count", 0) or 0) != expected["test_size"]:
            return False
        if not manifest_path.is_file():
            return False
    return True


def validate_selection(requested: Sequence[str], available: Sequence[str], label: str) -> None:
    unknown = sorted(set(requested).difference(available))
    if unknown:
        raise ValueError(f"{label} contains values absent from the matrix: {unknown}")


def resolve_test_files(
    *,
    pdes: Sequence[str],
    data_root: Path,
    test_file: str,
    distribution: str | None,
) -> dict[str, str]:
    if test_file and distribution:
        raise ValueError("use either --test-file or --distribution, not both")
    if test_file:
        if len(pdes) != 1:
            raise ValueError("--test-file requires exactly one selected PDE")
        candidate = Path(test_file).expanduser()
        if candidate.is_absolute():
            resolved = candidate.resolve()
            try:
                relative = resolved.relative_to(data_root)
            except ValueError as exc:
                raise ValueError(f"absolute test file must be below DATA_ROOT: {resolved}") from exc
        else:
            relative = candidate
            resolved = (data_root / relative).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"test file not found: {resolved}")
        return {pdes[0]: relative.as_posix()}
    if distribution:
        mapping = DISTRIBUTION_TEST_FILES[distribution]
        output: dict[str, str] = {}
        for pde in pdes:
            if pde not in mapping:
                raise ValueError(f"no {distribution!r} test file is configured for PDE {pde!r}")
            relative = mapping[pde]
            path = data_root / relative
            if not path.is_file():
                raise FileNotFoundError(f"test file not found: {path}")
            output[pde] = relative
        return output
    if pdes != ["poisson"]:
        raise ValueError("select --distribution, or pass --test-file for a single PDE")
    fallback = data_root / DEFAULT_TEST_FILE
    if not fallback.is_file():
        raise FileNotFoundError(f"default test file not found: {fallback}")
    return {"poisson": DEFAULT_TEST_FILE}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.test_size <= 0:
        raise ValueError(f"test size must be positive, got {args.test_size}")

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")
    output_root = (
        Path(args.output_root).expanduser().resolve() if args.output_root else detect_output_root()
    )
    train_root = (
        Path(args.train_root).expanduser().resolve()
        if args.train_root
        else output_root / "runs/main_results"
    )
    matrix_path = (
        Path(args.matrix).expanduser().resolve()
        if args.matrix
        else output_root / f"matrices/{DEFAULT_MATRIX_NAME}.jsonl"
    )
    config = resolve_repo_path(args.config)
    if not config.is_file():
        raise FileNotFoundError(f"baseline config not found: {config}")
    if not train_root.is_dir():
        raise FileNotFoundError(f"training run root not found: {train_root}")
    rows = load_matrix(matrix_path)

    available_pdes = stable_values(rows, "pde")
    pdes = split_selection([*args.pde_values, args.pdes])
    if not pdes:
        pdes = available_pdes if args.distribution else ["poisson"]
    if "all" in pdes:
        pdes = available_pdes
    baselines = split_selection(args.baselines)
    tasks = split_selection(args.tasks)
    task_groups = split_selection(args.task_groups)
    validate_selection(pdes, available_pdes, "PDE_LIST")
    validate_selection(baselines, stable_values(rows, "baseline"), "BASELINE_LIST")
    validate_selection(tasks, stable_values(rows, "task"), "TASKS")
    validate_selection(task_groups, stable_values(rows, "task_group"), "TASK_GROUPS")

    selected = filter_rows(
        rows,
        pdes=pdes,
        baselines=baselines,
        tasks=tasks,
        task_groups=task_groups,
    )
    if not selected:
        raise ValueError("the requested PDE/baseline/task filters select no matrix rows")
    selected_pdes = stable_values(selected, "pde")
    test_files = resolve_test_files(
        pdes=selected_pdes,
        data_root=data_root,
        test_file=args.test_file,
        distribution=args.distribution,
    )
    if args.eval_tag:
        eval_tag = sanitize_tag(args.eval_tag)
    elif args.distribution:
        eval_tag = sanitize_tag(args.distribution)
    else:
        eval_tag = sanitize_tag(Path(next(iter(test_files.values()))).stem)
    eval_root = (
        Path(args.eval_root).expanduser().resolve()
        if args.eval_root
        else output_root / "runs/evaluations" / eval_tag
    )

    log(f"MATRIX={matrix_path} selected_rows={len(selected)}/{len(rows)}")
    log(f"PDE_LIST={','.join(pdes)} BASELINE_LIST={','.join(baselines) if baselines else 'all'}")
    log(
        f"TASKS={','.join(tasks) if tasks else 'all'} "
        f"TASK_GROUPS={','.join(task_groups) if task_groups else 'all'}"
    )
    log(f"TEST_SIZE={args.test_size} TEST_FILES={json.dumps(test_files, sort_keys=True)}")
    log(
        f"EVAL_ROOT={eval_root} DEVICE={args.device} "
        f"SAVE_SAMPLES={int(args.save_samples)} RESUME={int(args.resume)} "
        f"DRY_RUN={int(args.dry_run)}"
    )

    lock_handle = None
    if not args.dry_run:
        eval_root.mkdir(parents=True, exist_ok=True)
        lock_handle = (eval_root / ".run_eval.lock").open("w", encoding="utf-8")
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another evaluation is already active for {eval_root}") from exc

    launched = 0
    completed = 0
    pending = 0
    checkpoint_runs = 0
    rerun_procedure_runs = 0
    for index, row in enumerate(selected, start=1):
        identity = f"{row['task_group']}/{row['baseline']}/{row['pde']}/seed={row['seed']}"
        try:
            run_dir, summary = load_source_summary(row, train_root, output_root)
            evaluation = build_evaluation_run(
                row,
                summary,
                run_dir,
                test_file=test_files[str(row["pde"])],
                test_size=args.test_size,
                eval_root=eval_root,
                eval_tag=eval_tag,
                data_root=data_root,
                config=config,
                python_bin=args.python_bin,
                device=args.device,
                save_samples=args.save_samples,
                train_root=train_root,
                output_root=output_root,
                resume=args.resume,
            )
        except (FileNotFoundError, ValueError) as exc:
            log(f"PENDING [{index}/{len(selected)}] {identity}: {exc}")
            pending += 1
            continue
        if args.resume and successful_summary(evaluation.output_dir / "summary.json", evaluation):
            log(f"SKIP completed [{index}/{len(selected)}] {identity}")
            completed += 1
            continue
        mode = "checkpoint" if evaluation.uses_checkpoint else "rerun-procedure"
        checkpoint_runs += int(evaluation.uses_checkpoint)
        rerun_procedure_runs += int(not evaluation.uses_checkpoint)
        log(f"RUN [{index}/{len(selected)}] mode={mode} {identity}")
        if args.dry_run:
            print(f"[dry-run] {shlex.join(evaluation.command)}", flush=True)
        else:
            subprocess.run(evaluation.command, cwd=ROOT, check=True)
        launched += 1

    if lock_handle is not None:
        lock_handle.close()
    log(
        "finished: "
        f"selected={len(selected)} launched={launched} already_complete={completed} pending={pending} "
        f"checkpoint_runs={checkpoint_runs} rerun_procedure_runs={rerun_procedure_runs}"
    )
    if launched == 0 and completed == 0:
        raise RuntimeError("no selected evaluation row is currently runnable")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        log(f"ERROR: {exc}")
        raise SystemExit(2) from exc
