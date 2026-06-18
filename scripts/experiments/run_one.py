#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORBIDDEN_PAPER_FLAGS = {
    "--dry-run",
    "--synthetic-data",
    "--allow-synthetic-fallback",
    "--prefer-test",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run one row from an FM4PDE baseline matrix.")
    parser.add_argument("matrix", nargs="?", default=os.environ.get("MATRIX", ""))
    parser.add_argument("index", nargs="?", default=os.environ.get("TASK_INDEX", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    return parser.parse_args(argv)


def load_row(matrix: str | Path, index: int) -> dict[str, Any]:
    path = Path(matrix)
    if not path.exists():
        raise FileNotFoundError(f"matrix not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if index < 0 or index >= len(rows):
        raise IndexError(f"matrix index {index} out of range for {path} with {len(rows)} rows")
    row = rows[index]
    if row.get("skip_reason"):
        raise RuntimeError(f"matrix row {index} is marked skipped: {row['skip_reason']}")
    return row


def build_command(row: dict[str, Any]) -> list[str]:
    values = effective_command_values(row)
    data_root = os.environ.get("DATA_ROOT", "")
    if not data_root:
        raise RuntimeError("DATA_ROOT must be set to a real PDE data root for paper-mode runs")
    cmd = [
        os.environ.get("PYTHON", "python"),
        "-m",
        "baselines.run",
        "--experiment-mode",
        "paper",
        "--baseline",
        str(row["baseline"]),
        "--pde",
        str(row["pde"]),
        "--task",
        str(row["task"]),
        "--data-root",
        data_root,
        "--config",
        str(values["config"]),
        "--train-size",
        str(values["train_size"]),
        "--val-size",
        str(values["val_size"]),
        "--test-size",
        str(values["test_size"]),
        "--train-shards",
        str(values["train_shards"]),
        "--batch-size",
        str(values["batch_size"]),
        "--epochs",
        str(values["epochs"]),
        "--seed",
        str(row["seed"]),
        "--device",
        os.environ.get("DEVICE", str(row["device"])),
        "--data-loading-mode",
        str(values["data_loading_mode"]),
        "--scalar-param-mode",
        str(values["scalar_param_mode"]),
        "--output-dir",
        str(row["output_dir"]),
        "--experiment-kind",
        str(row.get("experiment_kind", "")),
        "--ablation-factor",
        str(row.get("ablation_factor", "")),
        "--task-group",
        str(row.get("task_group", "")),
        "--run-id",
        str(row["run_id"]),
        "--run-name",
        str(row.get("run_name", row["run_id"])),
    ]
    if str(row["task"]).startswith("sparse"):
        cmd.extend(
            [
                "--num-sensors",
                str(values["num_sensors"]),
                "--sensor-mode",
                str(values["sensor_mode"]),
                "--noise-level",
                str(values["noise_level"]),
            ]
        )
    if _as_bool(row.get("load_full_trajectory", False)):
        cmd.append("--load-full-trajectory")

    steps = int(values["steps"])
    refine_steps = int(values["refine_steps"])
    particles = int(values["particles"])
    if steps > 0:
        cmd.extend(["--steps", str(steps)])
    if refine_steps > 0:
        cmd.extend(["--refine-steps", str(refine_steps)])
    if particles > 0:
        cmd.extend(["--particles", str(particles)])

    forbidden = FORBIDDEN_PAPER_FLAGS.intersection(cmd)
    if forbidden:
        raise RuntimeError(f"paper command contains forbidden flags: {sorted(forbidden)}")
    return cmd


def effective_command_values(row: dict[str, Any]) -> dict[str, Any]:
    allow_override = _env_flag("ALLOW_ROW_OVERRIDE")
    values = {
        "config": row_value(row, "config", "CONFIG", allow_override),
        "train_size": row_value(row, "train_size", "TRAIN_SIZE", allow_override),
        "val_size": row_value(row, "val_size", "VAL_SIZE", allow_override),
        "test_size": row_value(row, "test_size", "TEST_SIZE", allow_override),
        "train_shards": row_value(row, "train_shards", "TRAIN_SHARDS", allow_override),
        "batch_size": row_value(row, "batch_size", "BATCH_SIZE", allow_override),
        "epochs": row_value(row, "epochs", "EPOCHS", allow_override),
        "scalar_param_mode": row_value(row, "scalar_param_mode", "SCALAR_PARAM_MODE", allow_override),
        "data_loading_mode": row_value(row, "data_loading_mode", "DATA_LOADING_MODE", allow_override),
        "num_sensors": row_value(row, "num_sensors", "NUM_SENSORS", allow_override),
        "sensor_mode": row_value(row, "sensor_mode", "SENSOR_MODE", allow_override),
        "noise_level": row_value(row, "noise_level", "NOISE_LEVEL", allow_override),
        "steps": _method_steps(row, allow_override),
        "refine_steps": _method_refine_steps(row, allow_override),
        "particles": _method_particles(row, allow_override),
        "allow_row_override": allow_override,
    }
    return values


def run_one(row: dict[str, Any], cmd: list[str]) -> int:
    output_dir = Path(row["output_dir"])
    log_dir = Path(row.get("log_dir") or output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    done = output_dir / "run.done"
    running = output_dir / "run.running"
    failed = output_dir / "run.failed"
    started = output_dir / "run.started"
    status_file = Path(row.get("status_file") or (output_dir / "run.status.json"))
    force = _env_flag("FORCE")
    retry_failed = _env_flag("RETRY_FAILED")
    lock_timeout = int(os.environ.get("LOCK_TIMEOUT_SECONDS", "86400"))

    if done.exists() and not force:
        _write_status(status_file, row, "done", "existing done marker")
        print(f"skip done: {row['run_id']}")
        return 0
    if failed.exists() and not retry_failed and not force:
        _write_status(status_file, row, "failed", "existing failed marker")
        print(f"skip failed: {row['run_id']} (set RETRY_FAILED=1 to retry)")
        return 0
    if running.exists() and not force:
        age = time.time() - running.stat().st_mtime
        if age < lock_timeout:
            _write_status(status_file, row, "running", f"running lock age={age:.1f}s")
            print(f"skip running: {row['run_id']}")
            return 0

    attempt = _previous_attempt(failed) + 1
    command_text = shlex.join(cmd)
    start_payload = {
        "run_id": row["run_id"],
        "attempt": attempt,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": command_text,
    }
    started.write_text(json.dumps(start_payload, indent=2, sort_keys=True), encoding="utf-8")
    running.write_text(json.dumps({**start_payload, "pid": os.getpid()}, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")
    (output_dir / "env.txt").write_text(_env_text(), encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "row": row,
                "command": cmd,
                "effective_command_values": effective_command_values(row),
                "attempt": attempt,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_status(status_file, row, "running", "")

    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        proc = subprocess.run(cmd, stdout=stdout, stderr=stderr)

    running.unlink(missing_ok=True)
    if proc.returncode == 0:
        failed.unlink(missing_ok=True)
        done.write_text(
            json.dumps(
                {
                    "run_id": row["run_id"],
                    "attempt": attempt,
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "command": command_text,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        _write_status(status_file, row, "done", "")
        return 0

    failed.write_text(
        json.dumps(
            {
                "run_id": row["run_id"],
                "attempt": attempt,
                "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "exit_code": proc.returncode,
                "command": command_text,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_status(status_file, row, "failed", f"exit_code={proc.returncode}")
    return int(proc.returncode)


def row_value(row: dict[str, Any], key: str, env_name: str | None = None, allow_override: bool = False) -> Any:
    if allow_override and env_name and os.environ.get(env_name) not in {None, ""}:
        return os.environ[env_name]
    return row[key]


def _method_steps(row: dict[str, Any], allow_override: bool) -> int:
    baseline = str(row["baseline"])
    if baseline == "pinn_sparse":
        return int(row_value(row, "steps", "PINN_STEPS", allow_override) or 0)
    if baseline == "pde_opt":
        return int(row_value(row, "steps", "PDEOPT_STEPS", allow_override) or 0)
    if baseline == "var4d":
        return int(row_value(row, "steps", "VAR4D_STEPS", allow_override) or 0)
    if baseline == "pc_bnn":
        return int(row_value(row, "steps", "PCBNN_STEPS", allow_override) or 0)
    return int(row.get("steps", 0) or 0)


def _method_refine_steps(row: dict[str, Any], allow_override: bool) -> int:
    if str(row["baseline"]) == "vivid":
        return int(row_value(row, "refine_steps", "VIVID_REFINE_STEPS", allow_override) or 0)
    return int(row.get("refine_steps", 0) or 0)


def _method_particles(row: dict[str, Any], allow_override: bool) -> int:
    if str(row["baseline"]) == "pc_bnn":
        return int(row_value(row, "particles", "PCBNN_PARTICLES", allow_override) or 0)
    return int(row.get("particles", 0) or 0)


def _previous_attempt(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("attempt", 0))
    except Exception:
        return 0


def _write_status(path: Path, row: dict[str, Any], status: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": row["run_id"],
                "task_group": row["task_group"],
                "pde": row["pde"],
                "baseline": row["baseline"],
                "status": status,
                "message": message,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _env_text() -> str:
    lines = [f"{key}={value}" for key, value in sorted(os.environ.items())]
    return "\n".join(lines) + "\n"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.matrix:
        raise RuntimeError("matrix path is required")
    index = int(args.index)
    if os.environ.get("ONE_BASED_INDEX", "0") in {"1", "true", "yes"}:
        index -= 1
    row = load_row(args.matrix, index)
    cmd = build_command(row)
    if _env_flag("PRINT_COMMAND_ONLY"):
        print(shlex.join(cmd))
        return 0
    return run_one(row, cmd)


if __name__ == "__main__":
    raise SystemExit(main())
