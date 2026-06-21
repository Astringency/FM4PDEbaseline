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


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def progress_enabled() -> bool:
    return os.environ.get("RUN_PROGRESS", "1").lower() not in {"0", "false", "no", "off"}


def progress(message: str) -> None:
    if progress_enabled():
        print(f"[{timestamp()}] {message}", file=sys.stderr, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run one row from an FM4PDE baseline matrix.")
    parser.add_argument("matrix", nargs="?", default=os.environ.get("MATRIX", ""))
    parser.add_argument("index", nargs="?", default=os.environ.get("TASK_INDEX", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    return parser.parse_args(argv)


def load_matrix_rows(matrix: str | Path) -> list[dict[str, Any]]:
    path = Path(matrix)
    if not path.exists():
        raise FileNotFoundError(f"matrix not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_row(matrix: str | Path, index: int) -> dict[str, Any]:
    rows = load_matrix_rows(matrix)
    if index < 0 or index >= len(rows):
        raise IndexError(f"matrix index {index} out of range for {matrix} with {len(rows)} rows")
    row = rows[index]
    if row.get("skip_reason"):
        raise RuntimeError(f"matrix row {index} is marked skipped: {row['skip_reason']}")
    return row


def load_row_with_total(matrix: str | Path, index: int) -> tuple[dict[str, Any], int]:
    rows = load_matrix_rows(matrix)
    if index < 0 or index >= len(rows):
        raise IndexError(f"matrix index {index} out of range for {matrix} with {len(rows)} rows")
    return rows[index], len(rows)


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


def run_one(
    row: dict[str, Any],
    cmd: list[str],
    *,
    index: int | None = None,
    total: int | None = None,
    matrix: str | Path | None = None,
) -> int:
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
    index_text = _index_text(index, total)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    command_path = output_dir / "command.txt"

    if done.exists() and not force:
        _write_status(status_file, row, "done", "existing done marker")
        progress(f"[run skip] index={index_text} run_id={row['run_id']} reason=existing done marker output_dir={output_dir}")
        return 0
    if failed.exists() and not retry_failed and not force:
        _write_status(status_file, row, "failed", "existing failed marker")
        progress(
            f"[run skip] index={index_text} run_id={row['run_id']} reason=existing failed marker "
            f"output_dir={output_dir} stderr={stderr_path} retry_hint=RETRY_FAILED=1"
        )
        return 0
    if running.exists() and not force:
        age = time.time() - running.stat().st_mtime
        if age < lock_timeout:
            _write_status(status_file, row, "running", f"running lock age={age:.1f}s")
            progress(
                f"[run skip] index={index_text} run_id={row['run_id']} reason=running lock still fresh "
                f"lock_age={age:.1f}s lock_timeout={lock_timeout}s output_dir={output_dir}"
            )
            return 0
        progress(
            f"[run stale-lock] index={index_text} run_id={row['run_id']} lock_age={age:.1f}s "
            f"lock_timeout={lock_timeout}s output_dir={output_dir}"
        )

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
    command_path.write_text(command_text + "\n", encoding="utf-8")
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

    values = effective_command_values(row)
    progress(
        f"[run start] index={index_text} run_id={row['run_id']} task_group={row.get('task_group', '')} "
        f"baseline={row.get('baseline', '')} pde={row.get('pde', '')} task={row.get('task', '')} seed={row.get('seed', '')}"
    )
    if matrix is not None:
        progress(f"[run matrix] matrix={matrix} raw_index={index if index is not None else ''} total={total if total is not None else ''}")
    progress(f"[run paths] output_dir={output_dir} stdout={stdout_path} stderr={stderr_path} status={status_file}")
    progress(
        f"[run config] train_size={values['train_size']} val_size={values['val_size']} test_size={values['test_size']} "
        f"sensors={values['num_sensors']} sensor_mode={values['sensor_mode']} noise={values['noise_level']} "
        f"device={os.environ.get('DEVICE', str(row['device']))}"
    )
    _progress_command(command_text, command_path)

    started_monotonic = time.monotonic()
    interval = _progress_interval_seconds()
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
        running.write_text(
            json.dumps({**start_payload, "pid": os.getpid(), "child_pid": proc.pid}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        next_heartbeat = started_monotonic + interval
        while True:
            returncode = proc.poll()
            if returncode is not None:
                break
            now = time.monotonic()
            if interval > 0 and now >= next_heartbeat:
                progress(_heartbeat_message(row, index_text, now - started_monotonic, stdout_path, stderr_path))
                next_heartbeat = now + interval
            time.sleep(_poll_sleep_seconds(interval))

    running.unlink(missing_ok=True)
    elapsed = time.monotonic() - started_monotonic
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
        progress(f"[run done] run_id={row['run_id']} elapsed={_format_elapsed(elapsed)} exit_code=0 output_dir={output_dir}")
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
    progress(
        f"[run failed] run_id={row['run_id']} elapsed={_format_elapsed(elapsed)} exit_code={proc.returncode} "
        f"stderr={stderr_path} stderr_tail={_tail_for_progress(stderr_path, lines=5)}"
    )
    progress(
        f"[run failed paths] output_dir={output_dir} stdout={stdout_path} stderr={stderr_path} "
        f"command={command_path} status={status_file}"
    )
    return int(proc.returncode)


def _index_text(index: int | None, total: int | None) -> str:
    if index is None:
        return f"?/{total}" if total is not None else "?/?"
    human_index = index + 1
    return f"{human_index}/{total}" if total is not None else str(human_index)


def _progress_command(command_text: str, command_path: Path) -> None:
    if _env_flag("RUN_PRINT_COMMAND"):
        progress(f"[run command] {command_text}")
    else:
        progress(f"[run command] {_compact_text(command_text, max_chars=180)} command_txt={command_path} set_RUN_PRINT_COMMAND=1_for_full_command")


def _progress_interval_seconds() -> float:
    raw = os.environ.get("PROGRESS_INTERVAL_SECONDS", "60")
    try:
        interval = float(raw)
    except ValueError:
        interval = 60.0
    return max(interval, 0.0)


def _poll_sleep_seconds(interval: float) -> float:
    if interval <= 0:
        return 0.2
    return min(1.0, max(0.01, interval / 10.0))


def _heartbeat_message(row: dict[str, Any], index_text: str, elapsed: float, stdout_path: Path, stderr_path: Path) -> str:
    message = (
        f"[run heartbeat] index={index_text} run_id={row['run_id']} elapsed={_format_elapsed(elapsed)} "
        f"status=running stdout={stdout_path} stderr={stderr_path}"
    )
    if _env_flag("RUN_TAIL_LOGS"):
        message += f" stdout_tail={_tail_for_progress(stdout_path, lines=2)} stderr_tail={_tail_for_progress(stderr_path, lines=2)}"
    return message


def _format_elapsed(seconds: float) -> str:
    return f"{int(seconds)}s"


def _compact_text(text: str, max_chars: int = 600) -> str:
    text = text.replace("\n", " | ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _tail_for_progress(path: Path, *, lines: int = 2, max_chars: int = 600) -> str:
    if not path.exists():
        return "<missing>"
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192))
            text = f.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return f"<unreadable:{exc}>"
    selected = "\n".join(text.splitlines()[-lines:]).strip()
    if not selected:
        return "<empty>"
    selected = selected.replace("\n", " | ")
    return shlex.quote(_compact_text(selected, max_chars=max_chars))


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
    row, total = load_row_with_total(args.matrix, index)
    if row.get("skip_reason"):
        status_raw = str(row.get("status_file") or "")
        if status_raw:
            _write_status(Path(status_raw), row, "skipped", str(row["skip_reason"]))
        progress(
            f"[run skip] index={_index_text(index, total)} run_id={row.get('run_id', '')} "
            f"reason=row has skip_reason skip_reason={row['skip_reason']} output_dir={row.get('output_dir', '')}"
        )
        return 0
    cmd = build_command(row)
    if _env_flag("PRINT_COMMAND_ONLY"):
        progress(
            f"[print command only] index={_index_text(index, total)} run_id={row['run_id']} "
            f"matrix={args.matrix} output_dir={row.get('output_dir', '')}"
        )
        print(shlex.join(cmd), flush=True)
        return 0
    return run_one(row, cmd, index=index, total=total, matrix=args.matrix)


if __name__ == "__main__":
    raise SystemExit(main())
