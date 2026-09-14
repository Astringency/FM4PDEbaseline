#!/usr/bin/env python
"""Run one resumable fixed/mixed RecFNO queue in its own tmux session."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

DESIGN = ROOT / "configs/experiments/recfno_variable_sensors.yaml"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_logged(command, directory, label):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{label}_command.json").write_text(json.dumps(command, indent=2) + "\n")
    start = time.time()
    print(f"[{label}] start {directory}", flush=True)
    with (directory / f"{label}.log").open("a") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    record = dict(exit_code=completed.returncode, start_time=start, finish_time=time.time(),
                  command=command, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    (directory / f"{label}_exit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"[{label}] exit={completed.returncode} {directory}", flush=True)
    if completed.returncode:
        raise RuntimeError(f"{label} failed; inspect {directory / (label + '.log')}")


def train_command(args, design, pde, output):
    data_files = yaml.safe_load((ROOT / design["data_files_config"]).read_text())["data_files"][pde]
    counts = design["training_regimes"][args.regime]
    command = [sys.executable, str(ROOT / "scripts/run_recfno_variable_sensors.py"),
               "--train-sensor-counts", ",".join(map(str, counts)), "--count-seed", "1",
               "--baseline", "recfno", "--pde", pde, "--task", design["task"],
               "--data-root", str(args.data_root), "--config", str(ROOT / design["config"]),
               "--experiment-mode", "paper" if not args.pilot else "debug",
               "--sensor-mode", "random_per_sample", "--sensor-budget-mode", "total",
               "--num-sensors", "500", "--condition-mode", "mixed",
               "--condition-probabilities-json", json.dumps(design["condition_probabilities"]),
               "--train-size", str(64 if args.pilot else design["train_size"]),
               "--val-size", str(16 if args.pilot else design["val_size"]),
               "--test-size", str(16 if args.pilot else design["test_size"]),
               "--train-shards", "5", "--data-files-json", json.dumps(data_files),
               "--batch-size", str(args.batch_size), "--epochs", "1" if args.pilot else "200",
               "--seed", "1", "--sensor-seed", "1", "--device", "cuda",
               "--output-dir", str(output), "--run-id", f"recfno_{pde}_{args.regime}_s1",
               "--experiment-kind", "ablation", "--ablation-factor", "training_sensor_count",
               "--task-group", "recfno_variable_sensor_count", "--comparison-track", "sparse_solution_multicondition",
               "--task-protocol-version", design["task_protocol_version"],
               "--train-only", "--strict-size", "--num-workers", "4", "--persistent-workers",
               "--pin-memory", "--prefetch-factor", "2", "--max-loaded-dataset-gb", "60",
               "--method-override", "log_interval=500"]
    if args.pilot:
        command += ["--method-override", "max_steps=4", "--method-override", "max_val_steps=1"]
    else:
        manifest = args.study_root / "data_protocol/data_protocol_report.json"
        command += ["--data-manifest-path", str(manifest), "--data-manifest-sha256", sha256(manifest),
                    "--experiment-config-sha256", sha256(DESIGN)]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--regime", choices=["fixed500", "mixed"], required=True)
    parser.add_argument("--pdes", default="poisson,helmholtz,darcy,nsnonbounded")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--queue-tag", default="", help="Separate status file for a cooperating PDE queue.")
    args = parser.parse_args()
    if not args.study_root.is_absolute():
        raise ValueError("study-root must be an explicit absolute output path")
    design = yaml.safe_load(DESIGN.read_text())
    args.study_root.mkdir(parents=True, exist_ok=True)
    if args.queue_tag and not args.queue_tag.replace("_", "").isalnum():
        raise ValueError("queue-tag must contain only letters, digits, and underscores")
    suffix = f"_{args.queue_tag}" if args.queue_tag else ""
    queue_status = args.study_root / f"queue_{args.regime}{'_pilot' if args.pilot else ''}{suffix}.json"
    status = dict(pid=os.getpid(), phase="running", args={k:str(v) for k,v in vars(args).items()},
                  start_time=time.time(), code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    queue_status.write_text(json.dumps(status, indent=2) + "\n")
    try:
        for pde in args.pdes.split(","):
            if pde not in design["pdes"]:
                raise ValueError(f"unsupported PDE {pde}")
            directory = args.study_root / ("pilot" if args.pilot else "runs") / pde / args.regime
            train_dir = directory / "train"
            summary_path = train_dir / "summary.json"
            if not args.eval_only:
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text())
                    checkpoint = Path(summary.get("checkpoint_path", ""))
                    if summary.get("status") != "success" or not checkpoint.is_file() or sha256(checkpoint) != summary.get("checkpoint_sha256"):
                        raise RuntimeError(f"invalid existing training result {summary_path}")
                    print(f"[train] reuse completed {summary_path}", flush=True)
                else:
                    if (train_dir / "train_exit.json").exists():
                        raise RuntimeError(f"previous attempt needs inspection before retry: {train_dir}")
                    run_logged(train_command(args, design, pde, train_dir), train_dir, "train")
            if not args.pilot and not args.skip_eval:
                run_logged([sys.executable, str(ROOT / "scripts/eval_recfno_sensor_counts.py"),
                            "--train-summary", str(summary_path), "--data-root", str(args.data_root),
                            "--output-dir", str(directory / "evaluation"), "--device", "cuda"], directory, "evaluation")
        status.update(phase="completed", exit_code=0)
    except BaseException as exc:
        status.update(phase="failed", exit_code=1, error=repr(exc))
        raise
    finally:
        status["finish_time"] = time.time()
        queue_status.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
