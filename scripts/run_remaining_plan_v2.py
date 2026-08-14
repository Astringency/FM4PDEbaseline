#!/usr/bin/env python
"""Run every remaining ``experiment_plan_v2`` run, in dependency order.

The remaining set is computed dynamically: a run is considered complete when
``summary.json`` exists in its ``output_dir``. This keeps the script correct
after the matrix is regenerated (e.g. the sparse_forward_main_physics time-
dependent PDEs were removed from the plan) and after partial runs.

Phases
------
1. ``sparse_forward_main_physics`` (per-instance pde_opt / pinn_sparse, 6 runs):
   launched round-robin across the configured GPUs. These are tiny per-instance
   models, so several can share a GPU.
2. Remaining amortized runs (ifno / recfno / senseiver / voronoicnn, 32 runs):
   memory-aware scheduling: at most ``MAX_PER_GPU`` trainings per GPU and only
   when the GPU's free memory exceeds ``GPU_MEM_THRESHOLD_GB``.

Usage
-----
    python scripts/run_remaining_plan_v2.py            # run everything
    python scripts/run_remaining_plan_v2.py --dry-run  # only list what would run

Env overrides: OUTPUT_ROOT, DATA_ROOT, N_GPUS, MAX_PER_GPU, GPU_MEM_THRESHOLD_GB,
POLL_SECONDS, PYTHON, CUDA_VISIBLE_DEVICES (ignored; GPUs are assigned by index).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_one import build_command  # noqa: E402

# Directory containing the ``experiment_plan_v2`` outputs (matrix + runs + logs).
# Defaults to the local ``outputs/``; set OUTPUT_ROOT to run against a different
# location (e.g. an NFS copy) without creating a symlink.
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", ROOT / "outputs"))
MATRIX = OUTPUT_ROOT / "experiment_plan_v2" / "matrices" / "experiment_plan_v2.jsonl"
DEFAULT_DATA_ROOT = "/home/zhangxf/share/zhangxfA100/large_storage/PDEdata/"

PHYSICS_GROUP = "sparse_forward_main_physics"

N_GPUS = int(os.environ.get("N_GPUS", "2"))
# Chosen from measured per-run footprints on 2x RTX 4090 (24GB) / 48-core / 256GB RAM:
#   ifno ~4-7.4GB reserved, recfno ~1.3GB, senseiver ~3.1GB, voronoicnn ~0.8GB,
#   host RAM ~10-12GB/run (eager), ~5 threads/run. 4/GPU => 8 concurrent runs
#   (~40 threads, ~96GB RAM) stays comfortably under the CPU/RAM ceilings.
MAX_PER_GPU = int(os.environ.get("MAX_PER_GPU", "4"))
GPU_MEM_THRESHOLD_GB = float(os.environ.get("GPU_MEM_THRESHOLD_GB", "9.0"))
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "20"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run remaining experiment_plan_v2 runs")
    parser.add_argument("--dry-run", action="store_true", help="List remaining runs without launching them")
    return parser.parse_args()


def is_complete(row: dict[str, Any]) -> bool:
    return (Path(row["output_dir"]) / "summary.json").exists()


def _remap_output_paths(row: dict[str, Any]) -> dict[str, Any]:
    """Rewrite matrix ``outputs/...`` paths onto the configured OUTPUT_ROOT.

    The matrix stores relative paths like ``outputs/experiment_plan_v2/runs/...``.
    When OUTPUT_ROOT points elsewhere (e.g. an NFS copy), rewrite the run paths so
    completion detection, log writing, and ``build_command`` all target that root.
    """
    for key in ("output_dir", "log_dir", "status_file"):
        value = row.get(key)
        if isinstance(value, str) and value.startswith("outputs/"):
            row[key] = str(OUTPUT_ROOT / value[len("outputs/"):])
    return row


def load_rows() -> list[dict[str, Any]]:
    rows = []
    with MATRIX.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("skip_reason"):
                rows.append(_remap_output_paths(row))
    return rows


def load_remaining() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = load_rows()
    remaining = [row for row in rows if not is_complete(row)]
    physics = [row for row in remaining if row["task_group"] == PHYSICS_GROUP]
    amortized = [row for row in remaining if row["task_group"] != PHYSICS_GROUP]
    return physics, amortized


def _env_for(row: dict[str, Any], gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env["DATA_ROOT"] = env.get("DATA_ROOT") or DEFAULT_DATA_ROOT
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHON"] = sys.executable
    return env


def _log_path(row: dict[str, Any]) -> Path:
    path = OUTPUT_ROOT / "experiment_plan_v2" / "logs" / f"train_{row['run_id']}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _clean_stale_output(row: dict[str, Any]) -> None:
    """Remove partial per-sample results from an interrupted run.

    ``baselines.run`` appends to ``results_raw.jsonl`` / ``results_summary.jsonl``.
    If a run was killed before writing ``summary.json``, relaunching it would
    append a fresh 1000-sample pass on top of the partial rows and corrupt the
    metrics. Drop those stale append-only files so the relaunch starts clean.
    """
    out = Path(row["output_dir"])
    if (out / "summary.json").exists():
        return
    for name in ("results_raw.jsonl", "results_summary.jsonl"):
        path = out / name
        if path.exists():
            path.unlink()


def _launch(row: dict[str, Any], gpu: int) -> subprocess.Popen:
    _clean_stale_output(row)
    env = _env_for(row, gpu)
    cmd = build_command(row)
    log = _log_path(row)
    lf = log.open("w")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
    return proc, log, lf


def _label(row: dict[str, Any]) -> str:
    return f"{row['task_group']}/{row['pde']}/{row['baseline']}"


def _physics_cost(row: dict[str, Any]) -> int:
    # Per-instance wall time scales with optimization steps (pinn_sparse=1000,
    # pde_opt=500). Greedy-load-balance on this proxy so both GPUs finish
    # together instead of one GPU receiving all heavy pinn_sparse runs.
    steps = int(row.get("steps", 0) or 0)
    return steps if steps > 0 else 1


def _amortized_cost(row: dict[str, Any]) -> int:
    # Approximate single-run wall-time ranking (hours) from historical logs;
    # used only to order the queue so the longest runs start first.
    return {
        "ifno": 20,
        "voronoicnn": 15,
        "senseiver": 11,
        "recfno": 10,
    }.get(row.get("baseline"), 10)


def _assign_physics(physics: list[dict[str, Any]]) -> list[tuple[dict[str, Any], int]]:
    loads = [0] * N_GPUS
    assignment: list[tuple[dict[str, Any], int]] = []
    for row in sorted(physics, key=_physics_cost, reverse=True):
        gpu = min(range(N_GPUS), key=lambda g: loads[g])
        loads[gpu] += _physics_cost(row)
        assignment.append((row, gpu))
    return assignment


def run_physics(physics: list[dict[str, Any]]) -> tuple[int, int]:
    print(f"[physics] launching {len(physics)} per-instance runs across {N_GPUS} GPU(s)", flush=True)
    jobs: list[tuple[str, subprocess.Popen, Path]] = []
    for row, gpu in _assign_physics(physics):
        proc, log, _ = _launch(row, gpu)
        jobs.append((_label(row), proc, log))
        print(f"[physics] gpu{gpu} {_label(row)} -> {log.name}", flush=True)

    ok, bad = 0, 0
    for label, proc, log in jobs:
        code = proc.wait()
        if code == 0:
            ok += 1
        else:
            bad += 1
        print(f"[physics done] exit={code} {label} (log: {log.name})", flush=True)
    return ok, bad


def gpu_free_mem_gb() -> dict[int, float]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return {i: 0.0 for i in range(N_GPUS)}
    free: dict[int, float] = {}
    for line in out.strip().splitlines():
        if "," not in line:
            continue
        idx_s, mem_s = line.split(",", 1)
        free[int(idx_s.strip())] = float(mem_s.strip()) / 1024.0
    return free


def run_amortized(amortized: list[dict[str, Any]]) -> tuple[int, int]:
    print(f"[amortized] scheduling {len(amortized)} trainings (max {MAX_PER_GPU}/GPU, "
          f"free-mem threshold {GPU_MEM_THRESHOLD_GB} GB)", flush=True)
    pending = sorted(amortized, key=_amortized_cost, reverse=True)
    running: list[tuple[subprocess.Popen, dict[str, Any], int, Path]] = []
    ok, bad = 0, 0

    while pending or running:
        still: list[tuple[subprocess.Popen, dict[str, Any], int, Path]] = []
        for proc, row, gpu, log in running:
            if proc.poll() is None:
                still.append((proc, row, gpu, log))
            else:
                if proc.returncode == 0:
                    ok += 1
                else:
                    bad += 1
                print(f"[amortized done] exit={proc.returncode} gpu{gpu} {_label(row)} (log: {log.name})", flush=True)
        running = still

        gpu_count = {i: 0 for i in range(N_GPUS)}
        for _, _, gpu, _ in running:
            gpu_count[gpu] += 1

        free_mem = gpu_free_mem_gb()
        for gpu in range(N_GPUS):
            if not pending:
                break
            if gpu_count[gpu] >= MAX_PER_GPU:
                continue
            if free_mem.get(gpu, 0.0) < GPU_MEM_THRESHOLD_GB:
                continue
            row = pending.pop(0)
            proc, log, _ = _launch(row, gpu)
            running.append((proc, row, gpu, log))
            gpu_count[gpu] += 1
            print(f"[amortized start] gpu{gpu} {_label(row)} -> {log.name}", flush=True)

        time.sleep(POLL_SECONDS)

    return ok, bad


def main() -> int:
    args = parse_args()
    os.environ.setdefault("DATA_ROOT", DEFAULT_DATA_ROOT)

    physics, amortized = load_remaining()
    print(f"[plan] remaining={len(physics) + len(amortized)} "
          f"(physics={len(physics)}, amortized={len(amortized)})", flush=True)
    for row in physics + amortized:
        print(f"  {row['task_group']:<32} {row['baseline']:<12} {row['pde']}", flush=True)

    if args.dry_run:
        print("[dry-run] no commands launched", flush=True)
        return 0

    physics_ok = physics_bad = 0
    if physics:
        physics_ok, physics_bad = run_physics(physics)

    amortized_ok = amortized_bad = 0
    if amortized:
        amortized_ok, amortized_bad = run_amortized(amortized)

    total_ok = physics_ok + amortized_ok
    total_bad = physics_bad + amortized_bad
    print("\n" + "=" * 60)
    print(f"remaining runs finished: {total_ok} succeeded, {total_bad} failed")
    print(f"  physics:   {physics_ok} ok, {physics_bad} failed")
    print(f"  amortized: {amortized_ok} ok, {amortized_bad} failed")
    return 1 if total_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
