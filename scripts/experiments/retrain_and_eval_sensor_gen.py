#!/usr/bin/env python
"""Retrain one model with checkpoint saving, then measure sensor-location
generalization (test at train sensor locations vs unseen sensor locations).

Default experiment: recfno / sparse_forward / poisson (fastest model that still
learns the task well, ~5.6h train, relative L2 ~1.3%).

Steps:
  1. Retrain with SAVE_CHECKPOINT=amortized into a fresh output dir (writes .pt).
  2. For test sensor seed in {1 (train locations), 2, 3 (unseen locations)}:
     run ``baselines.run --eval-only`` on the trained checkpoint.
  3. Print a comparison table.

Usage:
    python scripts/experiments/retrain_and_eval_sensor_gen.py --dry-run
    python scripts/experiments/retrain_and_eval_sensor_gen.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_one import build_command  # noqa: E402

OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/home/zhangxf/share/zhangxfA100/large_storage/outputs/FM4PDEbaseline"))
DATA_ROOT = "/home/zhangxf/share/zhangxfA100/large_storage/PDEdata/"

PDE = "poisson"
TASK = "sparse_forward"
TASK_GROUP = "sparse_forward_main_amortized"
TEST_SEEDS = [1, 2, 3]


def find_source_run_id(baseline: str) -> str:
    run_dir = (
        OUTPUT_ROOT / "experiment_plan_v2" / "runs" / "experiment_plan_v2"
        / f"task_group={TASK_GROUP}" / f"pde={PDE}" / f"baseline={baseline}" / "seed=1"
    )
    configs = list(run_dir.glob("run=*/*_config.json"))
    if not configs:
        raise FileNotFoundError(f"source config not found under {run_dir}")
    return configs[0].parent.name.removeprefix("run=")


def load_source_args(baseline: str) -> dict:
    run_dir = (
        OUTPUT_ROOT / "experiment_plan_v2" / "runs" / "experiment_plan_v2"
        / f"task_group={TASK_GROUP}" / f"pde={PDE}" / f"baseline={baseline}" / "seed=1"
    )
    configs = list(run_dir.glob(f"run={find_source_run_id(baseline)}/*_config.json"))
    if not configs:
        raise FileNotFoundError(f"source config not found under {run_dir}")
    return json.loads(configs[0].read_text())["args"]


def make_row(old: dict, *, seed: int, output_dir: Path, run_id: str, run_name: str) -> dict:
    return {
        "baseline": old["baseline"],
        "pde": old["pde"],
        "task": old["task"],
        "seed": seed,
        "device": old.get("device", "cuda"),
        "output_dir": str(output_dir),
        "run_id": run_id,
        "run_name": run_name,
        "task_group": old.get("task_group", TASK_GROUP),
        "experiment_kind": old.get("experiment_kind", "main"),
        "ablation_factor": old.get("ablation_factor", ""),
        "config": old["config"],
        "train_size": old["train_size"],
        "val_size": old["val_size"],
        "test_size": old["test_size"],
        "train_shards": old.get("train_shards", 5),
        "batch_size": old.get("batch_size", 16),
        "epochs": old.get("epochs", 200),
        "data_loading_mode": old.get("data_loading_mode", "eager"),
        "num_workers": old.get("num_workers", 0),
        "prefetch_factor": old.get("prefetch_factor", 2),
        "scalar_param_mode": old.get("scalar_param_mode", "metadata"),
        "num_sensors": old.get("num_sensors", 500),
        "sensor_mode": old.get("sensor_mode", "random"),
        "noise_level": old.get("noise_level", 0.0),
        "load_full_trajectory": old.get("load_full_trajectory", False),
        "pin_memory": old.get("pin_memory", True),
        "persistent_workers": old.get("persistent_workers", False),
        "steps": old.get("steps") or 0,
        "refine_steps": old.get("refine_steps") or 0,
        "particles": old.get("particles") or 0,
    }


def run_command(cmd: list[str], out_dir: Path, log_name: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / log_name
    print(f"[run] {' '.join(cmd)}", flush=True)
    with log_path.open("w") as lf:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
        return proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="recfno", choices=["recfno", "senseiver", "voronoicnn"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-train", action="store_true", help="Skip training; only run eval using existing checkpoint")
    args = ap.parse_args()

    baseline = args.baseline
    retrain_run_id = f"retrain_{baseline}_{TASK}_{PDE}_s1"

    os.environ["DATA_ROOT"] = DATA_ROOT
    os.environ["PYTHON"] = sys.executable

    old = load_source_args(baseline)
    base = OUTPUT_ROOT / "sensor_generalization_retrain" / f"{baseline}_{TASK}_{PDE}"
    train_dir = base / "train"
    ckpt = train_dir / f"{retrain_run_id}.pt"

    # 1) Retrain (writes checkpoint).
    if not args.skip_train:
        os.environ["SAVE_CHECKPOINT"] = "amortized"
        train_row = make_row(
            old, seed=1, output_dir=train_dir,
            run_id=retrain_run_id, run_name=f"{baseline}/{TASK}/{PDE} retrain seed=1",
        )
        train_cmd = build_command(train_row)
        if args.dry_run:
            print(f"[dry-run] train: {' '.join(train_cmd)}")
        else:
            t0 = time.time()
            code = run_command(train_cmd, train_dir, "train.log")
            print(f"[train done] exit={code} elapsed_h={(time.time()-t0)/3600:.2f}", flush=True)
            if code != 0:
                print(f"[train failed] see {train_dir / 'train.log'}", flush=True)
                return code
    else:
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    # 2) Eval at train sensor locations (seed=1) and unseen locations (seed=2,3).
    os.environ["SAVE_CHECKPOINT"] = "off"
    results = {}
    for seed in TEST_SEEDS:
        eval_dir = base / f"eval_testseed{seed}"
        eval_row = make_row(
            old, seed=seed, output_dir=eval_dir,
            run_id=f"{retrain_run_id}_testseed{seed}",
            run_name=f"{baseline}/{TASK}/{PDE} eval testseed={seed}",
        )
        eval_cmd = build_command(eval_row) + ["--eval-only", "--checkpoint", str(ckpt)]
        if args.dry_run:
            print(f"[dry-run] eval seed={seed}: {' '.join(eval_cmd)}")
            continue
        code = run_command(eval_cmd, eval_dir, "eval.log")
        summary = eval_dir / "summary.json"
        metrics = {}
        if summary.exists():
            d = json.loads(summary.read_text())
            for k in ["mse_mean", "mae_mean", "relative_l2_solution_mean", "relative_l1_solution_mean"]:
                if k in d:
                    metrics[k] = d[k]
        results[seed] = {"exit": code, **metrics}
        print(f"[eval seed={seed}] exit={code} {metrics}", flush=True)

    if not args.dry_run:
        print("\n" + "=" * 70)
        print(f"{baseline} / {TASK} / {PDE}  — 换观测点位置后的 test 指标")
        print("=" * 70)
        for seed in TEST_SEEDS:
            print(f"  seed={seed}  {results[seed]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
