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
from scripts.experiments.sensor_generalization_common import (  # noqa: E402
    make_provenance_row,
    write_eval_request,
)

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
    os.environ["DATA_ROOT"] = DATA_ROOT
    os.environ["PYTHON"] = sys.executable

    old = load_source_args(baseline)
    base = OUTPUT_ROOT / "sensor_generalization_retrain" / f"{baseline}_{TASK}_{PDE}"
    train_dir = base / "train"
    train_row = make_provenance_row(
        old,
        execution_mode="train",
        output_dir=train_dir,
        run_label=f"retrain_{baseline}_{TASK}_{PDE}_s1",
        seed=1,
        sensor_seed=1,
        sensor_mode="random_per_sample",
    )
    ckpt = train_dir / f"{train_row['run_id']}.pt"

    # 1) Retrain (writes checkpoint).
    if not args.skip_train:
        os.environ["SAVE_CHECKPOINT"] = "amortized"
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

    if args.dry_run and not ckpt.exists():
        print(
            "[dry-run] eval rows deferred: their run fingerprints include the actual checkpoint SHA-256 "
            f"and can only be constructed after {ckpt} exists"
        )
        return 0

    # 2) Eval at train sensor locations (seed=1) and unseen locations (seed=2,3).
    os.environ["SAVE_CHECKPOINT"] = "off"
    results = {}
    for seed in TEST_SEEDS:
        eval_dir = base / f"eval_testseed{seed}"
        eval_row = make_provenance_row(
            train_row,
            execution_mode="eval_only",
            output_dir=eval_dir,
            run_label=f"sensor_gen_{baseline}_{TASK}_{PDE}_testseed{seed}",
            seed=1,
            sensor_seed=seed,
            sensor_mode="random_per_sample",
            source_train_run_id=train_row["run_id"],
            source_train_run_fingerprint=train_row["run_fingerprint"],
            source_train_seed=1,
            checkpoint_path=ckpt,
        )
        eval_cmd = build_command(eval_row)
        if args.dry_run:
            print(f"[dry-run] eval sensor_seed={seed} run_id={eval_row['run_id']}: {' '.join(eval_cmd)}")
            continue
        write_eval_request(eval_dir, eval_row, eval_cmd)
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
