#!/usr/bin/env python
"""Evaluate sensor-location generalization for sparse_solution checkpoints.

For each baseline (senseiver / recfno / voronoicnn), load the seed=1
sparse_solution / poisson checkpoint from the old ``main_results_20260624_161620``
outputs and run ``baselines.run --eval-only`` with test sensor locations at
seed = 1 (train locations), 2, and 3 (unseen locations). This isolates the
effect of moving the sparse observation points at test time.

Usage:
    python scripts/experiments/eval_sensor_generalization.py --dry-run
    python scripts/experiments/eval_sensor_generalization.py [--baselines recfno]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_one import build_command  # noqa: E402

OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/home/zhangxf/share/zhangxfA100/large_storage/outputs/FM4PDEbaseline"))
OLD_RUNS = OUTPUT_ROOT / "main_results_20260624_161620" / "runs" / "main_results"
EVAL_ROOT = OUTPUT_ROOT / "sensor_generalization_eval"
DATA_ROOT = "/home/zhangxf/share/zhangxfA100/large_storage/PDEdata/"

BASELINES = ["senseiver", "recfno", "voronoicnn"]
TEST_SEEDS = [1, 2, 3]


def find_checkpoint(baseline: str) -> Path:
    pat = f"task_group=sparse_solution_main_amortized/pde=poisson/baseline={baseline}/seed=1/run=*/*.pt"
    hits = list(OLD_RUNS.glob(pat))
    if not hits:
        raise FileNotFoundError(f"checkpoint not found: {OLD_RUNS / pat}")
    return hits[0]


def load_old_args(baseline: str) -> dict:
    run_dir = find_checkpoint(baseline).parent
    configs = list(run_dir.glob("*_config.json"))
    if not configs:
        raise FileNotFoundError(f"config not found in {run_dir}")
    return json.loads(configs[0].read_text())["args"]


def build_eval_command(baseline: str, test_seed: int) -> tuple[list[str], Path]:
    old = load_old_args(baseline)
    ckpt = find_checkpoint(baseline)
    out_dir = EVAL_ROOT / baseline / f"testseed{test_seed}"

    row = {
        "baseline": old["baseline"],
        "pde": old["pde"],
        "task": old["task"],
        "seed": test_seed,
        "device": old.get("device", "cuda"),
        "output_dir": str(out_dir),
        "run_id": f"{old['run_id']}_testseed{test_seed}",
        "run_name": f"{old['run_name']} testseed={test_seed}",
        "task_group": old.get("task_group", "sparse_solution_main_amortized"),
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

    env = dict(os.environ)
    env["DATA_ROOT"] = DATA_ROOT
    env["SAVE_CHECKPOINT"] = "off"
    env["PYTHON"] = sys.executable

    # build_command uses DATA_ROOT/DEVICE/PYTHON from the process env.
    os.environ["DATA_ROOT"] = DATA_ROOT
    os.environ["SAVE_CHECKPOINT"] = "off"
    os.environ["PYTHON"] = sys.executable
    cmd = build_command(row)
    cmd += ["--eval-only", "--checkpoint", str(ckpt)]
    return cmd, out_dir


def run(cmd: list[str], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "eval.log"
    print(f"[run] {' '.join(cmd)}", flush=True)
    with log_path.open("w") as lf:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
        code = proc.wait()
    print(f"[done] exit={code} log={log_path}", flush=True)
    return code


def metrics_from_summary(out_dir: Path) -> dict:
    s = out_dir / "summary.json"
    if not s.exists():
        return {"error": "no summary.json"}
    d = json.loads(s.read_text())
    keys = [
        "mse_mean", "mse_std",
        "mae_mean", "mae_std",
        "relative_l2_mean", "relative_l2_std",
        "relative_l1_mean", "relative_l1_std",
        "best_val_loss",
    ]
    out = {k: d.get(k) for k in keys if k in d}
    # also try any field containing relative
    for k in d:
        if "relative" in k and "mean" in k:
            out[k] = d[k]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--baselines", nargs="*", default=BASELINES)
    ap.add_argument("--seeds", nargs="*", type=int, default=TEST_SEEDS)
    args = ap.parse_args()

    os.environ.setdefault("DATA_ROOT", DATA_ROOT)

    results = {}
    for baseline in args.baselines:
        for seed in args.seeds:
            cmd, out_dir = build_eval_command(baseline, seed)
            key = (baseline, seed)
            if args.dry_run:
                print(f"[dry-run] {baseline} seed={seed} -> {out_dir}")
                continue
            code = run(cmd, out_dir)
            results[key] = {"exit": code, "metrics": metrics_from_summary(out_dir)}
            print(f"  {key} -> {results[key]}", flush=True)

    if args.dry_run:
        return 0

    print("\n" + "=" * 70)
    print("SUMMARY (test metrics)")
    print("=" * 70)
    for baseline in args.baselines:
        print(f"\n### {baseline}")
        for seed in args.seeds:
            m = results.get((baseline, seed), {}).get("metrics", {})
            if "error" in m:
                print(f"  seed={seed}: ERROR {m['error']} (exit={results.get((baseline,seed),{}).get('exit')})")
            else:
                fmt = {k: (f"{v:.6g}" if isinstance(v, (int, float)) else v) for k, v in m.items()}
                print(f"  seed={seed}: {fmt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
