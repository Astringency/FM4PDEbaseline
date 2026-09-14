#!/usr/bin/env python
"""Compare paired RecFNO evaluations, retaining sample-level uncertainty."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

PDES = ("poisson", "helmholtz", "darcy", "nsnonbounded")
DISTRIBUTIONS = ("original", "id", "smooth", "rough")
COUNTS = (50, 100, 250, 500, 1000)
CONDITIONS = ("a_only", "u_only", "both")
METRICS = ("rel_l2_a", "rel_l2_u", "joint_rel_l2")


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_stats(fixed, mixed):
    if fixed.shape != mixed.shape or fixed.ndim != 1 or not len(fixed):
        raise ValueError("paired errors must be nonempty vectors of identical length")
    if not np.isfinite(fixed).all() or not np.isfinite(mixed).all():
        raise ValueError("paired errors must be finite")
    rng = np.random.default_rng(20260914)
    delta = mixed - fixed
    bootstrap = []
    for _ in range(10):
        index = rng.integers(0, len(delta), size=(200, len(delta)))
        bootstrap.extend(delta[index].mean(axis=1))
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return dict(fixed_mean=float(fixed.mean()), mixed_mean=float(mixed.mean()),
                change_percent=float(100 * (mixed.mean() / fixed.mean() - 1)),
                paired_mean_difference=float(delta.mean()), paired_difference_ci95_low=float(low),
                paired_difference_ci95_high=float(high), test_size=len(fixed))


def audit_training_pair(fixed_dir, mixed_dir):
    left = json.loads((fixed_dir / "summary.json").read_text())
    right = json.loads((mixed_dir / "summary.json").read_text())
    for field in ("baseline_code_sha256", "normalization_stats_sha256", "commit_hash", "seed",
                  "effective_train_size", "val_size", "num_sensors", "batch_size"):
        if left.get(field) != right.get(field):
            raise ValueError(f"training pair differs in {field}: {left.get(field)} != {right.get(field)}")
    ignored = {"train_sensor_counts", "run_output_dir", "run_prefix", "train_history_json_path", "train_history_jsonl_path"}
    methods = [json.loads(Path(s["config_path"]).read_text())["method"] for s in (left, right)]
    cleaned = [{k:v for k,v in m.items() if k not in ignored} for m in methods]
    if cleaned[0] != cleaned[1]:
        raise ValueError("fixed and mixed training hyperparameters differ beyond sensor count")
    audits = [[json.loads(line) for line in (d / "sensor_count_audit.jsonl").read_text().splitlines()]
              for d in (fixed_dir, mixed_dir)]
    for a, b in zip(*audits):
        if a["epoch"] != b["epoch"] or a["sample_order_sha256"] != b["sample_order_sha256"]:
            raise ValueError("fixed and mixed training sample order differs")
    histories = [json.loads(s["train_history"]) if isinstance(s["train_history"], str) else s["train_history"]
                 for s in (left, right)]
    return {regime:dict(completed_epochs=len(h["train_loss"]), best_epoch=h["best_epoch"],
                       best_val_loss=h["best_val_loss"], train_time_hours=s["train_time"] / 3600,
                       checkpoint_sha256=s["checkpoint_sha256"])
            for regime, h, s in zip(("fixed500", "mixed"), histories, (left, right))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_root", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    rows, missing, training = [], [], {}
    for pde in PDES:
        directories = [args.study_root / "runs" / pde / regime for regime in ("fixed500", "mixed")]
        if all((d / "train/summary.json").is_file() for d in directories):
            training[pde] = audit_training_pair(*(d / "train" for d in directories))
        for distribution in DISTRIBUTIONS:
            for count in COUNTS:
                for condition in CONDITIONS:
                    cells = [d / "evaluation" / distribution / f"sensors={count}" / condition for d in directories]
                    if not all((cell / "summary.json").is_file() for cell in cells):
                        missing.append([pde, distribution, count, condition])
                        continue
                    summaries = [json.loads((cell / "summary.json").read_text()) for cell in cells]
                    for cell, summary in zip(cells, summaries):
                        if summary.get("status") != "success" or file_sha256(cell / "per_sample.npz") != summary.get("per_sample_sha256"):
                            raise ValueError(f"invalid or corrupted evaluation cell: {cell}")
                    identities = [s["identity"] for s in summaries]
                    for field in ("test_file_sha256", "test_size", "sensor_count", "sensor_seed", "condition", "code_revision"):
                        if identities[0][field] != identities[1][field]:
                            raise ValueError(f"evaluation pair differs in {field}: {cells}")
                    with np.load(cells[0] / "per_sample.npz") as fixed, np.load(cells[1] / "per_sample.npz") as mixed:
                        for field in ("sample_ids", "base_mask_ids"):
                            if not np.array_equal(fixed[field], mixed[field]):
                                raise ValueError(f"evaluation pair has different {field}: {cells}")
                        for metric in METRICS:
                            for data, summary in zip((fixed, mixed), summaries):
                                if not np.isclose(data[metric].mean(), summary[metric], rtol=1e-12, atol=1e-12):
                                    raise ValueError("summary mean disagrees with per-sample errors")
                            rows.append(dict(pde=pde, distribution=distribution, sensor_count=count,
                                             condition=condition, metric=metric,
                                             **paired_stats(fixed[metric], mixed[metric])))
    if missing and not args.allow_partial:
        raise ValueError(f"{len(missing)} paired evaluation cells remain incomplete")
    payload = dict(status="partial" if missing else "complete", missing=missing, training=training,
                   uncertainty="Paired bootstrap over test samples; one training seed, no estimate of training-seed variability.",
                   rows=rows)
    (args.study_root / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    if rows:
        with (args.study_root / "comparison.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(dict(status=payload["status"], paired_cells=len(rows)//3, missing_cells=len(missing), training=training), indent=2))


if __name__ == "__main__":
    main()
