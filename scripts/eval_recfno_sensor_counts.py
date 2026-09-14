#!/usr/bin/env python
"""Evaluate one RecFNO checkpoint on matched sensor budgets and distributions.

Stores per-sample errors, IDs, layout hashes and a small prediction sample.
Only complete cells whose input/checkpoint identities match are reused.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from baselines.common.data_adapter import PDEBatchDataset, build_default_registry, pde_collate
from baselines.methods.base import _to_device_batch
from baselines.run import _make_inference_batch, _multicondition_metrics, load_baseline_checkpoint
from scripts.experiments.provenance import repository_revision
from scripts.recfno_sensor_count_study import sha256
from scripts.run_eval import DISTRIBUTION_TEST_FILES
from scripts.run_recfno_variable_sensors import parse_counts
from scripts.recfno_study_lock import output_lock

METRICS = ("rel_l2_a", "rel_l2_u", "joint_rel_l2")
CONDITIONS = ("a_only", "u_only", "both")


def evaluate_cell(model, dataset, *, device, batch_size, output, identity):
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    errors_path = output / "per_sample.npz"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("identity") != identity or summary.get("status") != "success":
            raise ValueError(f"existing evaluation has a different identity: {output}")
        if sha256(errors_path) != summary.get("per_sample_sha256"):
            raise ValueError(f"existing per-sample data failed its checksum: {output}")
        if sha256(output / "example_predictions.pt") != summary.get("example_predictions_sha256"):
            raise ValueError(f"existing example predictions failed their checksum: {output}")
        return summary
    dataset.set_condition_mode(identity["condition"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pde_collate,
                        num_workers=0)
    errors = {key: [] for key in METRICS}
    ids, layout_ids = [], []
    start = time.monotonic()
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            inference = _to_device_batch(_make_inference_batch(batch), torch.device(device))
            prediction = model.predict_physical(inference).detach().cpu()
            if prediction.shape != batch.target_fields.shape or not torch.isfinite(prediction).all():
                raise ValueError("invalid prediction shape or nonfinite prediction")
            metrics = _multicondition_metrics(prediction, batch.target_fields, batch)
            for key in METRICS:
                errors[key].extend(metrics[key + "_values"])
            ids.extend(batch.global_sample_ids)
            layout_ids.extend(batch.metadata["base_mask_ids"])
            if step == 0:
                torch.save(dict(identity=identity, sample_ids=batch.global_sample_ids[:3],
                                prediction=prediction[:3], target=batch.target_fields[:3],
                                input_fields=batch.input_fields[:3], mask=batch.mask[:3]),
                           output / "example_predictions.pt")
    if len(ids) != identity["test_size"] or len(set(ids)) != len(ids):
        raise ValueError("evaluation did not cover exactly the requested distinct samples")
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in errors.items()}
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("nonfinite relative errors")
    np.savez_compressed(errors_path, sample_ids=np.asarray(ids), base_mask_ids=np.asarray(layout_ids), **arrays)
    summary = dict(status="success", identity=identity, test_size=len(ids),
                   elapsed_seconds=time.monotonic() - start, per_sample_sha256=sha256(errors_path),
                   example_predictions_sha256=sha256(output / "example_predictions.pt"))
    for key, array in arrays.items():
        summary[key] = float(array.mean())
        summary[key + "_std"] = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        summary[key + "_median"] = float(np.median(array))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-summary", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sensor-counts", default="50,100,250,500,1000")
    parser.add_argument("--distributions", default="original,id,smooth,rough")
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with output_lock(args.output_dir, "evaluation"):
        _run_locked(args)


def _run_locked(args):
    source = json.loads(args.train_summary.read_text())
    if source.get("status") != "success" or source.get("task") != "sparse_solution_multicondition":
        raise ValueError("source must be a successful multicondition training run")
    checkpoint = Path(source["checkpoint_path"])
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != source["checkpoint_sha256"]:
        raise ValueError("source checkpoint checksum mismatch")
    model = load_baseline_checkpoint(checkpoint, map_location=args.device, require_provenance=True,
                                    expected=dict(baseline="recfno", pde=source["pde"],
                                                  run_fingerprint=source["run_fingerprint"],
                                                  source_train_run_id=source["run_id"]))
    registry = build_default_registry()
    counts = parse_counts(args.sensor_counts)
    revision = repository_revision(ROOT)
    formal_files = yaml.safe_load((ROOT / "configs/data_files/formal_128.yaml").read_text())["data_files"]
    summaries = []
    for distribution in args.distributions.split(","):
        pde = source["pde"]
        test_file = (formal_files[pde]["test"][0] if distribution == "original"
                     else DISTRIBUTION_TEST_FILES[distribution][pde])
        test_path = args.data_root / test_file
        print(f"[evaluation] load {test_path}", flush=True)
        file_hash = sha256(test_path)
        raw = registry.load_raw(pde, args.data_root, split="test", max_samples=args.test_size,
                                strict_size=True, load_full_trajectory=False,
                                data_files={"test": [test_file]})
        base = registry.make_task(raw, pde, "sparse_solution_multicondition", num_sensors=500,
                                  sensor_mode="random_per_sample", sensor_budget_mode="total",
                                  condition_mode="mixed", seed=1, experiment_mode="paper")
        for count in counts:
            dataset = PDEBatchDataset(replace(base, metadata={**base.metadata, "num_sensors": count}))
            for condition in CONDITIONS:
                identity = dict(checkpoint_sha256=checkpoint_hash, train_run_id=source["run_id"],
                                train_run_fingerprint=source["run_fingerprint"], pde=pde,
                                test_file=test_file, test_file_sha256=file_hash, distribution=distribution,
                                test_size=args.test_size, sensor_count=count, sensor_seed=1,
                                condition=condition, code_revision=revision,
                                metric="physical_field_relative_L2_per_sample", batch_size=args.batch_size)
                summaries.append(evaluate_cell(model, dataset, device=args.device, batch_size=args.batch_size,
                                                output=args.output_dir / distribution / f"sensors={count}" / condition,
                                                identity=identity))
        del dataset, base, raw
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(dict(status="success", cells=summaries), indent=2) + "\n")


if __name__ == "__main__":
    main()
