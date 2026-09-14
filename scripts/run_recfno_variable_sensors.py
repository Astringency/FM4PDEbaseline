#!/usr/bin/env python
"""Opt-in RecFNO experiment; the existing runner and model files stay unchanged.

Each training batch receives one deterministic, uniformly sampled location
budget. Validation and test keep --num-sensors. Tuple dataset keys carry the
epoch and budget through worker prefetch queues, including persistent workers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from baselines.common.data_adapter import PDEBatchDataset, pde_collate
from baselines import run as runner
from scripts.recfno_sensor_count_study import sha256
from scripts.recfno_study_lock import output_lock

PROTOCOL = "recfno-batch-sensor-count-v1"


def parse_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(x.strip()) for x in value.split(","))
    if not counts or min(counts) <= 0 or len(set(counts)) != len(counts):
        raise ValueError("sensor counts must be distinct positive integers")
    return counts


def count_for_batch(counts: tuple[int, ...], seed: int, epoch: int, step: int) -> int:
    payload = f"{PROTOCOL}|{seed}|{epoch}|{step}".encode()
    draw = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return counts[draw % len(counts)]


class BatchCountDataset(PDEBatchDataset):
    def __init__(self, source: PDEBatchDataset):
        if source.batch.task != "sparse_solution_multicondition" or source.batch.split != "train":
            raise ValueError("batch-count sampling is restricted to multicondition training")
        super().__init__(replace(source.batch, metadata=dict(source.batch.metadata)))
        self._key_epoch = None

    @property
    def epoch(self):
        return super().epoch if self._key_epoch is None else self._key_epoch

    def __getitem__(self, key):
        index, count, epoch, step = key
        previous_count = self.batch.metadata["num_sensors"]
        previous_epoch = self._key_epoch
        try:
            self.batch.metadata["num_sensors"] = count
            self._key_epoch = epoch
            item = super().__getitem__(index)
            item.metadata.update(train_sensor_count=count, sensor_batch_step=step)
            return item
        finally:
            self.batch.metadata["num_sensors"] = previous_count
            self._key_epoch = previous_epoch


class SensorCountBatchSampler:
    def __init__(self, batch_sampler, dataset, counts, seed):
        self.batch_sampler = batch_sampler
        self.dataset = dataset
        self.counts = tuple(counts)
        self.seed = seed
        grid_points = dataset.batch.target_fields[0, 0].numel()
        if max(self.counts) > grid_points:
            raise ValueError(f"sensor count exceeds the {grid_points} spatial locations")

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        epoch = self.dataset.epoch
        for step, indices in enumerate(self.batch_sampler):
            count = count_for_batch(self.counts, self.seed, epoch, step)
            yield [(index, count, epoch, step) for index in indices]


class AuditedSensorLoader(DataLoader):
    audit_path: Path | None = None
    iteration = 0

    def __iter__(self):
        batch_counts, sample_counts, conditions = Counter(), Counter(), Counter()
        sample_hash, schedule_hash = hashlib.sha256(), hashlib.sha256()
        epoch = self.dataset.epoch
        iteration = self.iteration
        self.iteration += 1
        complete = False
        try:
            for batch in super().__iter__():
                count = int(batch.metadata["train_sensor_count"])
                batch_counts[count] += 1
                sample_counts[count] += len(batch.global_sample_ids)
                sample_hash.update(json.dumps(batch.global_sample_ids).encode())
                schedule_hash.update(f"{count},".encode())
                conditions.update(f"{count}:{c}" for c in batch.metadata["condition_modes"])
                yield batch
            complete = True
        finally:
            record = dict(iteration=iteration, epoch=epoch, complete=complete,
                          batches_by_count=dict(batch_counts), samples_by_count=dict(sample_counts),
                          conditions_by_count=dict(conditions), sample_order_sha256=sample_hash.hexdigest(),
                          count_schedule_sha256=schedule_hash.hexdigest())
            if self.audit_path is not None:
                with self.audit_path.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            print("[sensor-count audit] " + json.dumps(record, sort_keys=True), flush=True)


def make_training_loader(source_loader, counts, seed, audit_path=None):
    dataset = BatchCountDataset(source_loader.dataset)
    sampler = SensorCountBatchSampler(source_loader.batch_sampler, dataset, counts, seed)
    kwargs = dict(batch_sampler=sampler, collate_fn=pde_collate,
                  num_workers=source_loader.num_workers, pin_memory=source_loader.pin_memory)
    if source_loader.num_workers:
        kwargs.update(persistent_workers=source_loader.persistent_workers,
                      prefetch_factor=source_loader.prefetch_factor)
    loader = AuditedSensorLoader(dataset, **kwargs)
    loader.audit_path = audit_path
    return loader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--train-sensor-counts", default="50,100,250,500,1000")
    parser.add_argument("--count-seed", type=int, default=1)
    extra, remaining = parser.parse_known_args(argv)
    counts = parse_counts(extra.train_sensor_counts)
    args = runner.parse_args(remaining)
    if args.baseline != "recfno" or args.task != "sparse_solution_multicondition":
        raise ValueError("this opt-in experiment supports only multicondition RecFNO")
    if args.sensor_mode != "random_per_sample":
        raise ValueError("the experiment requires random_per_sample locations")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output, "training"):
        _run_locked(extra, remaining, counts, args, output)


def _run_locked(extra, remaining, counts, args, output):
    extension_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    contract = dict(protocol=PROTOCOL, train_sensor_counts=counts, count_seed=extra.count_seed,
                    sampling="uniform_per_batch", validation_sensor_count=args.num_sensors,
                    script_sha256=extension_hash, original_argv=remaining)
    contract_path = output / "sensor_count_experiment.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != json.loads(json.dumps(contract)):
        raise ValueError("output directory already contains a different sensor-count experiment")
    summary_path = output / "summary.json"
    if args.train_only and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        checkpoint = Path(summary.get("checkpoint_path", ""))
        expected_config = runner._config_content_sha256(args.config, runner.load_yaml(args.config))
        if (summary.get("status") != "success" or not checkpoint.is_file()
                or sha256(checkpoint) != summary.get("checkpoint_sha256")
                or summary.get("config_content_sha256") != expected_config
                or summary.get("baseline_code_sha256") != runner.baseline_code_sha256("recfno", root=ROOT)):
            raise ValueError("existing completed training result failed identity/checksum validation")
        print(f"[training] reuse checksum-validated completed result {summary_path}", flush=True)
        return
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    for key, value in dict(train_sensor_counts=list(counts), sensor_count_seed=extra.count_seed,
                           sensor_count_protocol=PROTOCOL, sensor_count_code_sha256=extension_hash).items():
        remaining.extend(["--method-override", f"{key}={json.dumps(value)}"])
    original = runner.build_pde_dataloader

    def build_loader(dataset, args, shuffle, batch_size=None):
        loader = original(dataset, args, shuffle, batch_size)
        if shuffle and dataset.batch.split == "train" and not args.eval_only:
            return make_training_loader(loader, counts, extra.count_seed, output / "sensor_count_audit.jsonl")
        return loader

    runner.build_pde_dataloader = build_loader
    start = time.monotonic()
    try:
        runner.main(remaining)
    finally:
        runner.build_pde_dataloader = original
        if str(args.device).startswith("cuda") and torch.cuda.is_initialized():
            (output / "cuda_peak_memory.json").write_text(json.dumps(dict(
                max_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                max_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
                elapsed_seconds=time.monotonic() - start), indent=2) + "\n")


if __name__ == "__main__":
    main()
