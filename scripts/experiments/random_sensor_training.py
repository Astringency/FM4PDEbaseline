#!/usr/bin/env python
"""Train recfno with *random per-sample* sensor locations and test whether it
generalizes better to unseen sensor locations than fixed-sensor training.

Contrast:
  - fixed training: every sample shares one deterministic sensor mask (seed=1).
  - random training: every sample gets a fresh random mask each epoch.

Evaluation:
  - random_test: fresh random masks at test time (in-distribution).
  - fixed seed=2/3: a single unseen fixed mask, directly comparable to the
    fixed-sensor model's ~28% relative L2 on seed=2/3.

Usage:
    python scripts/experiments/random_sensor_training.py --smoke   # tiny sanity
    python scripts/experiments/random_sensor_training.py           # full run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.common.data_adapter import PDEBatch, build_default_registry  # noqa: E402
from baselines.common.sensors import make_sensor_mask  # noqa: E402
from baselines.methods.base import _to_device_batch  # noqa: E402
from baselines.run import BASELINES  # noqa: E402

DATA_ROOT = "/home/zhangxf/share/zhangxfA100/large_storage/PDEdata/"
OUTPUT_ROOT = Path("/home/zhangxf/share/zhangxfA100/large_storage/outputs/FM4PDEbaseline")
RETRAIN_CONFIG = (
    OUTPUT_ROOT / "sensor_generalization_retrain" / "recfno_sparse_forward_poisson" / "train"
    / "retrain_recfno_sparse_forward_poisson_s1_config.json"
)

NUM_SENSORS = 500
BATCH_SIZE = 16


def load_full_batch(split: str, size: int) -> PDEBatch:
    registry = build_default_registry()
    ds = registry.make_dataset(
        pde_name="poisson",
        data_root=DATA_ROOT,
        task="forward",
        split=split,
        max_samples=size,
        train_shards=5,
        seed=1,
        scalar_param_mode="metadata",
        data_loading_mode="eager",
        load_full_trajectory=False,
        experiment_mode="paper",
    )
    return ds.batch


class RandomSensorDataset(Dataset):
    """Each sample gets a fresh random mask on every __getitem__ call."""

    def __init__(self, full: PDEBatch, num_sensors: int):
        self.full = full
        self.num_sensors = num_sensors

    def __len__(self) -> int:
        return int(self.full.input_fields.shape[0])

    def __getitem__(self, i: int) -> PDEBatch:
        inp = self.full.input_fields[i : i + 1]
        tgt = self.full.target_fields[i : i + 1]
        c, h, w = inp.shape[1], inp.shape[2], inp.shape[3]
        mask = torch.zeros(1, 1, h, w)
        total = h * w
        num = min(self.num_sensors, total)
        perm = torch.randperm(total)[:num]
        mask.reshape(-1)[perm] = 1.0
        mask = mask.repeat(1, c, 1, 1)
        return self._batch(inp, tgt, mask, self.full.full_tensor[i : i + 1])

    def _batch(self, inp: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor, full: torch.Tensor) -> PDEBatch:
        masked = inp * mask
        return PDEBatch(
            pde_name=self.full.pde_name,
            task="sparse_forward",
            full_tensor=full,
            input_fields=masked,
            target_fields=tgt,
            coords=None,
            mask=mask,
            obs_values=None,
            obs_coords=None,
            channel_names=self.full.channel_names,
            input_channel_names=self.full.input_channel_names,
            target_channel_names=self.full.target_channel_names,
            metadata={},
            pde_params={},
            split=self.full.split,
            sample_indices=None,
            global_sample_ids=[],
            file_paths=[],
        )


class FixedSensorDataset(Dataset):
    """One fixed mask (seed) shared by all samples."""

    def __init__(self, full: PDEBatch, num_sensors: int, seed: int):
        self.full = full
        self.num_sensors = num_sensors
        c, h, w = full.input_fields.shape[1], full.input_fields.shape[2], full.input_fields.shape[3]
        self.mask = make_sensor_mask((c, h, w), num_sensors, "random", seed)  # [c,h,w]
        self.masked = full.input_fields * self.mask.unsqueeze(0)

    def __len__(self) -> int:
        return int(self.full.input_fields.shape[0])

    def __getitem__(self, i: int) -> PDEBatch:
        inp = self.masked[i : i + 1]
        tgt = self.full.target_fields[i : i + 1]
        mask = self.mask.unsqueeze(0)
        return PDEBatch(
            pde_name=self.full.pde_name,
            task="sparse_forward",
            full_tensor=self.full.full_tensor[i : i + 1],
            input_fields=inp,
            target_fields=tgt,
            coords=None,
            mask=mask,
            obs_values=None,
            obs_coords=None,
            channel_names=self.full.channel_names,
            input_channel_names=self.full.input_channel_names,
            target_channel_names=self.full.target_channel_names,
            metadata={},
            pde_params={},
            split=self.full.split,
            sample_indices=None,
            global_sample_ids=[],
            file_paths=[],
        )


def collate(items: list[PDEBatch]) -> PDEBatch:
    first = items[0]
    return PDEBatch(
        pde_name=first.pde_name,
        task=first.task,
        full_tensor=torch.cat([b.full_tensor for b in items], dim=0),
        input_fields=torch.cat([b.input_fields for b in items], dim=0),
        target_fields=torch.cat([b.target_fields for b in items], dim=0),
        coords=None,
        mask=torch.cat([b.mask for b in items], dim=0),
        obs_values=None,
        obs_coords=None,
        channel_names=first.channel_names,
        input_channel_names=first.input_channel_names,
        target_channel_names=first.target_channel_names,
        metadata=first.metadata,
        pde_params={},
        split=first.split,
        sample_indices=None,
        global_sample_ids=[],
        file_paths=[],
    )


def make_loader(ds: Dataset, shuffle: bool, batch_size: int = BATCH_SIZE) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, num_workers=0)


def evaluate(model, loader: DataLoader, device: str, name: str) -> dict:
    model.eval()
    preds: list[torch.Tensor] = []
    tgts: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device_batch(batch, torch.device(device))
            pred = model.predict_physical(batch)
            preds.append(pred.detach().cpu())
            tgts.append(batch.target_fields.detach().cpu())
    pred = torch.cat(preds, dim=0)
    tgt = torch.cat(tgts, dim=0)
    mse = float((pred - tgt).pow(2).mean())
    mae = float((pred - tgt).abs().mean())
    diff = torch.linalg.vector_norm((pred - tgt).reshape(pred.shape[0], -1), dim=1)
    denom = torch.linalg.vector_norm(tgt.reshape(tgt.shape[0], -1), dim=1).clamp_min(1e-12)
    rel_l2 = float((diff / denom).mean())
    print(f"[eval {name}] mse={mse:.6g} mae={mae:.6g} rel_l2={rel_l2:.6g}", flush=True)
    return {"mse_mean": mse, "mae_mean": mae, "relative_l2_solution_mean": rel_l2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--train-size", type=int, default=50000)
    ap.add_argument("--val-size", type=int, default=1000)
    ap.add_argument("--test-size", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.smoke:
        train_size, val_size, test_size, epochs = 32, 16, 16, 2
    else:
        train_size, val_size, test_size, epochs = args.train_size, args.val_size, args.test_size, args.epochs

    torch.manual_seed(0)

    cfg = json.loads(RETRAIN_CONFIG.read_text())
    method_cfg = dict(cfg["method"])
    method_cfg["device"] = args.device
    method_cfg["epochs"] = epochs
    data_spec = cfg["data_spec"]

    model = BASELINES["recfno"]().build(method_cfg, data_spec).to(args.device)

    print("[data] loading full forward splits ...", flush=True)
    full_train = load_full_batch("train", train_size)
    train_batch = _slice_full(full_train, 0, train_size - val_size)
    val_batch = _slice_full(full_train, train_size - val_size, train_size)
    full_test = load_full_batch("test", test_size)

    train_loader = make_loader(RandomSensorDataset(train_batch, NUM_SENSORS), shuffle=True)
    val_loader = make_loader(RandomSensorDataset(val_batch, NUM_SENSORS), shuffle=False)
    test_random = make_loader(RandomSensorDataset(full_test, NUM_SENSORS), shuffle=False)
    test_fixed2 = make_loader(FixedSensorDataset(full_test, NUM_SENSORS, 2), shuffle=False)
    test_fixed3 = make_loader(FixedSensorDataset(full_test, NUM_SENSORS, 3), shuffle=False)

    print(f"[train] recfno random-per-sample sensors, epochs={epochs} train={train_size} val={val_size}", flush=True)
    t0 = time.time()
    history = model.fit(train_loader, val_loader)
    print(f"[train done] elapsed_h={(time.time()-t0)/3600:.2f} "
          f"best_val={history.get('best_val_loss')} best_epoch={history.get('best_epoch')} "
          f"early_stopped={history.get('early_stopped')} stop_epoch={history.get('stop_epoch')}", flush=True)

    results = {
        "random_test": evaluate(model, test_random, args.device, "random_test"),
        "fixed_seed2": evaluate(model, test_fixed2, args.device, "fixed_seed2"),
        "fixed_seed3": evaluate(model, test_fixed3, args.device, "fixed_seed3"),
    }

    print("\n" + "=" * 70)
    print("recfno / sparse_forward / poisson — 随机观测点训练 的 test 指标")
    print("=" * 70)
    for name, r in results.items():
        print(f"  {name:16s} {r}")
    return 0


def _slice_full(batch: PDEBatch, start: int, end: int) -> PDEBatch:
    sl = slice(start, end)
    return PDEBatch(
        pde_name=batch.pde_name,
        task=batch.task,
        full_tensor=batch.full_tensor[sl],
        input_fields=batch.input_fields[sl],
        target_fields=batch.target_fields[sl],
        coords=batch.coords[sl] if batch.coords is not None and batch.coords.shape[0] == batch.full_tensor.shape[0] else batch.coords,
        mask=batch.mask,
        obs_values=batch.obs_values[sl] if batch.obs_values is not None else None,
        obs_coords=batch.obs_coords[sl] if batch.obs_coords is not None else None,
        channel_names=batch.channel_names,
        input_channel_names=batch.input_channel_names,
        target_channel_names=batch.target_channel_names,
        metadata=batch.metadata,
        pde_params=batch.pde_params,
        split=batch.split,
        sample_indices=batch.sample_indices[sl] if batch.sample_indices is not None else None,
        global_sample_ids=batch.global_sample_ids[start:end],
        file_paths=batch.file_paths,
    )


if __name__ == "__main__":
    raise SystemExit(main())
