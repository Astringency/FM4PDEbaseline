#!/usr/bin/env python3
"""Compare train-fit and test distributions for the formal FM4PDE cohort.

The script follows ``configs/data_files/formal_128.yaml`` and the split sizes in
``configs/experiments/main_results.yaml``: 45,000 fitting samples are taken from
the 50,000-sample train/validation pool and the first 1,000 configured test
samples are evaluated. MATLAB-v5 arrays are profiled exactly. Large HDF5 arrays
are profiled with deterministic, shard-stratified train sampling; the complete
1,000-sample test cohort is used by default.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import scipy.io
import yaml


ArrayTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class QuantitySpec:
    name: str
    source_key: str
    transform: ArrayTransform


@dataclass
class MomentAccumulator:
    sample_count: int = 0
    value_count: int = 0
    total: float = 0.0
    total_squares: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    sample_means: list[float] = field(default_factory=list)
    sample_rms: list[float] = field(default_factory=list)

    def update(self, values: np.ndarray, batch_size: int = 32) -> None:
        values = np.asarray(values)
        if values.ndim == 0:
            values = values.reshape(1, 1)
        if values.shape[0] == 0:
            return
        for start in range(0, int(values.shape[0]), batch_size):
            chunk = np.asarray(values[start : start + batch_size], dtype=np.float64)
            flat = chunk.reshape(chunk.shape[0], -1)
            row_sums = flat.sum(axis=1, dtype=np.float64)
            row_sumsq = np.square(flat).sum(axis=1, dtype=np.float64)
            row_count = int(flat.shape[1])
            self.sample_count += int(flat.shape[0])
            self.value_count += int(flat.size)
            self.total += float(row_sums.sum(dtype=np.float64))
            self.total_squares += float(row_sumsq.sum(dtype=np.float64))
            self.minimum = min(self.minimum, float(flat.min()))
            self.maximum = max(self.maximum, float(flat.max()))
            self.sample_means.extend((row_sums / row_count).tolist())
            self.sample_rms.extend(np.sqrt(row_sumsq / row_count).tolist())

    def summary(self) -> dict[str, float | int]:
        if self.value_count == 0:
            raise ValueError("cannot summarize an empty accumulator")
        mean = self.total / self.value_count
        variance = max(self.total_squares / self.value_count - mean * mean, 0.0)
        sample_means = np.asarray(self.sample_means, dtype=np.float64)
        sample_rms = np.asarray(self.sample_rms, dtype=np.float64)
        return {
            "samples_observed": self.sample_count,
            "values_observed": self.value_count,
            "mean": mean,
            "variance": variance,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
            "sample_mean_std": float(sample_means.std(ddof=1)) if sample_means.size > 1 else 0.0,
            "sample_rms_mean": float(sample_rms.mean()),
            "sample_rms_p05": float(np.quantile(sample_rms, 0.05)),
            "sample_rms_median": float(np.quantile(sample_rms, 0.50)),
            "sample_rms_p95": float(np.quantile(sample_rms, 0.95)),
        }


IDENTITY: ArrayTransform = lambda value: value
LAST_TIME_2D: ArrayTransform = lambda value: value[:, -1, :]
LAST_TIME_3D: ArrayTransform = lambda value: value[..., -1]


QUANTITIES: dict[str, list[QuantitySpec]] = {
    "poisson": [
        QuantitySpec("f", "f_data", IDENTITY),
        QuantitySpec("phi", "phi_data", IDENTITY),
    ],
    "helmholtz": [
        QuantitySpec("f", "f_data", IDENTITY),
        QuantitySpec("psi", "psi_data", IDENTITY),
    ],
    "darcy": [
        QuantitySpec("a", "thresh_a_data", IDENTITY),
        QuantitySpec("p", "thresh_p_data", IDENTITY),
    ],
    "burger": [
        QuantitySpec("u0", "input", IDENTITY),
        QuantitySpec("u_trajectory", "output", IDENTITY),
        QuantitySpec("uT", "output", LAST_TIME_2D),
    ],
    "nsnonbounded": [
        QuantitySpec("w0", "w0", IDENTITY),
        QuantitySpec("w_trajectory_t1_t10", "w", IDENTITY),
        QuantitySpec("wT", "w", LAST_TIME_3D),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("~/share/PDEdata").expanduser())
    parser.add_argument("--data-files-config", type=Path, default=Path("configs/data_files/formal_128.yaml"))
    parser.add_argument("--train-fit-size", type=int, default=45_000)
    parser.add_argument("--test-size", type=int, default=1_000)
    parser.add_argument("--h5-train-samples", type=int, default=1_000)
    parser.add_argument("--h5-test-samples", type=int, default=1_000)
    parser.add_argument(
        "--h5-io-batch-size",
        type=int,
        default=128,
        help="Maximum HDF5 samples loaded per read while accumulating moments.",
    )
    parser.add_argument("--pdes", nargs="+", choices=sorted(QUANTITIES), default=list(QUANTITIES))
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/train_test_distribution"))
    return parser.parse_args()


def _load_file_config(path: Path) -> dict[str, dict[str, list[str]]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    mapping = payload.get("data_files", payload)
    if not isinstance(mapping, dict):
        raise ValueError(f"expected a data_files mapping in {path}")
    return mapping


def _available_counts(paths: list[Path], source_key: str, sample_axis: int) -> list[int]:
    counts = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            counts.append(int(handle[source_key].shape[sample_axis]))
    return counts


def _bounded_counts(available: list[int], population_size: int) -> list[int]:
    remaining = int(population_size)
    counts = []
    for count in available:
        take = min(count, max(remaining, 0))
        counts.append(take)
        remaining -= take
    if remaining > 0:
        raise ValueError(f"requested {population_size} samples but only {sum(available)} are available")
    return counts


def _allocate_sample_quotas(counts: list[int], requested: int) -> list[int]:
    total = sum(counts)
    requested = min(int(requested), total)
    if requested == total:
        return list(counts)
    raw = [requested * count / total for count in counts]
    quotas = [min(count, int(math.floor(value))) for count, value in zip(counts, raw)]
    remaining = requested - sum(quotas)
    order = sorted(range(len(counts)), key=lambda index: (raw[index] - quotas[index], counts[index]), reverse=True)
    for index in order:
        if remaining <= 0:
            break
        if quotas[index] < counts[index]:
            quotas[index] += 1
            remaining -= 1
    return quotas


def _sample_indices(count: int, quota: int, seed: int, identity: str) -> np.ndarray:
    if quota >= count:
        return np.arange(count, dtype=np.int64)
    stable_seed = (seed + zlib.crc32(identity.encode("utf-8"))) % (2**32)
    rng = np.random.default_rng(stable_seed)
    return np.sort(rng.choice(count, size=quota, replace=False)).astype(np.int64)


def _read_h5_samples(dataset: h5py.Dataset, indices: np.ndarray, sample_axis: int) -> np.ndarray:
    if sample_axis < 0:
        sample_axis += dataset.ndim
    if indices.size == 0:
        shape = list(dataset.shape)
        shape[sample_axis] = 0
        return np.empty(shape, dtype=dataset.dtype)
    contiguous = bool(np.array_equal(indices, np.arange(indices[0], indices[0] + len(indices))))
    selector: slice | np.ndarray = slice(int(indices[0]), int(indices[-1]) + 1) if contiguous else indices
    if sample_axis == 0:
        return np.asarray(dataset[selector])
    if sample_axis == dataset.ndim - 1:
        return np.moveaxis(np.asarray(dataset[..., selector]), -1, 0)
    raise ValueError(f"unsupported HDF5 sample axis {sample_axis} for shape {dataset.shape}")


def _profile_mat_split(
    paths: list[Path],
    specs: list[QuantitySpec],
    population_size: int,
    split: str,
) -> tuple[dict[str, dict[str, float | int]], dict[str, str]]:
    accumulators = {spec.name: MomentAccumulator() for spec in specs}
    specs_by_key: dict[str, list[QuantitySpec]] = {}
    for spec in specs:
        specs_by_key.setdefault(spec.source_key, []).append(spec)
    remaining = int(population_size)
    for path in paths:
        if remaining <= 0:
            break
        for source_key, key_specs in specs_by_key.items():
            print(f"[{split}] MAT {path.name}:{source_key}", flush=True)
            raw = scipy.io.loadmat(path, variable_names=[source_key])[source_key]
            take = min(remaining, int(raw.shape[0]))
            selected = raw[:take]
            for spec in key_specs:
                accumulators[spec.name].update(spec.transform(selected))
            del selected, raw
            gc.collect()
        first_key = next(iter(specs_by_key))
        count = next(iter(accumulators.values())).sample_count
        prior_count = population_size - remaining
        file_take = min(population_size - prior_count, count - prior_count)
        remaining -= file_take
    if remaining > 0:
        raise ValueError(f"{split}: requested {population_size} MAT samples but {remaining} are missing")
    return (
        {name: accumulator.summary() for name, accumulator in accumulators.items()},
        {"mode": "exact", "population_samples": str(population_size)},
    )


def _profile_h5_split(
    paths: list[Path],
    specs: list[QuantitySpec],
    population_size: int,
    sample_size: int,
    split: str,
    seed: int,
    sample_axis: int,
    io_batch_size: int,
) -> tuple[dict[str, dict[str, float | int]], dict[str, str]]:
    accumulators = {spec.name: MomentAccumulator() for spec in specs}
    specs_by_key: dict[str, list[QuantitySpec]] = {}
    for spec in specs:
        specs_by_key.setdefault(spec.source_key, []).append(spec)
    reference_key = specs[0].source_key
    counts = _bounded_counts(_available_counts(paths, reference_key, sample_axis), population_size)
    quotas = _allocate_sample_quotas(counts, sample_size)
    for path, count, quota in zip(paths, counts, quotas):
        if count <= 0 or quota <= 0:
            continue
        indices = _sample_indices(count, quota, seed, f"{split}:{path}")
        with h5py.File(path, "r") as handle:
            for source_key, key_specs in specs_by_key.items():
                print(f"[{split}] H5 {path.name}:{source_key} ({quota}/{count} samples)", flush=True)
                for start in range(0, len(indices), io_batch_size):
                    batch_indices = indices[start : start + io_batch_size]
                    selected = _read_h5_samples(handle[source_key], batch_indices, sample_axis)
                    for spec in key_specs:
                        accumulators[spec.name].update(spec.transform(selected))
                    del selected
                gc.collect()
    observed = sum(quotas)
    mode = "exact" if observed == population_size else "deterministic_stratified_sample"
    return (
        {name: accumulator.summary() for name, accumulator in accumulators.items()},
        {"mode": mode, "population_samples": str(population_size), "observed_samples": str(observed)},
    )


def _severity(mean_shift: float, std_ratio: float) -> str:
    symmetric_std_ratio = max(std_ratio, 1.0 / max(std_ratio, 1e-12))
    if mean_shift >= 0.5 or symmetric_std_ratio >= 1.5:
        return "large"
    if mean_shift >= 0.25 or symmetric_std_ratio >= 1.25:
        return "moderate"
    if mean_shift >= 0.1 or symmetric_std_ratio >= 1.1:
        return "small"
    return "negligible"


def _comparison_row(
    pde: str,
    quantity: str,
    train: dict[str, float | int],
    test: dict[str, float | int],
    train_meta: dict[str, str],
    test_meta: dict[str, str],
) -> dict[str, object]:
    train_std = float(train["std"])
    test_std = float(test["std"])
    train_variance = float(train["variance"])
    test_variance = float(test["variance"])
    mean_delta = float(test["mean"]) - float(train["mean"])
    mean_shift = abs(mean_delta) / max(train_std, 1e-12)
    std_ratio = test_std / max(train_std, 1e-12)
    variance_ratio = test_variance / max(train_variance, 1e-24)
    rms_ratio = float(test["sample_rms_mean"]) / max(float(train["sample_rms_mean"]), 1e-12)
    return {
        "pde": pde,
        "quantity": quantity,
        "train_sampling": train_meta["mode"],
        "test_sampling": test_meta["mode"],
        "train_population_samples": int(train_meta["population_samples"]),
        "test_population_samples": int(test_meta["population_samples"]),
        "train_samples_observed": int(train["samples_observed"]),
        "test_samples_observed": int(test["samples_observed"]),
        "train_mean": float(train["mean"]),
        "test_mean": float(test["mean"]),
        "mean_delta": mean_delta,
        "standardized_mean_shift": mean_shift,
        "train_variance": train_variance,
        "test_variance": test_variance,
        "variance_ratio_test_over_train": variance_ratio,
        "train_std": train_std,
        "test_std": test_std,
        "std_ratio_test_over_train": std_ratio,
        "train_sample_rms_mean": float(train["sample_rms_mean"]),
        "test_sample_rms_mean": float(test["sample_rms_mean"]),
        "sample_rms_ratio_test_over_train": rms_ratio,
        "train_sample_rms_p05": float(train["sample_rms_p05"]),
        "train_sample_rms_median": float(train["sample_rms_median"]),
        "train_sample_rms_p95": float(train["sample_rms_p95"]),
        "test_sample_rms_p05": float(test["sample_rms_p05"]),
        "test_sample_rms_median": float(test["sample_rms_median"]),
        "test_sample_rms_p95": float(test["sample_rms_p95"]),
        "train_min": float(train["min"]),
        "train_max": float(train["max"]),
        "test_min": float(test["min"]),
        "test_max": float(test["max"]),
        "shift_severity": _severity(mean_shift, std_ratio),
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    file_config = _load_file_config(args.data_files_config)
    rows: list[dict[str, object]] = []
    details: dict[str, object] = {
        "data_root": str(data_root),
        "data_files_config": str(args.data_files_config.resolve()),
        "train_fit_size": args.train_fit_size,
        "test_size": args.test_size,
        "h5_train_samples": args.h5_train_samples,
        "h5_test_samples": args.h5_test_samples,
        "h5_io_batch_size": args.h5_io_batch_size,
        "seed": args.seed,
        "pdes": {},
    }
    for pde in args.pdes:
        specs = QUANTITIES[pde]
        print(f"\n=== {pde} ===", flush=True)
        configured = file_config[pde]
        train_paths = [data_root / item for item in configured["train"]]
        test_paths = [data_root / item for item in configured["test"]]
        if pde in {"darcy", "nsnonbounded"}:
            sample_axis = -1 if pde == "darcy" else 0
            train_stats, train_meta = _profile_h5_split(
                train_paths,
                specs,
                args.train_fit_size,
                args.h5_train_samples,
                "train",
                args.seed,
                sample_axis,
                args.h5_io_batch_size,
            )
            test_stats, test_meta = _profile_h5_split(
                test_paths,
                specs,
                args.test_size,
                args.h5_test_samples,
                "test",
                args.seed,
                sample_axis,
                args.h5_io_batch_size,
            )
        else:
            train_stats, train_meta = _profile_mat_split(train_paths, specs, args.train_fit_size, "train")
            test_stats, test_meta = _profile_mat_split(test_paths, specs, args.test_size, "test")
        for spec in specs:
            rows.append(_comparison_row(pde, spec.name, train_stats[spec.name], test_stats[spec.name], train_meta, test_meta))
        details["pdes"][pde] = {
            "train_files": [str(path) for path in train_paths],
            "test_files": [str(path) for path in test_paths],
            "train_sampling": train_meta,
            "test_sampling": test_meta,
            "train": train_stats,
            "test": test_stats,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "train_test_distribution.csv"
    json_path = args.output_dir / "train_test_distribution.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"metadata": details, "comparisons": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "rows": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
