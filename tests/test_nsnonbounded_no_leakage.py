from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from baselines.common.data_adapter import build_default_registry


def test_nsnonbounded_test_split_selects_test_file(tiny_data_root):
    registry = build_default_registry()
    raw = registry.load_raw("nsnonbounded", tiny_data_root, split="test", max_samples=1)
    basenames = [Path(p).name for p in raw["file_paths"]]
    assert basenames
    assert all("test" in name.lower() or name.lower().startswith("nsnonbounded_1000-128-128-10") for name in basenames)
    assert all("_new" not in name.lower() for name in basenames)


def test_nsnonbounded_test_split_does_not_fallback_to_train_shard(tmp_path: Path):
    ns = tmp_path / "nsnonbounded"
    ns.mkdir()
    with h5py.File(ns / "nsnonbounded_10000-128-128-10_1_new.mat", "w") as f:
        f["w0"] = np.zeros((2, 8, 8), dtype="float32")
        f["w"] = np.zeros((2, 8, 8, 10), dtype="float32")
    registry = build_default_registry()
    with pytest.raises(FileNotFoundError):
        registry.load_raw("nsnonbounded", tmp_path, split="test", max_samples=1)
