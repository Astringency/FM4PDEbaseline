from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import scipy.io

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def tiny_data_root(tmp_path: Path) -> Path:
    n, s = 3, 8
    rng = np.random.default_rng(1)

    def mkdir(name):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    d = mkdir("darcy")
    with h5py.File(d / "darcy_10000-128-128_1.mat", "w") as f:
        f["thresh_a_data"] = rng.normal(size=(s, s, n)).astype("float32")
        f["thresh_p_data"] = rng.normal(size=(s, s, n)).astype("float32")

    p = mkdir("poisson")
    scipy.io.savemat(p / "poisson_10000-128-128_1.mat", {"f_data": rng.normal(size=(n, s, s)), "phi_data": rng.normal(size=(n, s, s))})

    h = mkdir("helmholtz")
    scipy.io.savemat(h / "helmholtz_10000-128-128_1.mat", {"f_data": rng.normal(size=(n, s, s)), "psi_data": rng.normal(size=(n, s, s))})

    b = mkdir("burgers")
    scipy.io.savemat(b / "burger_10000-128-128_1.mat", {"input": rng.normal(size=(n, s)), "output": rng.normal(size=(n, s, s))})

    ns = mkdir("nsnonbounded")
    with h5py.File(ns / "nsnonbounded_10000-128-128-10_1_new.mat", "w") as f:
        f["w0"] = rng.normal(size=(n, s, s)).astype("float32")
        f["w"] = rng.normal(size=(n, s, s, 10)).astype("float32")

    rd = mkdir("reaction_diffusion")
    with h5py.File(rd / "reaction_diffusion-128-128-10_0.h5", "w") as f:
        for i in range(n):
            g = f.create_group(str(i))
            g["data"] = rng.normal(size=(10, s, s, 2)).astype("float32")

    swe = mkdir("shallow_water")
    with h5py.File(swe / "2d_swe_128_128_10_0.h5", "w") as f:
        for i in range(n):
            g = f.create_group(str(i)).create_group("data")
            for key in ("h", "hu", "hv"):
                g[key] = rng.normal(size=(11, s, s, 1)).astype("float32")

    return tmp_path
