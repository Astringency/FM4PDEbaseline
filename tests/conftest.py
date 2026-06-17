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
    with h5py.File(d / "darcy_test_1000-128-128.mat", "w") as f:
        f["thresh_a_data"] = rng.normal(size=(s, s, n)).astype("float32")
        f["thresh_p_data"] = rng.normal(size=(s, s, n)).astype("float32")

    p = mkdir("poisson")
    scipy.io.savemat(p / "poisson_10000-128-128_1.mat", {"f_data": rng.normal(size=(n, s, s)), "phi_data": rng.normal(size=(n, s, s))})
    scipy.io.savemat(p / "poisson_test_1000-128-128.mat", {"f_data": rng.normal(size=(n, s, s)), "phi_data": rng.normal(size=(n, s, s))})

    h = mkdir("helmholtz")
    scipy.io.savemat(h / "helmholtz_10000-128-128_1.mat", {"f_data": rng.normal(size=(n, s, s)), "psi_data": rng.normal(size=(n, s, s))})

    b = mkdir("burgers")
    scipy.io.savemat(b / "burger_10000-128-128_1.mat", {"input": rng.normal(size=(n, s)), "output": rng.normal(size=(n, s, s))})

    ns = mkdir("nsnonbounded")
    with h5py.File(ns / "nsnonbounded_10000-128-128-10_1_new.mat", "w") as f:
        f["w0"] = rng.normal(size=(n, s, s)).astype("float32")
        f["w"] = rng.normal(size=(n, s, s, 10)).astype("float32")
    with h5py.File(ns / "nsnonbounded_test_1000-128-128-10.mat", "w") as f:
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

    heat = mkdir("heat")
    with h5py.File(heat / "heat_10000-128-128_1.h5", "w") as f:
        f["input_data"] = rng.normal(size=(n, 1, s, s)).astype("float32")
        f["output_data"] = rng.normal(size=(n, 1, s, s)).astype("float32")
        f["alpha"] = np.full(n, 1e-3, dtype="float32")
        f["full_trajectory"] = rng.normal(size=(n, 1, 3, s, s)).astype("float32")
        f["t"] = np.linspace(0, 1, 3).astype("float32")
        f.attrs["T"] = 1.0
        f.attrs["boundary_condition"] = "periodic"
    with h5py.File(heat / "heat_test_1000-128-128.h5", "w") as f:
        f["input_data"] = rng.normal(size=(n, 1, s, s)).astype("float32")
        f["output_data"] = rng.normal(size=(n, 1, s, s)).astype("float32")
        f["alpha"] = np.full(n, 2e-3, dtype="float32")
        f.attrs["T"] = 1.0
        f.attrs["boundary_condition"] = "periodic"

    wave = mkdir("wave")
    with h5py.File(wave / "wave_10000-128-128_1.h5", "w") as f:
        f["input_data"] = rng.normal(size=(n, 2, s, s)).astype("float32")
        f["output_data"] = rng.normal(size=(n, 2, s, s)).astype("float32")
        f["t"] = np.linspace(0, 1, 3).astype("float32")
        f.attrs["T"] = 1.0
        f.attrs["fixed_c"] = 1.0
        f.attrs["boundary_condition"] = "periodic"

    adv = mkdir("advection_diffusion")
    with h5py.File(adv / "advection_diffusion_10000-128-128_1.h5", "w") as f:
        f["input_data"] = rng.normal(size=(n, 1, s, s)).astype("float32")
        f["output_data"] = rng.normal(size=(n, 1, s, s)).astype("float32")
        f["b_x"] = np.full(n, 0.1, dtype="float32")
        f["b_y"] = np.full(n, -0.2, dtype="float32")
        f["kappa"] = np.full(n, 1e-3, dtype="float32")
        f["t"] = np.linspace(0, 1, 3).astype("float32")
        f.attrs["T"] = 1.0
        f.attrs["boundary_condition"] = "periodic"

    shc = mkdir("steady_heat_conduction")
    with h5py.File(shc / "steady_heat_conduction_10000-128-128_1.h5", "w") as f:
        f["input_data"] = rng.normal(size=(n, 1, s, s)).astype("float32")
        f["output_data"] = rng.normal(size=(n, 1, s, s)).astype("float32")
        f["u_D"] = np.full(n, 298.0, dtype="float32")
        f["source_x"] = rng.normal(size=(n, 3)).astype("float32")
        f["source_y"] = rng.normal(size=(n, 3)).astype("float32")
        f["source_amp"] = rng.normal(size=(n, 3)).astype("float32")
        f["source_sigma"] = rng.random(size=(n, 3)).astype("float32")

    return tmp_path
