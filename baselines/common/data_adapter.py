from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import h5py
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

from .sensors import build_observation_tensors, make_coordinate_grid


@dataclass
class PDEBatch:
    pde_name: str
    task: str
    full_tensor: torch.Tensor
    input_fields: torch.Tensor
    target_fields: torch.Tensor
    coords: torch.Tensor | None
    mask: torch.Tensor | None
    obs_values: torch.Tensor | None
    obs_coords: torch.Tensor | None
    channel_names: list[str]
    metadata: dict


@dataclass
class PDESpec:
    name: str
    channel_names: list[str]
    input_indices: list[int]
    target_indices: list[int]
    loader: Callable[..., dict[str, Any]]
    aliases: tuple[str, ...] = ()
    time_dependent: bool = False
    supports_trajectory: bool = False
    notes: str = ""


def _as_float_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=torch.float32)


def _take_np(array: np.ndarray, max_samples: int | None) -> np.ndarray:
    if max_samples is None:
        return array
    return array[:max_samples]


def _h5_samples(ds: h5py.Dataset, n: int) -> np.ndarray:
    """Read first ``n`` samples from HDF5 datasets with sample dim first or last."""
    shape = ds.shape
    if len(shape) < 3:
        return ds[:n]
    if shape[0] >= n and shape[1] == shape[2]:
        # [N,H,W] or [N,H,W,T]
        return ds[:n]
    if shape[-1] >= n and shape[0] == shape[1]:
        # MATLAB v7.3 convention used by Darcy: [H,W,N]
        return np.moveaxis(ds[..., :n], -1, 0)
    return ds[:n]


def _cat_or_empty(parts: list[torch.Tensor], shape: tuple[int, ...]) -> torch.Tensor:
    if parts:
        return torch.cat(parts, dim=0)
    return torch.empty(shape, dtype=torch.float32)


def _safe_loadmat(path: Path, keys: Iterable[str]) -> dict[str, np.ndarray]:
    raw = scipy.io.loadmat(path, variable_names=list(keys))
    return {k: raw[k] for k in keys if k in raw}


class PDEDataRegistry:
    """Registry/adapter layer converting FM4PDE raw files to task batches.

    Canonical conventions:
      * Static 2D fields: ``full_tensor`` is ``[N, C, H, W]``.
      * Time-dependent 2D trajectories: ``full_tensor`` is ``[N, C, T, H, W]``.
      * Burgers stores an x-t field as ``[N, 1, T, X]`` and marks axes in metadata.

    The task tensors exposed to baseline models are channel-first 2D grids when
    possible, with trajectory time flattened into channels for amortized 2D
    baselines. The original trajectory remains in ``full_tensor`` and metadata.
    """

    def __init__(self) -> None:
        self._specs: dict[str, PDESpec] = {}
        self._register_defaults()

    def register(self, spec: PDESpec) -> None:
        names = (spec.name, *spec.aliases)
        for name in names:
            self._specs[name.lower()] = spec

    def available(self) -> list[str]:
        return sorted({spec.name for spec in self._specs.values()})

    def get(self, pde_name: str) -> PDESpec:
        key = pde_name.lower()
        if key not in self._specs:
            raise KeyError(f"Unknown PDE '{pde_name}'. Known PDEs: {self.available()}")
        return self._specs[key]

    def load_raw(
        self,
        pde_name: str,
        data_root: str | Path,
        split: str = "train",
        max_samples: int | None = None,
        prefer_test: bool = False,
        synthetic_if_missing: bool = False,
        synthetic_resolution: int = 32,
    ) -> dict[str, Any]:
        spec = self.get(pde_name)
        root = Path(data_root)
        try:
            return spec.loader(root, split=split, max_samples=max_samples, prefer_test=prefer_test)
        except FileNotFoundError as exc:
            if synthetic_if_missing:
                warnings.warn(
                    f"{spec.name}: {exc}. Falling back to synthetic smoke data.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self.synthetic_raw(spec.name, max_samples or 8, synthetic_resolution)
            raise

    def to_canonical(self, raw: dict[str, Any], pde_name: str) -> dict[str, Any]:
        spec = self.get(pde_name)
        return {
            "pde_name": spec.name,
            "full_tensor": raw["full_tensor"].float(),
            "channel_names": list(raw.get("channel_names", spec.channel_names)),
            "metadata": dict(raw.get("metadata", {})),
        }

    def make_task(
        self,
        raw_or_canonical: dict[str, Any],
        pde_name: str,
        task: str,
        num_sensors: int | None = None,
        sensor_mode: str = "random",
        noise_level: float = 0.0,
        seed: int = 0,
    ) -> PDEBatch:
        spec = self.get(pde_name)
        canonical = (
            raw_or_canonical
            if "full_tensor" in raw_or_canonical and "metadata" in raw_or_canonical
            else self.to_canonical(raw_or_canonical, pde_name)
        )
        full = canonical["full_tensor"].float()
        metadata = dict(canonical.get("metadata", {}))
        metadata.setdefault("time_dependent", spec.time_dependent)
        metadata.setdefault("supports_trajectory", spec.supports_trajectory)

        input_fields, target_fields, task_channel_names = self._split_task(full, spec, task, metadata)
        coords = make_coordinate_grid(tuple(target_fields.shape[2:]), batch_size=target_fields.shape[0])

        mask = obs_values = obs_coords = None
        if task.startswith("sparse") or num_sensors:
            obs = build_observation_tensors(
                target_fields,
                num_sensors=num_sensors or 500,
                mode=sensor_mode,
                seed=seed,
                noise_level=noise_level,
            )
            mask = obs["mask"]
            obs_values = obs["obs_values"]
            obs_coords = obs["obs_coords"]
            metadata.update(
                {
                    "masked_grid": obs["masked_grid"],
                    "voronoi_grid": obs["voronoi_grid"],
                    "sensor_mode": sensor_mode,
                    "num_sensors": int(num_sensors or 500),
                    "noise_level": float(noise_level),
                    "mask_id": obs["mask_id"],
                }
            )
            if task in {"sparse_solution", "sparse_reconstruction"}:
                input_fields = obs["masked_grid"]
            elif task == "sparse_inverse":
                input_fields = obs["masked_grid"]

        metadata["input_shape"] = tuple(input_fields.shape)
        metadata["target_shape"] = tuple(target_fields.shape)
        metadata["full_shape"] = tuple(full.shape)
        metadata["task_channel_names"] = task_channel_names

        return PDEBatch(
            pde_name=spec.name,
            task=task,
            full_tensor=full,
            input_fields=input_fields.float(),
            target_fields=target_fields.float(),
            coords=coords,
            mask=mask,
            obs_values=obs_values,
            obs_coords=obs_coords,
            channel_names=canonical.get("channel_names", spec.channel_names),
            metadata=metadata,
        )

    def make_dataset(
        self,
        pde_name: str,
        data_root: str | Path,
        task: str,
        split: str = "train",
        max_samples: int | None = None,
        num_sensors: int | None = None,
        sensor_mode: str = "random",
        noise_level: float = 0.0,
        seed: int = 0,
        prefer_test: bool = False,
        synthetic_if_missing: bool = False,
        synthetic_resolution: int = 32,
    ) -> "PDEBatchDataset":
        raw = self.load_raw(
            pde_name,
            data_root,
            split=split,
            max_samples=max_samples,
            prefer_test=prefer_test,
            synthetic_if_missing=synthetic_if_missing,
            synthetic_resolution=synthetic_resolution,
        )
        batch = self.make_task(
            self.to_canonical(raw, pde_name),
            pde_name,
            task,
            num_sensors=num_sensors,
            sensor_mode=sensor_mode,
            noise_level=noise_level,
            seed=seed,
        )
        return PDEBatchDataset(batch)

    def synthetic_raw(self, pde_name: str, n: int = 8, resolution: int = 32) -> dict[str, Any]:
        pde_name = pde_name.lower()
        gen = torch.Generator().manual_seed(17)
        h = w = resolution
        if pde_name in {"darcy", "poisson", "helmholtz"}:
            c = 2
            x = torch.randn(n, c, h, w, generator=gen)
            channels = self.get(pde_name).channel_names
            meta = {"source": "synthetic", "canonical_layout": "NCHW"}
        elif pde_name == "heat":
            u0 = torch.randn(n, 1, h, w, generator=gen)
            uT = torch.randn(n, 1, h, w, generator=gen)
            alpha = torch.full((n, 1, h, w), 1e-3)
            x = torch.cat([u0, alpha, uT, alpha], dim=1)
            channels = self.get(pde_name).channel_names
            meta = {"source": "synthetic", "canonical_layout": "NCHW", "alpha": torch.full((n,), 1e-3), "final_time": 1.0, "bc": "periodic"}
        elif pde_name == "wave":
            u0 = torch.randn(n, 1, h, w, generator=gen)
            v0 = torch.zeros(n, 1, h, w)
            uT = torch.randn(n, 1, h, w, generator=gen)
            vT = torch.randn(n, 1, h, w, generator=gen)
            x = torch.cat([u0, v0, uT, vT], dim=1)
            channels = self.get(pde_name).channel_names
            meta = {"source": "synthetic", "canonical_layout": "NCHW", "fixed_c": 1.0, "final_time": 1.0, "bc": "periodic"}
        elif pde_name == "advection_diffusion":
            u0 = torch.randn(n, 1, h, w, generator=gen)
            uT = torch.randn(n, 1, h, w, generator=gen)
            bx = torch.full((n, 1, h, w), 0.25)
            by = torch.full((n, 1, h, w), -0.15)
            kappa = torch.full((n, 1, h, w), 1e-3)
            x = torch.cat([u0, bx, by, kappa, uT, bx, by, kappa], dim=1)
            channels = self.get(pde_name).channel_names
            meta = {
                "source": "synthetic",
                "canonical_layout": "NCHW",
                "b_x": torch.full((n,), 0.25),
                "b_y": torch.full((n,), -0.15),
                "kappa": torch.full((n,), 1e-3),
                "final_time": 1.0,
                "bc": "periodic",
            }
        elif pde_name in {"steady_heat_conduction", "steady_heat"}:
            source = torch.randn(n, 1, h, w, generator=gen)
            u_d = torch.full((n, 1, h, w), 298.0)
            solution = 298.0 + torch.randn(n, 1, h, w, generator=gen) * 0.01
            x = torch.cat([source, u_d, solution, u_d], dim=1)
            channels = self.get(pde_name).channel_names
            meta = {"source": "synthetic", "canonical_layout": "NCHW", "u_D": torch.full((n,), 298.0)}
        elif pde_name == "nsnonbounded":
            x = torch.randn(n, 1, 11, h, w, generator=gen)
            channels = ["w"]
            meta = {"source": "synthetic", "canonical_layout": "NCTHW", "time_values": [i / 10 for i in range(11)], "final_time": 1.0, "nu": 1e-3}
        elif pde_name == "burger":
            x = torch.randn(n, 1, h, w, generator=gen)
            channels = ["u"]
            meta = {
                "source": "synthetic",
                "canonical_layout": "NCTX",
                "axes": ["time", "x"],
                "time_values": [i / max(h - 1, 1) for i in range(h)],
                "final_time": 1.0,
                "nu": 0.01,
            }
        elif pde_name == "reaction_diffusion":
            x = torch.randn(n, 2, 10, h, w, generator=gen)
            channels = ["u", "v"]
            meta = {"source": "synthetic", "canonical_layout": "NCTHW", "input_time_index": 0, "final_time": 5.0, "D_u": 1e-3, "D_v": 5e-3, "k": 5e-3}
        elif pde_name == "shallow_water":
            x = torch.randn(n, 3, 11, h, w, generator=gen)
            channels = ["h", "hu", "hv"]
            x[:, 0] = x[:, 0].abs() + 1.0
            meta = {"source": "synthetic", "canonical_layout": "NCTHW", "final_time": 1.0, "g": 1.0, "domain_length": 5.0}
        else:
            x = torch.randn(n, 2, h, w, generator=gen)
            channels = ["input", "target"]
            meta = {"source": "synthetic", "canonical_layout": "NCHW"}
        return {"full_tensor": x, "channel_names": channels, "metadata": meta}

    def _split_task(
        self, full: torch.Tensor, spec: PDESpec, task: str, metadata: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        name = spec.name
        if name == "burger":
            target = full
            initial = metadata.get("initial_1d")
            if initial is None:
                input_fields = full[:, :, :1, :].repeat(1, 1, full.shape[-2], 1)
            else:
                init = initial.to(full.device, full.dtype).reshape(full.shape[0], 1, 1, full.shape[-1])
                input_fields = init.repeat(1, 1, full.shape[-2], 1)
            if task in {"inverse", "sparse_inverse"}:
                return target, input_fields[:, :, :1, :], ["u0"]
            return input_fields, target, ["u"]

        if full.ndim == 5:
            if name == "nsnonbounded":
                inp = full[:, :, 0]
                target = full[:, :, 1:].reshape(full.shape[0], -1, full.shape[-2], full.shape[-1])
                target_names = [f"w_t{i}" for i in range(1, full.shape[2])]
            elif name in {"reaction_diffusion", "shallow_water"}:
                input_idx = int(metadata.get("input_time_index", 0))
                inp = full[:, :, input_idx]
                target = full[:, :, -1]
                suffix = "T"
                target_names = [f"{c}{suffix}" for c in spec.channel_names]
            else:
                inp = full[:, :, 0]
                target = full[:, :, -1]
                target_names = [f"{c}_target" for c in spec.channel_names]
        else:
            inp = full[:, spec.input_indices]
            target = full[:, spec.target_indices]
            target_names = [spec.channel_names[i] for i in spec.target_indices]

        if task in {"inverse", "sparse_inverse"}:
            return target, inp, [spec.channel_names[i] for i in spec.input_indices]
        if task in {"both", "joint"}:
            return full, full, list(spec.channel_names)
        return inp, target, target_names

    def _register_defaults(self) -> None:
        self.register(
            PDESpec("darcy", ["a", "p"], [0], [1], _load_darcy, aliases=("darcy_flow",))
        )
        self.register(PDESpec("poisson", ["f", "phi"], [0], [1], _load_poisson))
        self.register(PDESpec("helmholtz", ["f", "psi"], [0], [1], _load_helmholtz))
        self.register(
            PDESpec(
                "nsnonbounded",
                ["w"],
                [0],
                list(range(1, 11)),
                _load_nsnonbounded,
                aliases=("navier_stokes", "ns"),
                time_dependent=True,
                supports_trajectory=True,
                notes="Canonical full tensor keeps trajectory as [N,1,11,H,W].",
            )
        )
        self.register(
            PDESpec(
                "burger",
                ["u"],
                [0],
                [0],
                _load_burger,
                aliases=("burgers",),
                time_dependent=True,
                supports_trajectory=True,
                notes="Burgers output is stored as an x-t image; initial 1D field is metadata.",
            )
        )
        self.register(
            PDESpec(
                "reaction_diffusion",
                ["u", "v"],
                [0, 1],
                [2, 3],
                _load_reaction_diffusion,
                aliases=("rd",),
                time_dependent=True,
                supports_trajectory=True,
            )
        )
        self.register(
            PDESpec(
                "shallow_water",
                ["h", "hu", "hv"],
                [0, 1, 2],
                [3, 4, 5],
                _load_shallow_water,
                aliases=("swe",),
                time_dependent=True,
                supports_trajectory=True,
            )
        )
        self.register(
            PDESpec(
                "heat",
                ["u0", "alpha", "uT", "alpha_T"],
                [0, 1],
                [2, 3],
                _load_heat,
                time_dependent=True,
                notes="Random/fixed-alpha HDF5 layout materialized as [u0, alpha, uT, alpha].",
            )
        )
        self.register(
            PDESpec(
                "wave",
                ["u0", "v0", "uT", "vT"],
                [0, 1],
                [2, 3],
                _load_wave,
                time_dependent=True,
                notes="Fixed-c HDF5 layout is [u0, v0, uT, vT].",
            )
        )
        self.register(
            PDESpec(
                "advection_diffusion",
                ["u0", "b_x", "b_y", "kappa", "uT", "b_x_T", "b_y_T", "kappa_T"],
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                _load_advection_diffusion,
                aliases=("advdiff",),
                time_dependent=True,
                notes="HDF5 layout materialized as [u0, b_x, b_y, kappa, uT, b_x, b_y, kappa].",
            )
        )
        self.register(
            PDESpec(
                "steady_heat_conduction",
                ["f", "u_D", "u", "u_D_T"],
                [0, 1],
                [2, 3],
                _load_steady_heat_conduction,
                aliases=("steady_heat", "nonlinear_heat_conduction"),
                notes="HDF5 layout materialized as [f, u_D, u, u_D].",
            )
        )


class PDEBatchDataset(Dataset):
    def __init__(self, batch: PDEBatch) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return int(self.batch.input_fields.shape[0])

    def __getitem__(self, index: int) -> PDEBatch:
        return slice_pde_batch(self.batch, index)


def slice_pde_batch(batch: PDEBatch, index: int) -> PDEBatch:
    sl = slice(index, index + 1)
    metadata = dict(batch.metadata)
    for key, value in list(metadata.items()):
        if isinstance(value, torch.Tensor) and value.shape[:1] == batch.full_tensor.shape[:1]:
            metadata[key] = value[sl]
    return PDEBatch(
        pde_name=batch.pde_name,
        task=batch.task,
        full_tensor=batch.full_tensor[sl],
        input_fields=batch.input_fields[sl],
        target_fields=batch.target_fields[sl],
        coords=batch.coords[sl] if batch.coords is not None and batch.coords.shape[0] == len(batch.full_tensor) else batch.coords,
        mask=batch.mask,
        obs_values=batch.obs_values[sl] if batch.obs_values is not None else None,
        obs_coords=batch.obs_coords[sl] if batch.obs_coords is not None else None,
        channel_names=batch.channel_names,
        metadata=metadata,
    )


def pde_collate(items: list[PDEBatch]) -> PDEBatch:
    if len(items) == 1:
        return items[0]
    first = items[0]
    coords = None
    if first.coords is not None:
        if first.coords.shape[0] == 1:
            coords = torch.cat([b.coords for b in items if b.coords is not None], dim=0)
        else:
            coords = first.coords
    metadata = dict(first.metadata)
    for key, value in list(metadata.items()):
        if isinstance(value, torch.Tensor) and value.shape[:1] == first.full_tensor.shape[:1]:
            metadata[key] = torch.cat([b.metadata[key] for b in items], dim=0)
    return PDEBatch(
        pde_name=first.pde_name,
        task=first.task,
        full_tensor=torch.cat([b.full_tensor for b in items], dim=0),
        input_fields=torch.cat([b.input_fields for b in items], dim=0),
        target_fields=torch.cat([b.target_fields for b in items], dim=0),
        coords=coords,
        mask=first.mask,
        obs_values=torch.cat([b.obs_values for b in items], dim=0) if first.obs_values is not None else None,
        obs_coords=torch.cat([b.obs_coords for b in items], dim=0) if first.obs_coords is not None else None,
        channel_names=first.channel_names,
        metadata=metadata,
    )


def _candidate_files(root: Path, pde: str, split: str, patterns: list[str], aliases: tuple[str, ...] = ()) -> list[Path]:
    dirs = [root / pde, *(root / a for a in aliases)]
    if split == "test":
        dirs.insert(0, root / "test1125")
    found: list[Path] = []
    for base in dirs:
        for pat in patterns:
            found.extend(sorted(base.glob(pat)))
    return [p for p in found if p.exists()]


def _load_darcy(root: Path, split: str, max_samples: int | None, prefer_test: bool = False) -> dict[str, Any]:
    patterns = ["darcy_test_1000-128-128.mat", "darcy_1000-128-128_test.mat"] if split == "test" or prefer_test else ["darcy_10000-128-128_*.mat"]
    files = _candidate_files(root, "darcy", split if not prefer_test else "test", patterns)
    if not files:
        raise FileNotFoundError("Darcy files not found")
    parts = []
    remaining = max_samples
    for path in files:
        with h5py.File(path, "r") as f:
            ds = f["thresh_a_data"]
            n_total = ds.shape[-1] if len(ds.shape) >= 3 and ds.shape[0] == ds.shape[1] else ds.shape[0]
            n = n_total if remaining is None else min(remaining, n_total)
            a = _h5_samples(f["thresh_a_data"], n)
            p = _h5_samples(f["thresh_p_data"], n)
        parts.append(torch.stack((_as_float_tensor(a), _as_float_tensor(p)), dim=1))
        if remaining is not None:
            remaining -= n
            if remaining <= 0:
                break
    return {"full_tensor": torch.cat(parts, dim=0), "channel_names": ["a", "p"], "metadata": {"files": [str(p) for p in files], "canonical_layout": "NCHW"}}


def _load_poisson(root: Path, split: str, max_samples: int | None, prefer_test: bool = False) -> dict[str, Any]:
    return _load_static_mat(
        root,
        "poisson",
        ("f_data", "phi_data"),
        ["poisson_test_1000-128-128.mat", "poisson_1000-128-128_test.mat"] if split == "test" or prefer_test else ["poisson_10000-128-128_*.mat"],
        split if not prefer_test else "test",
        max_samples,
        ["f", "phi"],
    )


def _load_helmholtz(root: Path, split: str, max_samples: int | None, prefer_test: bool = False) -> dict[str, Any]:
    return _load_static_mat(
        root,
        "helmholtz",
        ("f_data", "psi_data"),
        ["helmholtz_test_1000-128-128-k1.mat", "helmholtz_1000-128-128_test.mat"] if split == "test" or prefer_test else ["helmholtz_10000-128-128_*.mat"],
        split if not prefer_test else "test",
        max_samples,
        ["f", "psi"],
        metadata={"k": 1.0},
    )


def _load_static_mat(
    root: Path,
    pde: str,
    keys: tuple[str, str],
    patterns: list[str],
    split: str,
    max_samples: int | None,
    channel_names: list[str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    files = _candidate_files(root, pde, split, patterns)
    if not files:
        raise FileNotFoundError(f"{pde} files not found")
    parts = []
    remaining = max_samples
    for path in files:
        raw = _safe_loadmat(path, keys)
        n_total = raw[keys[0]].shape[0]
        n = n_total if remaining is None else min(remaining, n_total)
        x = _as_float_tensor(_take_np(raw[keys[0]], n))
        y = _as_float_tensor(_take_np(raw[keys[1]], n))
        parts.append(torch.stack((x, y), dim=1))
        if remaining is not None:
            remaining -= n
            if remaining <= 0:
                break
    meta = {"files": [str(p) for p in files], "canonical_layout": "NCHW"}
    if metadata:
        meta.update(metadata)
    return {"full_tensor": torch.cat(parts, dim=0), "channel_names": channel_names, "metadata": meta}


def _load_nsnonbounded(root: Path, split: str, max_samples: int | None, prefer_test: bool = False) -> dict[str, Any]:
    patterns = ["nsnonbounded_1000-128-128-10_1.mat"] if split == "test" or prefer_test else ["nsnonbounded_10000-128-128-10_*_new.mat"]
    files = _candidate_files(root, "nsnonbounded", split if not prefer_test else "test", patterns)
    if not files:
        raise FileNotFoundError("nsnonbounded files not found")
    parts = []
    remaining = max_samples
    for path in files:
        with h5py.File(path, "r") as f:
            n_total = f["w0"].shape[0]
            n = n_total if remaining is None else min(remaining, n_total)
            w0 = f["w0"][:n]
            w = f["w"][:n]
        traj = np.concatenate([w0[:, None, :, :], np.moveaxis(w, -1, 1)], axis=1)
        parts.append(_as_float_tensor(traj).unsqueeze(1))
        if remaining is not None:
            remaining -= n
            if remaining <= 0:
                break
    return {
        "full_tensor": torch.cat(parts, dim=0),
        "channel_names": ["w"],
        "metadata": {
            "files": [str(p) for p in files],
            "canonical_layout": "NCTHW",
            "time_values": [i / 10 for i in range(11)],
            "final_time": 1.0,
            "nu": 1e-3,
        },
    }


def _load_burger(root: Path, split: str, max_samples: int | None, prefer_test: bool = False) -> dict[str, Any]:
    patterns = ["burger_test_1000-128-128.mat"] if split == "test" or prefer_test else ["burger_10000-128-128_*.mat"]
    files = _candidate_files(root, "burger", split if not prefer_test else "test", patterns, aliases=("burgers",))
    if not files:
        raise FileNotFoundError("Burgers files not found")
    parts = []
    initials = []
    remaining = max_samples
    for path in files:
        raw = _safe_loadmat(path, ("input", "output"))
        n_total = raw["output"].shape[0]
        n = n_total if remaining is None else min(remaining, n_total)
        parts.append(_as_float_tensor(raw["output"][:n]).unsqueeze(1))
        if "input" in raw:
            initials.append(_as_float_tensor(raw["input"][:n]))
        if remaining is not None:
            remaining -= n
            if remaining <= 0:
                break
    t_steps = parts[0].shape[-2] if parts else 0
    meta = {
        "files": [str(p) for p in files],
        "canonical_layout": "NCTX",
        "axes": ["time", "x"],
        "time_values": [i / max(t_steps - 1, 1) for i in range(t_steps)],
        "final_time": 1.0,
        "nu": 0.01,
    }
    if initials:
        meta["initial_1d"] = torch.cat(initials, dim=0)
    return {"full_tensor": torch.cat(parts, dim=0), "channel_names": ["u"], "metadata": meta}


def _load_reaction_diffusion(root: Path, split: str, max_samples: int | None, prefer_test: bool = False) -> dict[str, Any]:
    patterns = ["reaction_diffusion_test_1000-128-128-10.h5"] if split == "test" or prefer_test else ["reaction_diffusion-128-128-*_*.h5", "2D_diff-react_NA_NA.h5"]
    files = _candidate_files(root, "reaction_diffusion", split if not prefer_test else "test", patterns)
    if not files:
        raise FileNotFoundError("reaction_diffusion files not found")
    parts = []
    remaining = max_samples
    for path in files:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            for key in keys:
                data = np.asarray(f[key]["data"][:])
                # raw [T,H,W,2] -> canonical [2,T,H,W]
                parts.append(_as_float_tensor(np.moveaxis(data, -1, 0)))
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        break
        if remaining is not None and remaining <= 0:
            break
    full = torch.stack(parts, dim=0)
    input_idx = 50 if full.shape[2] > 50 else 0
    is_test = split == "test" or prefer_test
    return {
        "full_tensor": full,
        "channel_names": ["u", "v"],
        "metadata": {
            "files": [str(p) for p in files],
            "canonical_layout": "NCTHW",
            "input_time_index": input_idx,
            "final_time": 5.0,
            "D_u": 2e-3 if is_test else 1e-3,
            "D_v": 4e-3 if is_test else 5e-3,
            "k": 3e-3 if is_test else 5e-3,
        },
    }


def _load_shallow_water(root: Path, split: str, max_samples: int | None, prefer_test: bool = False) -> dict[str, Any]:
    patterns = ["swe_test_1000-128-128-10.h5"] if split == "test" or prefer_test else ["2d_swe_128_128_10_*.h5"]
    files = _candidate_files(root, "shallow_water", split if not prefer_test else "test", patterns)
    if not files:
        raise FileNotFoundError("shallow_water files not found")
    parts = []
    remaining = max_samples
    for path in files:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            for key in keys:
                group = f[key]["data"]
                channels = []
                for field in ("h", "hu", "hv"):
                    arr = np.asarray(group[field][:])
                    if arr.ndim == 4 and arr.shape[-1] == 1:
                        arr = arr[..., 0]
                    channels.append(arr)
                parts.append(_as_float_tensor(np.stack(channels, axis=0)))
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        break
        if remaining is not None and remaining <= 0:
            break
    return {
        "full_tensor": torch.stack(parts, dim=0),
        "channel_names": ["h", "hu", "hv"],
        "metadata": {
            "files": [str(p) for p in files],
            "canonical_layout": "NCTHW",
            "final_time": 1.0,
            "g": 1.0,
            "domain_length": 5.0,
        },
    }


def _load_heat(root: Path, split: str, max_samples: int | None, prefer_test: bool = False) -> dict[str, Any]:
    files = _candidate_files(
        root,
        "heat",
        split if not prefer_test else "test",
        ["heat_test*.h5"] if split == "test" or prefer_test else ["heat_10000-128-128_*.h5", "heat_*.h5"],
    )
    if not files:
        raise FileNotFoundError("heat files not found")
    return _load_future_field_h5(files, max_samples, "heat")


def _load_wave(root: Path, split: str, max_samples: int | None, prefer_test: bool = False) -> dict[str, Any]:
    files = _candidate_files(
        root,
        "wave",
        split if not prefer_test else "test",
        ["wave_test*.h5"] if split == "test" or prefer_test else ["wave_10000-128-128_*.h5", "wave_*.h5"],
    )
    if not files:
        raise FileNotFoundError("wave files not found")
    return _load_future_field_h5(files, max_samples, "wave")


def _load_advection_diffusion(root: Path, split: str, max_samples: int | None, prefer_test: bool = False) -> dict[str, Any]:
    files = _candidate_files(
        root,
        "advection_diffusion",
        split if not prefer_test else "test",
        ["advection_diffusion_test*.h5"] if split == "test" or prefer_test else ["advection_diffusion_10000-128-128_*.h5", "advection_diffusion_*.h5"],
    )
    if not files:
        raise FileNotFoundError("advection_diffusion files not found")
    return _load_future_field_h5(files, max_samples, "advection_diffusion")


def _load_steady_heat_conduction(root: Path, split: str, max_samples: int | None, prefer_test: bool = False) -> dict[str, Any]:
    train_patterns = ["steady_heat_conduction_10000-128-128_*.h5", "steady_heat_conduction_*.h5"]
    files = _candidate_files(
        root,
        "steady_heat_conduction",
        split if not prefer_test else "test",
        ["steady_heat_conduction_test*.h5"] if split == "test" or prefer_test else train_patterns,
    )
    if not files and (split == "test" or prefer_test):
        files = _candidate_files(root, "steady_heat_conduction", "train", train_patterns)
    if not files:
        raise FileNotFoundError("steady_heat_conduction files not found")
    return _load_future_field_h5(files, max_samples, "steady_heat_conduction")


def _load_future_field_h5(files: list[Path], max_samples: int | None, pde: str) -> dict[str, Any]:
    parts: list[torch.Tensor] = []
    scalar_meta: dict[str, list[torch.Tensor]] = {}
    meta: dict[str, Any] = {"files": [str(p) for p in files], "canonical_layout": "NCHW"}
    remaining = max_samples
    channel_names: list[str]
    for path in files:
        try:
            with h5py.File(path, "r") as f:
                n_total = int(f["input_data"].shape[0])
                n = n_total if remaining is None else min(remaining, n_total)
                inp = _as_float_tensor(f["input_data"][:n])
                out = _as_float_tensor(f["output_data"][:n])
                attrs = _h5_attrs_to_python(f)
                for key, value in attrs.items():
                    meta.setdefault(key, value)
                if "t" in f:
                    meta.setdefault("time_values", np.asarray(f["t"][:], dtype=np.float32).tolist())
                if "T" in attrs:
                    meta.setdefault("final_time", float(attrs["T"]))
                if "boundary_condition" in attrs:
                    meta.setdefault("bc", str(attrs["boundary_condition"]))

                if pde == "heat":
                    alpha = _scalar_or_attr_field(f, "alpha", "fixed_alpha", n, inp.shape[-2:], default=1e-3)
                    parts.append(torch.cat([inp[:, :1], alpha, out[:, :1], alpha.clone()], dim=1))
                    _append_scalar_meta(scalar_meta, "alpha", alpha)
                    channel_names = ["u0", "alpha", "uT", "alpha_T"]
                elif pde == "wave":
                    parts.append(torch.cat([inp[:, :2], out[:, :2]], dim=1))
                    if "c" in f:
                        c = _as_float_tensor(f["c"][:n])
                        scalar_meta.setdefault("c", []).append(c)
                    elif "fixed_c" in attrs:
                        meta.setdefault("fixed_c", float(attrs["fixed_c"]))
                    channel_names = ["u0", "v0", "uT", "vT"]
                elif pde == "advection_diffusion":
                    bx = _scalar_or_attr_field(f, "b_x", "b_x", n, inp.shape[-2:], default=0.0)
                    by = _scalar_or_attr_field(f, "b_y", "b_y", n, inp.shape[-2:], default=0.0)
                    kappa = _scalar_or_attr_field(f, "kappa", "kappa", n, inp.shape[-2:], default=1e-3)
                    parts.append(torch.cat([inp[:, :1], bx, by, kappa, out[:, :1], bx.clone(), by.clone(), kappa.clone()], dim=1))
                    _append_scalar_meta(scalar_meta, "b_x", bx)
                    _append_scalar_meta(scalar_meta, "b_y", by)
                    _append_scalar_meta(scalar_meta, "kappa", kappa)
                    channel_names = ["u0", "b_x", "b_y", "kappa", "uT", "b_x_T", "b_y_T", "kappa_T"]
                elif pde == "steady_heat_conduction":
                    u_d = _scalar_or_attr_field(f, "u_D", "u_D", n, inp.shape[-2:], default=298.0)
                    parts.append(torch.cat([inp[:, :1], u_d, out[:, :1], u_d.clone()], dim=1))
                    _append_scalar_meta(scalar_meta, "u_D", u_d)
                    channel_names = ["f", "u_D", "u", "u_D_T"]
                else:
                    raise ValueError(f"Unsupported future PDE '{pde}'")
        except OSError as exc:
            raise OSError(f"Could not read {pde} HDF5 file {path}: {exc}") from exc

        if remaining is not None:
            remaining -= n
            if remaining <= 0:
                break

    for key, values in scalar_meta.items():
        meta[key] = torch.cat(values, dim=0)
    return {"full_tensor": torch.cat(parts, dim=0), "channel_names": channel_names, "metadata": meta}


def _scalar_or_attr_field(
    f: h5py.File,
    dataset_key: str,
    attr_key: str,
    n: int,
    spatial_shape: tuple[int, int],
    default: float,
) -> torch.Tensor:
    if dataset_key in f:
        scalar = _as_float_tensor(f[dataset_key][:n])
    elif attr_key in f.attrs:
        scalar = torch.full((n,), float(f.attrs[attr_key]), dtype=torch.float32)
    else:
        scalar = torch.full((n,), float(default), dtype=torch.float32)
    return scalar.reshape(n, 1, 1, 1).expand(n, 1, spatial_shape[0], spatial_shape[1]).clone()


def _append_scalar_meta(target: dict[str, list[torch.Tensor]], key: str, field: torch.Tensor) -> None:
    scalar = field[:, 0, 0, 0].detach().clone()
    target.setdefault(key, []).append(scalar)


def _h5_attrs_to_python(f: h5py.File) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key, value in f.attrs.items():
        if isinstance(value, np.generic):
            attrs[key] = value.item()
        elif isinstance(value, bytes):
            attrs[key] = value.decode("utf-8")
        else:
            attrs[key] = value
    return attrs


def _missing_future_loader(name: str) -> Callable[..., dict[str, Any]]:
    def loader(root: Path, split: str, max_samples: int | None, prefer_test: bool = False) -> dict[str, Any]:
        patterns = [f"{name}*.mat", f"{name}*.h5"]
        files = _candidate_files(root, name, split, patterns)
        if not files:
            raise FileNotFoundError(
                f"No files for reserved PDE '{name}' under {root}. Add an adapter when data is available."
            )
        raise NotImplementedError(f"Files exist for '{name}', but no adapter has been implemented yet.")

    return loader


def build_default_registry() -> PDEDataRegistry:
    return PDEDataRegistry()
