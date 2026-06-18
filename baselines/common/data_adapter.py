from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
import re

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
    input_channel_names: list[str]
    target_channel_names: list[str]
    metadata: dict
    pde_params: dict[str, Any]
    split: str
    sample_indices: torch.Tensor | None
    global_sample_ids: list[str]
    file_paths: list[str]


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


@dataclass(frozen=True)
class LazySampleRef:
    path: Path
    local_index: int
    global_index: int
    key: str | None = None


LAZY_PDES = {
    "nsnonbounded",
    "reaction_diffusion",
    "shallow_water",
    "heat",
    "wave",
    "advection_diffusion",
    "steady_heat_conduction",
}


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


def _validate_split(split: str) -> str:
    split = str(split).lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be one of train/val/test, got {split!r}")
    return split


def _expand_scalar_to_field(values: torch.Tensor | np.ndarray, h: int, w: int) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1, 1, 1)
    return tensor.expand(tensor.shape[0], 1, h, w).clone()


def _names_from_indices(channel_names: list[str], indices: Any) -> list[str]:
    if indices is None:
        return []
    try:
        idx = [int(i) for i in indices]
        out = [channel_names[i] for i in idx if 0 <= i < len(channel_names)]
    except Exception:
        return []
    return out if len(out) == len(idx) else []


def _apply_sample_offset_tensor(x: torch.Tensor, offset: int, max_samples: int | None) -> torch.Tensor:
    offset = max(int(offset), 0)
    end = None if max_samples is None else offset + int(max_samples)
    return x[offset:end]


def _file_sample_ids(path: Path, local_start: int, count: int) -> list[str]:
    return [f"{path.name}:{local_start + i}" for i in range(count)]


def _finalize_loaded_raw(raw: dict[str, Any], max_samples: int | None, strict_size: bool) -> dict[str, Any]:
    n = int(raw["full_tensor"].shape[0])
    raw.setdefault("metadata", {})
    raw.setdefault("pde_params", raw["metadata"].get("pde_params", {}))
    raw.setdefault("split", raw["metadata"].get("split", ""))
    raw.setdefault("file_paths", raw["metadata"].get("files", []))
    raw.setdefault("sample_indices", raw["metadata"].get("sample_indices", torch.arange(n, dtype=torch.long)))
    raw.setdefault("global_sample_ids", raw["metadata"].get("global_sample_ids", []))
    raw["metadata"].setdefault("available_count", n)
    raw["metadata"].setdefault("loaded_count", n)
    raw["metadata"].setdefault("files", list(raw.get("file_paths", [])))
    raw["metadata"].setdefault("pde_params", raw.get("pde_params", {}))
    raw["metadata"].setdefault("data_loading_mode", "eager")
    raw["metadata"].setdefault("train_size_loaded_in_memory", n)
    raw["metadata"].setdefault(
        "loaded_full_trajectory",
        bool(raw["full_tensor"].ndim == 5 or isinstance(raw["metadata"].get("full_trajectory"), torch.Tensor)),
    )
    if max_samples is not None and n < int(max_samples):
        message = f"Requested {max_samples} samples but only loaded {n} from split={raw.get('split', raw['metadata'].get('split', 'unknown'))}."
        if strict_size:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return raw


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
        train_shards: int = 5,
        sample_offset: int = 0,
        val_from_train_offset: int | None = None,
        prefer_test: bool = False,
        synthetic_if_missing: bool = False,
        synthetic_resolution: int = 32,
        synthetic_seed: int = 17,
        scalar_param_mode: str = "metadata",
        load_full_trajectory: bool = True,
        strict_size: bool = False,
    ) -> dict[str, Any]:
        spec = self.get(pde_name)
        split = _validate_split(split)
        root = Path(data_root)
        if scalar_param_mode == "global":
            warnings.warn(
                "--scalar-param-mode global is reserved for future global-conditioning adapters; "
                "using metadata-only scalar parameters for this run.",
                RuntimeWarning,
                stacklevel=2,
            )
            scalar_param_mode = "metadata"
        try:
            raw = spec.loader(
                root,
                split=split,
                max_samples=max_samples,
                train_shards=train_shards,
                sample_offset=sample_offset,
                val_from_train_offset=val_from_train_offset,
                prefer_test=prefer_test,
                scalar_param_mode=scalar_param_mode,
                load_full_trajectory=load_full_trajectory,
                strict_size=strict_size,
            )
            if split == "val":
                raw.setdefault("metadata", {}).setdefault("split_source", "independent_val")
            return raw
        except FileNotFoundError as exc:
            if split == "val" and val_from_train_offset is not None:
                train_offset = int(val_from_train_offset)
                try:
                    raw = spec.loader(
                        root,
                        split="train",
                        max_samples=max_samples,
                        train_shards=train_shards,
                        sample_offset=train_offset,
                        val_from_train_offset=val_from_train_offset,
                        prefer_test=False,
                        scalar_param_mode=scalar_param_mode,
                        load_full_trajectory=load_full_trajectory,
                        strict_size=strict_size,
                    )
                    raw.setdefault("metadata", {})["split"] = "val"
                    raw["metadata"]["split_source"] = "deterministic_train_subset"
                    raw["metadata"]["val_from_train_offset"] = train_offset
                    raw["split"] = "val"
                    return _finalize_loaded_raw(raw, max_samples, strict_size=strict_size)
                except FileNotFoundError:
                    pass
            if synthetic_if_missing:
                warnings.warn(
                    f"{spec.name}: {exc}. Falling back to synthetic smoke data.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self.synthetic_raw(
                    spec.name,
                    max_samples or 8,
                    synthetic_resolution,
                    split=split,
                    seed=synthetic_seed,
                    scalar_param_mode=scalar_param_mode,
                )
            raise

    def to_canonical(self, raw: dict[str, Any], pde_name: str) -> dict[str, Any]:
        spec = self.get(pde_name)
        return {
            "pde_name": spec.name,
            "full_tensor": raw["full_tensor"].float(),
            "channel_names": list(raw.get("channel_names", spec.channel_names)),
            "metadata": dict(raw.get("metadata", {})),
            "input_channel_names": list(raw.get("input_channel_names", [])),
            "target_channel_names": list(raw.get("target_channel_names", [])),
            "pde_params": dict(raw.get("pde_params", raw.get("metadata", {}).get("pde_params", {}))),
            "split": str(raw.get("split", raw.get("metadata", {}).get("split", ""))),
            "sample_indices": raw.get("sample_indices"),
            "global_sample_ids": list(raw.get("global_sample_ids", raw.get("metadata", {}).get("global_sample_ids", []))),
            "file_paths": list(raw.get("file_paths", raw.get("metadata", {}).get("files", []))),
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
        experiment_mode: str = "debug",
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
        metadata.setdefault("split", canonical.get("split", ""))
        metadata.setdefault("files", list(canonical.get("file_paths", [])))
        pde_params = dict(canonical.get("pde_params", metadata.get("pde_params", {})))
        if pde_params:
            metadata["pde_params"] = pde_params
            for key, value in pde_params.items():
                metadata.setdefault(key, value)

        input_fields, target_fields, input_names, target_names = self._split_task(full, spec, task, metadata, canonical)
        if task in {"sparse_solution", "sparse_reconstruction"} and sensor_mode == "time_varying" and full.ndim == 5:
            trajectory_target, trajectory_names = _trajectory_target_for_sparse(full, spec, metadata)
            if trajectory_target is not None:
                target_fields = trajectory_target
                target_names = trajectory_names
        original_input_fields = input_fields
        metadata["original_input_fields"] = original_input_fields
        metadata["background_fields"] = _background_fields_for_task(full, spec, metadata, original_input_fields)
        coords = make_coordinate_grid(tuple(target_fields.shape[2:]), batch_size=target_fields.shape[0])

        mask = obs_values = obs_coords = None
        if task.startswith("sparse") or num_sensors:
            requested_sensor_mode = sensor_mode
            effective_sensor_mode = sensor_mode
            observation_source = target_fields
            observation_names = list(target_names)
            if task == "sparse_inverse":
                observation_source = input_fields
                observation_names = list(input_names)
                metadata["observed_solution_fields"] = observation_source
                metadata["observation_source_fields"] = observation_source
                metadata["observation_source_channel_names"] = observation_names
            has_time_axis = observation_source.ndim == 5 or (spec.name == "burger" and observation_source.ndim == 4)
            time_varying_valid = not (requested_sensor_mode == "time_varying") or has_time_axis
            if requested_sensor_mode == "time_varying" and not time_varying_valid:
                message = (
                    "sensor_mode='time_varying' requires task observation fields shaped [B,C,T,H,W] or Burgers [B,C,T,X]; "
                    f"got {tuple(observation_source.shape)} for {spec.name}/{task}."
                )
                if experiment_mode == "paper":
                    raise ValueError(message)
                warnings.warn(message + " Falling back to random sensors for this smoke/debug run.", RuntimeWarning, stacklevel=2)
                effective_sensor_mode = "random"
            obs = build_observation_tensors(
                observation_source,
                num_sensors=num_sensors or 500,
                mode=effective_sensor_mode,
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
                    "requested_sensor_mode": requested_sensor_mode,
                    "effective_sensor_mode": effective_sensor_mode,
                    "sensor_mode": effective_sensor_mode,
                    "time_varying_sensor_valid": bool(time_varying_valid),
                    "num_sensors": int(num_sensors or 500),
                    "noise_level": float(noise_level),
                    "mask_id": obs["mask_id"],
                }
            )
            if task in {"sparse_solution", "sparse_reconstruction"}:
                input_fields = obs["masked_grid"]
                input_names = list(target_names)
            elif task == "sparse_inverse":
                input_fields = obs["masked_grid"]
                input_names = observation_names
        else:
            metadata.setdefault("requested_sensor_mode", "none")
            metadata.setdefault("effective_sensor_mode", "none")
            metadata.setdefault("time_varying_sensor_valid", True)

        metadata["input_shape"] = tuple(input_fields.shape)
        metadata["target_shape"] = tuple(target_fields.shape)
        metadata["full_shape"] = tuple(full.shape)
        metadata["input_channel_names"] = input_names
        metadata["target_channel_names"] = target_names
        metadata["task_channel_names"] = target_names
        metadata["pde_params_available"] = sorted(pde_params)

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
            input_channel_names=input_names,
            target_channel_names=target_names,
            metadata=metadata,
            pde_params=pde_params,
            split=str(canonical.get("split", metadata.get("split", ""))),
            sample_indices=canonical.get("sample_indices"),
            global_sample_ids=list(canonical.get("global_sample_ids", [])),
            file_paths=list(canonical.get("file_paths", metadata.get("files", []))),
        )

    def make_dataset(
        self,
        pde_name: str,
        data_root: str | Path,
        task: str,
        split: str = "train",
        max_samples: int | None = None,
        train_shards: int = 5,
        sample_offset: int = 0,
        val_from_train_offset: int | None = None,
        num_sensors: int | None = None,
        sensor_mode: str = "random",
        noise_level: float = 0.0,
        seed: int = 0,
        prefer_test: bool = False,
        synthetic_if_missing: bool = False,
        synthetic_resolution: int = 32,
        synthetic_seed: int = 17,
        scalar_param_mode: str = "metadata",
        data_loading_mode: str = "eager",
        load_full_trajectory: bool = True,
        experiment_mode: str = "debug",
        strict_size: bool = False,
    ) -> Dataset:
        if data_loading_mode not in {"eager", "lazy"}:
            raise ValueError(f"data_loading_mode must be eager or lazy, got {data_loading_mode!r}")
        spec = self.get(pde_name)
        if data_loading_mode == "lazy" and spec.name in LAZY_PDES:
            try:
                return self.make_lazy_dataset(
                    pde_name,
                    data_root,
                    task,
                    split=split,
                    max_samples=max_samples,
                    train_shards=train_shards,
                    sample_offset=sample_offset,
                    val_from_train_offset=val_from_train_offset,
                    num_sensors=num_sensors,
                    sensor_mode=sensor_mode,
                    noise_level=noise_level,
                    seed=seed,
                    prefer_test=prefer_test,
                    scalar_param_mode=scalar_param_mode,
                    load_full_trajectory=load_full_trajectory,
                    experiment_mode=experiment_mode,
                    strict_size=strict_size,
                )
            except NotImplementedError:
                pass
        raw = self.load_raw(
            pde_name,
            data_root,
            split=split,
            max_samples=max_samples,
            train_shards=train_shards,
            sample_offset=sample_offset,
            val_from_train_offset=val_from_train_offset,
            prefer_test=prefer_test,
            synthetic_if_missing=synthetic_if_missing,
            synthetic_resolution=synthetic_resolution,
            synthetic_seed=synthetic_seed,
            scalar_param_mode=scalar_param_mode,
            load_full_trajectory=load_full_trajectory,
            strict_size=strict_size,
        )
        batch = self.make_task(
            self.to_canonical(raw, pde_name),
            pde_name,
            task,
            num_sensors=num_sensors,
            sensor_mode=sensor_mode,
            noise_level=noise_level,
            seed=seed,
            experiment_mode=experiment_mode,
        )
        return PDEBatchDataset(batch)

    def make_lazy_dataset(
        self,
        pde_name: str,
        data_root: str | Path,
        task: str,
        split: str = "train",
        max_samples: int | None = None,
        train_shards: int = 5,
        sample_offset: int = 0,
        val_from_train_offset: int | None = None,
        num_sensors: int | None = None,
        sensor_mode: str = "random",
        noise_level: float = 0.0,
        seed: int = 0,
        prefer_test: bool = False,
        scalar_param_mode: str = "metadata",
        load_full_trajectory: bool = False,
        experiment_mode: str = "debug",
        strict_size: bool = False,
    ) -> "LazyPDEBatchDataset":
        refs, meta = _build_lazy_refs(
            self.get(pde_name).name,
            Path(data_root),
            split=split,
            max_samples=max_samples,
            train_shards=train_shards,
            sample_offset=sample_offset,
            val_from_train_offset=val_from_train_offset,
            prefer_test=prefer_test,
            strict_size=strict_size,
        )
        return LazyPDEBatchDataset(
            registry=self,
            pde_name=self.get(pde_name).name,
            task=task,
            refs=refs,
            split=_validate_split(split),
            file_paths=meta["file_paths"],
            num_sensors=num_sensors,
            sensor_mode=sensor_mode,
            noise_level=noise_level,
            seed=seed,
            scalar_param_mode=scalar_param_mode,
            load_full_trajectory=load_full_trajectory,
            experiment_mode=experiment_mode,
            metadata=meta,
        )

    def synthetic_raw(
        self,
        pde_name: str,
        n: int = 8,
        resolution: int = 32,
        split: str = "train",
        seed: int = 17,
        scalar_param_mode: str = "metadata",
    ) -> dict[str, Any]:
        pde_name = pde_name.lower()
        split = _validate_split(split)
        materialize = scalar_param_mode == "materialize"
        gen = torch.Generator().manual_seed(seed)
        h = w = resolution
        pde_params: dict[str, torch.Tensor] = {}
        if pde_name in {"darcy", "poisson", "helmholtz"}:
            c = 2
            x = torch.randn(n, c, h, w, generator=gen)
            channels = self.get(pde_name).channel_names
            meta = {"source": "synthetic", "canonical_layout": "NCHW"}
        elif pde_name == "heat":
            u0 = torch.randn(n, 1, h, w, generator=gen)
            uT = torch.randn(n, 1, h, w, generator=gen)
            alpha = torch.full((n,), 1e-3)
            pde_params = {"alpha": alpha}
            if materialize:
                alpha_field = _expand_scalar_to_field(alpha, h, w)
                x = torch.cat([u0, alpha_field, uT, alpha_field.clone()], dim=1)
                channels = ["u0", "alpha", "uT", "alpha_T"]
                meta_indices = {"input_indices": [0, 1], "target_indices": [2, 3]}
            else:
                x = torch.cat([u0, uT], dim=1)
                channels = self.get(pde_name).channel_names
                meta_indices = {"input_indices": [0], "target_indices": [1]}
            meta = {"source": "synthetic", "canonical_layout": "NCHW", "alpha": alpha, "final_time": 1.0, "bc": "periodic", **meta_indices}
        elif pde_name == "wave":
            u0 = torch.randn(n, 1, h, w, generator=gen)
            v0 = torch.zeros(n, 1, h, w)
            uT = torch.randn(n, 1, h, w, generator=gen)
            vT = torch.randn(n, 1, h, w, generator=gen)
            x = torch.cat([u0, v0, uT, vT], dim=1)
            channels = self.get(pde_name).channel_names
            c_param = torch.full((n,), 1.0)
            pde_params = {"c": c_param}
            meta = {"source": "synthetic", "canonical_layout": "NCHW", "fixed_c": 1.0, "c": c_param, "final_time": 1.0, "bc": "periodic"}
        elif pde_name == "advection_diffusion":
            u0 = torch.randn(n, 1, h, w, generator=gen)
            uT = torch.randn(n, 1, h, w, generator=gen)
            bx = torch.full((n,), 0.25)
            by = torch.full((n,), -0.15)
            kappa = torch.full((n,), 1e-3)
            pde_params = {"b_x": bx, "b_y": by, "kappa": kappa}
            if materialize:
                bx_f = _expand_scalar_to_field(bx, h, w)
                by_f = _expand_scalar_to_field(by, h, w)
                k_f = _expand_scalar_to_field(kappa, h, w)
                x = torch.cat([u0, bx_f, by_f, k_f, uT, bx_f.clone(), by_f.clone(), k_f.clone()], dim=1)
                channels = ["u0", "b_x", "b_y", "kappa", "uT", "b_x_T", "b_y_T", "kappa_T"]
                meta_indices = {"input_indices": [0, 1, 2, 3], "target_indices": [4, 5, 6, 7]}
            else:
                x = torch.cat([u0, uT], dim=1)
                channels = self.get(pde_name).channel_names
                meta_indices = {"input_indices": [0], "target_indices": [1]}
            meta = {
                "source": "synthetic",
                "canonical_layout": "NCHW",
                "b_x": bx,
                "b_y": by,
                "kappa": kappa,
                "final_time": 1.0,
                "bc": "periodic",
                **meta_indices,
            }
        elif pde_name in {"steady_heat_conduction", "steady_heat"}:
            source = torch.randn(n, 1, h, w, generator=gen)
            u_d = torch.full((n,), 298.0)
            solution = 298.0 + torch.randn(n, 1, h, w, generator=gen) * 0.01
            pde_params = {"u_D": u_d}
            if materialize:
                u_d_f = _expand_scalar_to_field(u_d, h, w)
                x = torch.cat([source, u_d_f, solution, u_d_f.clone()], dim=1)
                channels = ["f", "u_D", "u", "u_D_T"]
                meta_indices = {"input_indices": [0, 1], "target_indices": [2, 3]}
            else:
                x = torch.cat([source, solution], dim=1)
                channels = self.get(pde_name).channel_names
                meta_indices = {"input_indices": [0], "target_indices": [1]}
            meta = {"source": "synthetic", "canonical_layout": "NCHW", "u_D": u_d, **meta_indices}
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
        sample_indices = torch.arange(n, dtype=torch.long)
        meta.update(
            {
                "split": split,
                "scalar_param_mode": scalar_param_mode,
                "data_loading_mode": "eager",
                "load_full_trajectory": True,
                "loaded_full_trajectory": bool(x.ndim == 5),
                "pde_params": pde_params,
                "pde_params_available": sorted(pde_params),
                "sample_indices": sample_indices,
                "global_sample_ids": [f"synthetic:{pde_name}:{split}:{seed}:{i}" for i in range(n)],
            }
        )
        input_channel_names = _names_from_indices(channels, meta.get("input_indices"))
        target_channel_names = _names_from_indices(channels, meta.get("target_indices"))
        return _finalize_loaded_raw(
            {
                "full_tensor": x,
                "channel_names": channels,
                "input_channel_names": input_channel_names,
                "target_channel_names": target_channel_names,
                "metadata": meta,
                "pde_params": pde_params,
                "split": split,
                "sample_indices": sample_indices,
                "global_sample_ids": meta["global_sample_ids"],
                "file_paths": [],
            },
            max_samples=n,
            strict_size=True,
        )

    def _split_task(
        self, full: torch.Tensor, spec: PDESpec, task: str, metadata: dict[str, Any], canonical: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
        name = spec.name
        canonical = canonical or {}
        if name == "burger":
            target = full
            initial = metadata.get("initial_1d")
            if initial is None:
                input_fields = full[:, :, :1, :].repeat(1, 1, full.shape[-2], 1)
            else:
                init = initial.to(full.device, full.dtype).reshape(full.shape[0], 1, 1, full.shape[-1])
                input_fields = init.repeat(1, 1, full.shape[-2], 1)
            if task in {"inverse", "sparse_inverse"}:
                return target, input_fields[:, :, :1, :], ["u"], ["u0"]
            return input_fields, target, ["u0"], ["u"]

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
            input_indices = list(metadata.get("input_indices", spec.input_indices))
            target_indices = list(metadata.get("target_indices", spec.target_indices))
            inp = full[:, input_indices]
            target = full[:, target_indices]
            target_names = list(canonical.get("target_channel_names") or metadata.get("target_channel_names") or [spec.channel_names[i] for i in target_indices])

        if task in {"inverse", "sparse_inverse"}:
            input_names = target_names
            target_names = list(canonical.get("input_channel_names") or metadata.get("input_channel_names") or [spec.channel_names[i] for i in metadata.get("input_indices", spec.input_indices)])
            return target, inp, input_names, target_names
        if task in {"both", "joint"}:
            return full, full, list(spec.channel_names), list(spec.channel_names)
        input_names = list(canonical.get("input_channel_names") or metadata.get("input_channel_names") or [spec.channel_names[i] for i in metadata.get("input_indices", spec.input_indices)])
        return inp, target, input_names, target_names

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
                ["u0", "uT"],
                [0],
                [1],
                _load_heat,
                time_dependent=True,
                notes="Future HDF5 layout uses physical fields [u0,uT]; alpha is metadata unless scalar_param_mode=materialize.",
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
                ["u0", "uT"],
                [0],
                [1],
                _load_advection_diffusion,
                aliases=("advdiff",),
                time_dependent=True,
                notes="Future HDF5 layout uses physical fields [u0,uT]; b_x/b_y/kappa are metadata unless materialized.",
            )
        )
        self.register(
            PDESpec(
                "steady_heat_conduction",
                ["f", "u"],
                [0],
                [1],
                _load_steady_heat_conduction,
                aliases=("steady_heat", "nonlinear_heat_conduction"),
                notes="Future HDF5 layout uses physical fields [f,u]; u_D is metadata unless materialized.",
            )
        )


class PDEBatchDataset(Dataset):
    def __init__(self, batch: PDEBatch) -> None:
        self.batch = batch
        self.loaded_in_memory_samples = int(batch.input_fields.shape[0])
        self.loaded_full_trajectory = _batch_has_full_trajectory(batch)
        self.data_loading_mode = "eager"

    def __len__(self) -> int:
        return int(self.batch.input_fields.shape[0])

    def __getitem__(self, index: int) -> PDEBatch:
        return slice_pde_batch(self.batch, index)


class LazyPDEBatchDataset(Dataset):
    def __init__(
        self,
        registry: PDEDataRegistry,
        pde_name: str,
        task: str,
        refs: list[LazySampleRef],
        split: str,
        file_paths: list[str],
        num_sensors: int | None,
        sensor_mode: str,
        noise_level: float,
        seed: int,
        scalar_param_mode: str,
        load_full_trajectory: bool,
        experiment_mode: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not refs:
            raise ValueError(f"Lazy dataset for {pde_name}/{split} has no samples")
        self.registry = registry
        self.pde_name = pde_name
        self.task = task
        self.refs = refs
        self.split = split
        self.file_paths = file_paths
        self.num_sensors = num_sensors
        self.sensor_mode = sensor_mode
        self.noise_level = float(noise_level)
        self.seed = int(seed)
        self.scalar_param_mode = scalar_param_mode
        self.load_full_trajectory = bool(load_full_trajectory)
        self.experiment_mode = experiment_mode
        self.metadata = dict(metadata or {})
        self.data_loading_mode = "lazy"
        self.loaded_in_memory_samples = 0
        self.loaded_full_trajectory = bool(load_full_trajectory)
        self.samples_read = 0
        self._prototype: PDEBatch | None = None

    def __len__(self) -> int:
        return len(self.refs)

    @property
    def batch(self) -> PDEBatch:
        if self._prototype is None:
            self._prototype = self[0]
            self._prototype.file_paths = list(self.file_paths)
            self._prototype.metadata.update(
                {
                    "lazy_dataset_total_samples": len(self.refs),
                    "files": list(self.file_paths),
                    "data_loading_mode": "lazy",
                    "train_size_loaded_in_memory": 0,
                }
            )
        return self._prototype

    def __getitem__(self, index: int) -> PDEBatch:
        if index < 0:
            index += len(self.refs)
        if index < 0 or index >= len(self.refs):
            raise IndexError(index)
        ref = self.refs[index]
        raw = _read_lazy_raw_sample(
            self.pde_name,
            ref,
            split=self.split,
            scalar_param_mode=self.scalar_param_mode,
            load_full_trajectory=self.load_full_trajectory,
            file_paths=self.file_paths,
            dataset_metadata=self.metadata,
        )
        self.samples_read += 1
        return self.registry.make_task(
            raw,
            self.pde_name,
            self.task,
            num_sensors=self.num_sensors,
            sensor_mode=self.sensor_mode,
            noise_level=self.noise_level,
            seed=self.seed,
            experiment_mode=self.experiment_mode,
        )


def slice_pde_batch(batch: PDEBatch, index: int) -> PDEBatch:
    sl = slice(index, index + 1)
    metadata = _slice_metadata(batch.metadata, sl, batch.full_tensor.shape[0])
    pde_params = _slice_metadata(batch.pde_params, sl, batch.full_tensor.shape[0])
    global_ids = batch.global_sample_ids[index : index + 1] if batch.global_sample_ids else []
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
        input_channel_names=batch.input_channel_names,
        target_channel_names=batch.target_channel_names,
        metadata=metadata,
        pde_params=pde_params,
        split=batch.split,
        sample_indices=batch.sample_indices[sl] if batch.sample_indices is not None else None,
        global_sample_ids=global_ids,
        file_paths=batch.file_paths,
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
    metadata = _collate_metadata([b.metadata for b in items], first.full_tensor.shape[0])
    pde_params = _collate_metadata([b.pde_params for b in items], first.full_tensor.shape[0])
    sample_indices = None
    if first.sample_indices is not None:
        sample_indices = torch.cat([b.sample_indices for b in items if b.sample_indices is not None], dim=0)
    global_ids: list[str] = []
    for b in items:
        global_ids.extend(b.global_sample_ids)
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
        input_channel_names=first.input_channel_names,
        target_channel_names=first.target_channel_names,
        metadata=metadata,
        pde_params=pde_params,
        split=first.split,
        sample_indices=sample_indices,
        global_sample_ids=global_ids,
        file_paths=first.file_paths,
    )


def _batch_has_full_trajectory(batch: PDEBatch) -> bool:
    if batch.full_tensor.ndim == 5:
        return True
    return isinstance(batch.metadata.get("full_trajectory"), torch.Tensor)


def _trajectory_target_for_sparse(full: torch.Tensor, spec: PDESpec, metadata: dict[str, Any]) -> tuple[torch.Tensor | None, list[str]]:
    if full.ndim != 5:
        return None, []
    name = spec.name
    if name == "nsnonbounded":
        return full[:, :, 1:], list(spec.channel_names)
    input_idx = int(metadata.get("input_time_index", 0))
    input_idx = max(0, min(input_idx, full.shape[2] - 1))
    return full[:, :, input_idx:], list(spec.channel_names)


def _background_fields_for_task(full: torch.Tensor, spec: PDESpec, metadata: dict[str, Any], original_input: torch.Tensor) -> torch.Tensor:
    name = spec.name
    if full.ndim == 5:
        if name == "nsnonbounded":
            return full[:, :, 0]
        idx = int(metadata.get("input_time_index", 0))
        idx = max(0, min(idx, full.shape[2] - 1))
        return full[:, :, idx]
    if name == "burger":
        initial = metadata.get("initial_1d")
        if isinstance(initial, torch.Tensor):
            return initial.reshape(full.shape[0], 1, 1, full.shape[-1])
        if full.ndim == 4:
            return full[:, :1, :1, :]
    return original_input


def _slice_metadata(metadata: dict[str, Any], sl: slice, batch_n: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == (batch_n,):
            out[key] = value[sl]
        elif isinstance(value, dict):
            out[key] = _slice_metadata(value, sl, batch_n)
        else:
            out[key] = value
    return out


def _collate_metadata(items: list[dict[str, Any]], item_n: int) -> dict[str, Any]:
    if not items:
        return {}
    out = dict(items[0])
    for key, value in list(out.items()):
        if isinstance(value, torch.Tensor) and value.shape[:1] == (item_n,):
            tensors = [m[key] for m in items if isinstance(m.get(key), torch.Tensor)]
            out[key] = torch.cat(tensors, dim=0)
        elif isinstance(value, dict):
            out[key] = _collate_metadata([m.get(key, {}) for m in items], item_n)
    return out


def _candidate_files(root: Path, pde: str, split: str, patterns: list[str], aliases: tuple[str, ...] = ()) -> list[Path]:
    root = Path(root).expanduser()
    dirs = [root / pde, *(root / a for a in aliases), root]
    if split == "test":
        # Formal per-PDE test files are preferred. ``test1125`` is a legacy
        # compatibility directory and is searched only after PDE-local paths.
        dirs.append(root / "test1125")
    found: list[Path] = []
    for base in dirs:
        for pat in patterns:
            found.extend(sorted(base.glob(pat)))
    unique: list[Path] = []
    seen = set()
    for path in found:
        if path.exists() and path not in seen:
            unique.append(path)
            seen.add(path)
    return sorted(unique, key=_shard_sort_key)


def _candidate_descriptions(root: Path, pde: str, split: str, patterns: list[str], aliases: tuple[str, ...] = ()) -> list[str]:
    dirs = [Path(root).expanduser() / pde, *(Path(root).expanduser() / a for a in aliases), Path(root).expanduser()]
    if split == "test":
        dirs.append(Path(root).expanduser() / "test1125")
    return [str(base / pat) for base in dirs for pat in patterns]


def _missing_error(root: Path, pde: str, split: str, patterns: list[str], aliases: tuple[str, ...] = ()) -> FileNotFoundError:
    candidates = "\n  - ".join(_candidate_descriptions(root, pde, split, patterns, aliases))
    return FileNotFoundError(f"No {split} files found for PDE '{pde}'. Candidate paths:\n  - {candidates}")


def _shard_sort_key(path: Path) -> tuple[str, int, str]:
    stem = path.stem
    tail = stem.rsplit("_", 1)[-1]
    try:
        shard = int(tail)
    except ValueError:
        shard = 10**9
    return (stem.rsplit("_", 1)[0], shard, path.name)


def _train_limited(files: list[Path], split: str, train_shards: int | None) -> list[Path]:
    if split == "train":
        files = [p for p in files if "_test" not in p.name and "_val" not in p.name and "test" not in p.stem.lower() and "val" not in p.stem.lower()]
    if split == "train" and train_shards is not None:
        return files[: int(train_shards)]
    return files


def _filter_nsnonbounded_test_files(files: list[Path]) -> list[Path]:
    legal: list[Path] = []
    train_shard = re.compile(r"^nsnonbounded_10000-128-128-10_\d+_new\.mat$")
    for path in files:
        name = path.name.lower()
        if "_new" in name and "test" not in name:
            continue
        if train_shard.match(name):
            continue
        if "10000" in name and "test" not in name:
            continue
        if "test" in name or name.startswith("nsnonbounded_1000-128-128-10"):
            legal.append(path)
    return legal


def _build_lazy_refs(
    pde: str,
    root: Path,
    split: str,
    max_samples: int | None,
    train_shards: int,
    sample_offset: int,
    val_from_train_offset: int | None,
    prefer_test: bool,
    strict_size: bool,
) -> tuple[list[LazySampleRef], dict[str, Any]]:
    split = _validate_split(split)
    active_split = "test" if prefer_test else split
    offset = max(int(sample_offset), 0)
    split_source = "independent_val" if split == "val" else split
    if split == "val" and val_from_train_offset is not None:
        active_split = "train"
        offset = int(val_from_train_offset)
        split_source = "deterministic_train_subset"
    files = _lazy_candidate_files(root, pde, active_split, train_shards)
    if not files:
        raise _missing_error(root, pde, active_split, _lazy_patterns(pde, active_split))
    refs: list[LazySampleRef] = []
    skip = offset
    remaining = max_samples
    seen = 0
    for path in files:
        sample_keys = _lazy_group_keys(path, pde)
        n_total = len(sample_keys) if sample_keys is not None else _lazy_dense_count(path, pde)
        if skip >= n_total:
            skip -= n_total
            seen += n_total
            continue
        local_start = skip
        n = n_total - local_start if remaining is None else min(int(remaining), n_total - local_start)
        for j in range(n):
            local = local_start + j
            refs.append(LazySampleRef(path=path, local_index=local, global_index=seen + local, key=sample_keys[local] if sample_keys else None))
        skip = 0
        seen += n_total
        if remaining is not None:
            remaining -= n
            if remaining <= 0:
                break
    if max_samples is not None and len(refs) < int(max_samples):
        message = f"Requested {max_samples} samples but indexed {len(refs)} lazily from split={split}."
        if strict_size:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    if not refs:
        raise FileNotFoundError(f"No lazy samples indexed for PDE '{pde}' from files: {[str(p) for p in files]}")
    meta = {
        "files": [str(p) for p in files],
        "file_paths": [str(p) for p in files],
        "split": split,
        "active_split": active_split,
        "split_source": split_source,
        "available_count": len(refs),
        "loaded_count": len(refs),
        "data_loading_mode": "lazy",
        "train_size_loaded_in_memory": 0,
    }
    if split_source == "deterministic_train_subset":
        meta["val_from_train_offset"] = offset
    return refs, meta


def _lazy_patterns(pde: str, split: str) -> list[str]:
    if pde == "nsnonbounded":
        if split == "test":
            return [
                "nsnonbounded_test_*-128-128-10*.mat",
                "nsnonbounded_1000-128-128-10*.mat",
                "nsnonbounded_10000-128-128-10_test*.mat",
                "nsnonbounded_*_test*.mat",
            ]
        if split == "val":
            return ["nsnonbounded_val_*-128-128-10_*.mat", "nsnonbounded_*-128-128-10_val*.mat"]
        return ["nsnonbounded_10000-128-128-10_*_new.mat"]
    if pde == "reaction_diffusion":
        if split == "test":
            return ["reaction_diffusion_test_*-128-128-*.h5"]
        if split == "val":
            return ["reaction_diffusion_val_*-128-128-*.h5", "reaction_diffusion-128-128-*_val*.h5"]
        return ["reaction_diffusion-128-128-*_*.h5", "2D_diff-react_NA_NA.h5"]
    if pde == "shallow_water":
        if split == "test":
            return ["swe_test_*-128-128-*.h5", "2d_swe_test*.h5"]
        if split == "val":
            return ["swe_val_*-128-128-*.h5", "2d_swe_val*.h5"]
        return ["2d_swe_128_128_10_*.h5"]
    if pde in {"heat", "wave", "advection_diffusion", "steady_heat_conduction"}:
        return _future_patterns(pde, split)
    raise NotImplementedError(f"Lazy loading is not implemented for {pde}")


def _lazy_candidate_files(root: Path, pde: str, active_split: str, train_shards: int) -> list[Path]:
    aliases: tuple[str, ...] = ()
    if pde == "nsnonbounded":
        aliases = ("navier_stokes", "ns")
    files = _train_limited(_candidate_files(root, pde, active_split, _lazy_patterns(pde, active_split), aliases=aliases), active_split, train_shards)
    if pde == "nsnonbounded" and active_split == "test":
        files = _filter_nsnonbounded_test_files(files)
    return files


def _lazy_group_keys(path: Path, pde: str) -> list[str] | None:
    if pde not in {"reaction_diffusion", "shallow_water"}:
        return None
    with h5py.File(path, "r") as f:
        return sorted([str(k) for k in f.keys()], key=lambda x: int(x) if x.isdigit() else x)


def _lazy_dense_count(path: Path, pde: str) -> int:
    with h5py.File(path, "r") as f:
        if pde == "nsnonbounded":
            return int(f["w0"].shape[0])
        if pde in {"heat", "wave", "advection_diffusion", "steady_heat_conduction"}:
            return int(f["input_data"].shape[0])
    raise NotImplementedError(f"Lazy dense count is not implemented for {pde}")


def _read_lazy_raw_sample(
    pde: str,
    ref: LazySampleRef,
    split: str,
    scalar_param_mode: str,
    load_full_trajectory: bool,
    file_paths: list[str],
    dataset_metadata: dict[str, Any],
) -> dict[str, Any]:
    if pde == "nsnonbounded":
        raw = _read_lazy_ns_sample(ref, split, load_full_trajectory, file_paths)
    elif pde == "reaction_diffusion":
        raw = _read_lazy_reaction_diffusion_sample(ref, split, load_full_trajectory, file_paths)
    elif pde == "shallow_water":
        raw = _read_lazy_shallow_water_sample(ref, split, load_full_trajectory, file_paths)
    elif pde in {"heat", "wave", "advection_diffusion", "steady_heat_conduction"}:
        raw = _read_lazy_future_sample(ref, pde, split, scalar_param_mode, load_full_trajectory, file_paths)
    else:
        raise NotImplementedError(f"Lazy loading is not implemented for {pde}")
    raw["metadata"].update({k: v for k, v in dataset_metadata.items() if k not in raw["metadata"]})
    raw["metadata"].update(
        {
            "data_loading_mode": "lazy",
            "load_full_trajectory": bool(load_full_trajectory),
            "lazy_sample_ref": f"{ref.path.name}:{ref.key if ref.key is not None else ref.local_index}",
            "loaded_count": 1,
        }
    )
    return raw


def _finalize_lazy_raw(
    full: torch.Tensor,
    channel_names: list[str],
    input_names: list[str],
    target_names: list[str],
    split: str,
    ref: LazySampleRef,
    file_paths: list[str],
    metadata: dict[str, Any],
    pde_params: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    pde_params = pde_params or {}
    metadata = dict(metadata)
    metadata.update(
        {
            "files": list(file_paths),
            "split": split,
            "sample_indices": torch.tensor([ref.global_index], dtype=torch.long),
            "global_sample_ids": [f"{ref.path.name}:{ref.key if ref.key is not None else ref.local_index}"],
            "pde_params": pde_params,
            "pde_params_available": sorted(pde_params),
        }
    )
    return _finalize_loaded_raw(
        {
            "full_tensor": full.float(),
            "channel_names": channel_names,
            "input_channel_names": input_names,
            "target_channel_names": target_names,
            "metadata": metadata,
            "pde_params": pde_params,
            "split": split,
            "file_paths": list(file_paths),
            "sample_indices": metadata["sample_indices"],
            "global_sample_ids": metadata["global_sample_ids"],
        },
        max_samples=1,
        strict_size=True,
    )


def _read_lazy_ns_sample(ref: LazySampleRef, split: str, load_full_trajectory: bool, file_paths: list[str]) -> dict[str, Any]:
    with h5py.File(ref.path, "r") as f:
        w0 = f["w0"][ref.local_index : ref.local_index + 1]
        if load_full_trajectory:
            w = f["w"][ref.local_index : ref.local_index + 1]
            traj = np.concatenate([w0[:, None, :, :], np.moveaxis(w, -1, 1)], axis=1)
            full = _as_float_tensor(traj).unsqueeze(1)
            channels = ["w"]
            target_names = [f"w_t{i}" for i in range(1, full.shape[2])]
            meta = {"canonical_layout": "NCTHW", "time_values": [i / max(full.shape[2] - 1, 1) for i in range(full.shape[2])]}
        else:
            wT = f["w"][ref.local_index : ref.local_index + 1, :, :, -1]
            full = torch.stack((_as_float_tensor(w0), _as_float_tensor(wT)), dim=1)
            channels = ["w0", "wT"]
            target_names = ["wT"]
            meta = {"canonical_layout": "NCHW", "input_indices": [0], "target_indices": [1], "time_values": [0.0, 1.0]}
    meta.update({"final_time": 1.0, "nu": 1e-3, "loaded_full_trajectory": bool(load_full_trajectory)})
    return _finalize_lazy_raw(full, channels, ["w0"], target_names, split, ref, file_paths, meta)


def _read_lazy_reaction_diffusion_sample(ref: LazySampleRef, split: str, load_full_trajectory: bool, file_paths: list[str]) -> dict[str, Any]:
    with h5py.File(ref.path, "r") as f:
        if ref.key is None:
            raise KeyError(f"Lazy reaction_diffusion sample {ref} is missing an HDF5 group key")
        data = np.asarray(f[ref.key]["data"][:])
    input_idx = 50 if data.shape[0] > 50 else 0
    if load_full_trajectory:
        full = _as_float_tensor(np.moveaxis(data, -1, 0)).unsqueeze(0)
        channels = ["u", "v"]
        meta = {"canonical_layout": "NCTHW", "input_time_index": input_idx}
    else:
        endpoints = np.concatenate([data[input_idx], data[-1]], axis=-1)
        full = _as_float_tensor(np.moveaxis(endpoints, -1, 0)).unsqueeze(0)
        channels = ["u0", "v0", "uT", "vT"]
        meta = {"canonical_layout": "NCHW", "input_indices": [0, 1], "target_indices": [2, 3], "input_time_index": 0}
    is_test = _validate_split(split) == "test"
    meta.update(
        {
            "final_time": 5.0,
            "D_u": 2e-3 if is_test else 1e-3,
            "D_v": 4e-3 if is_test else 5e-3,
            "k": 3e-3 if is_test else 5e-3,
            "loaded_full_trajectory": bool(load_full_trajectory),
        }
    )
    return _finalize_lazy_raw(full, channels, ["u0", "v0"], ["uT", "vT"], split, ref, file_paths, meta)


def _read_lazy_shallow_water_sample(ref: LazySampleRef, split: str, load_full_trajectory: bool, file_paths: list[str]) -> dict[str, Any]:
    with h5py.File(ref.path, "r") as f:
        if ref.key is None:
            raise KeyError(f"Lazy shallow_water sample {ref} is missing an HDF5 group key")
        group = f[ref.key]["data"]
        arrays = []
        for field in ("h", "hu", "hv"):
            arr = np.asarray(group[field][:])
            if arr.ndim == 4 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            arrays.append(arr)
    if load_full_trajectory:
        full = _as_float_tensor(np.stack(arrays, axis=0)).unsqueeze(0)
        channels = ["h", "hu", "hv"]
        meta = {"canonical_layout": "NCTHW"}
    else:
        endpoints = [arr[0] for arr in arrays] + [arr[-1] for arr in arrays]
        full = _as_float_tensor(np.stack(endpoints, axis=0)).unsqueeze(0)
        channels = ["h0", "hu0", "hv0", "hT", "huT", "hvT"]
        meta = {"canonical_layout": "NCHW", "input_indices": [0, 1, 2], "target_indices": [3, 4, 5]}
    meta.update({"final_time": 1.0, "g": 1.0, "domain_length": 5.0, "loaded_full_trajectory": bool(load_full_trajectory)})
    return _finalize_lazy_raw(full, channels, ["h0", "hu0", "hv0"], ["hT", "huT", "hvT"], split, ref, file_paths, meta)


def _read_lazy_future_sample(
    ref: LazySampleRef,
    pde: str,
    split: str,
    scalar_param_mode: str,
    load_full_trajectory: bool,
    file_paths: list[str],
) -> dict[str, Any]:
    with h5py.File(ref.path, "r") as f:
        inp = _as_float_tensor(f["input_data"][ref.local_index : ref.local_index + 1])
        out = _as_float_tensor(f["output_data"][ref.local_index : ref.local_index + 1])
        attrs = _h5_attrs_to_python(f)
        meta: dict[str, Any] = {"canonical_layout": "NCHW", "scalar_param_mode": scalar_param_mode}
        for key, value in attrs.items():
            meta.setdefault(key, value)
        if "t" in f:
            meta.setdefault("time_values", np.asarray(f["t"][:], dtype=np.float32).tolist())
        if "T" in attrs:
            meta.setdefault("final_time", float(attrs["T"]))
        if "boundary_condition" in attrs:
            meta.setdefault("bc", str(attrs["boundary_condition"]))
        if load_full_trajectory and "full_trajectory" in f:
            full_traj = _as_float_tensor(f["full_trajectory"][ref.local_index : ref.local_index + 1])
            meta["full_trajectory"] = full_traj
            meta["full_trajectory_shape"] = tuple(full_traj.shape)

        pde_params: dict[str, torch.Tensor] = {}
        if pde == "heat":
            alpha = _scalar_or_attr(f, "alpha", "fixed_alpha", 1, ref.local_index, required=False)
            if alpha is not None:
                pde_params["alpha"] = alpha
                meta["alpha"] = alpha
            if scalar_param_mode == "materialize":
                alpha_field = _expand_scalar_to_field(alpha if alpha is not None else torch.full((1,), float("nan")), inp.shape[-2], inp.shape[-1])
                full = torch.cat([inp[:, :1], alpha_field, out[:, :1], alpha_field.clone()], dim=1)
                channels = ["u0", "alpha", "uT", "alpha_T"]
                input_names, target_names = ["u0", "alpha"], ["uT", "alpha_T"]
                meta.update({"input_indices": [0, 1], "target_indices": [2, 3]})
            else:
                full = torch.cat([inp[:, :1], out[:, :1]], dim=1)
                channels = ["u0", "uT"]
                input_names, target_names = ["u0"], ["uT"]
                meta.update({"input_indices": [0], "target_indices": [1]})
        elif pde == "wave":
            c = _scalar_or_attr(f, "c", "fixed_c", 1, ref.local_index, required=False)
            if c is not None:
                pde_params["c"] = c
                meta["c"] = c
            if scalar_param_mode == "materialize" and c is not None:
                c_field = _expand_scalar_to_field(c, inp.shape[-2], inp.shape[-1])
                full = torch.cat([inp[:, :2], c_field, out[:, :2], c_field.clone()], dim=1)
                channels = ["u0", "v0", "c", "uT", "vT", "c_T"]
                input_names, target_names = ["u0", "v0", "c"], ["uT", "vT", "c_T"]
                meta.update({"input_indices": [0, 1, 2], "target_indices": [3, 4, 5]})
            else:
                full = torch.cat([inp[:, :2], out[:, :2]], dim=1)
                channels = ["u0", "v0", "uT", "vT"]
                input_names, target_names = ["u0", "v0"], ["uT", "vT"]
                meta.update({"input_indices": [0, 1], "target_indices": [2, 3]})
        elif pde == "advection_diffusion":
            bx = _scalar_or_attr(f, "b_x", "b_x", 1, ref.local_index, required=True)
            by = _scalar_or_attr(f, "b_y", "b_y", 1, ref.local_index, required=True)
            kappa = _scalar_or_attr(f, "kappa", "kappa", 1, ref.local_index, required=True)
            pde_params.update({"b_x": bx, "b_y": by, "kappa": kappa})
            meta.update(pde_params)
            if scalar_param_mode == "materialize":
                bx_f = _expand_scalar_to_field(bx, inp.shape[-2], inp.shape[-1])
                by_f = _expand_scalar_to_field(by, inp.shape[-2], inp.shape[-1])
                k_f = _expand_scalar_to_field(kappa, inp.shape[-2], inp.shape[-1])
                full = torch.cat([inp[:, :1], bx_f, by_f, k_f, out[:, :1], bx_f.clone(), by_f.clone(), k_f.clone()], dim=1)
                channels = ["u0", "b_x", "b_y", "kappa", "uT", "b_x_T", "b_y_T", "kappa_T"]
                input_names, target_names = ["u0", "b_x", "b_y", "kappa"], ["uT", "b_x_T", "b_y_T", "kappa_T"]
                meta.update({"input_indices": [0, 1, 2, 3], "target_indices": [4, 5, 6, 7]})
            else:
                full = torch.cat([inp[:, :1], out[:, :1]], dim=1)
                channels = ["u0", "uT"]
                input_names, target_names = ["u0"], ["uT"]
                meta.update({"input_indices": [0], "target_indices": [1]})
        elif pde == "steady_heat_conduction":
            u_d = _scalar_or_attr(f, "u_D", "u_D", 1, ref.local_index, required=True)
            pde_params["u_D"] = u_d
            meta["u_D"] = u_d
            if scalar_param_mode == "materialize":
                u_d_f = _expand_scalar_to_field(u_d, inp.shape[-2], inp.shape[-1])
                full = torch.cat([inp[:, :1], u_d_f, out[:, :1], u_d_f.clone()], dim=1)
                channels = ["f", "u_D", "u", "u_D_T"]
                input_names, target_names = ["f", "u_D"], ["u", "u_D_T"]
                meta.update({"input_indices": [0, 1], "target_indices": [2, 3]})
            else:
                full = torch.cat([inp[:, :1], out[:, :1]], dim=1)
                channels = ["f", "u"]
                input_names, target_names = ["f"], ["u"]
                meta.update({"input_indices": [0], "target_indices": [1]})
        else:
            raise ValueError(f"Unsupported future PDE '{pde}'")
    meta["loaded_full_trajectory"] = isinstance(meta.get("full_trajectory"), torch.Tensor)
    meta["load_full_trajectory"] = bool(load_full_trajectory)
    return _finalize_lazy_raw(full, channels, input_names, target_names, split, ref, file_paths, meta, pde_params)


def _load_darcy(
    root: Path,
    split: str,
    max_samples: int | None,
    train_shards: int = 5,
    sample_offset: int = 0,
    val_from_train_offset: int | None = None,
    prefer_test: bool = False,
    scalar_param_mode: str = "metadata",
    load_full_trajectory: bool = True,
    strict_size: bool = False,
) -> dict[str, Any]:
    split = _validate_split(split)
    active_split = "test" if prefer_test else split
    if active_split == "test":
        patterns = ["darcy_test_*-128-128.mat", "darcy_*-128-128_test.mat"]
    elif active_split == "val":
        patterns = ["darcy_val_*-128-128.mat", "darcy_*-128-128_val.mat"]
    else:
        patterns = ["darcy_10000-128-128_*.mat"]
    files = _train_limited(_candidate_files(root, "darcy", active_split, patterns), active_split, train_shards)
    if not files:
        raise _missing_error(root, "darcy", active_split, patterns)
    parts = []
    remaining = max_samples
    offset = max(int(sample_offset), 0)
    global_ids: list[str] = []
    sample_indices: list[int] = []
    loaded_start = offset
    for path in files:
        with h5py.File(path, "r") as f:
            ds = f["thresh_a_data"]
            n_total = ds.shape[-1] if len(ds.shape) >= 3 and ds.shape[0] == ds.shape[1] else ds.shape[0]
            if offset >= n_total:
                offset -= n_total
                continue
            n = n_total - offset if remaining is None else min(remaining, n_total - offset)
            a = _h5_samples(f["thresh_a_data"], offset + n)[offset:]
            p = _h5_samples(f["thresh_p_data"], offset + n)[offset:]
        parts.append(torch.stack((_as_float_tensor(a), _as_float_tensor(p)), dim=1))
        sample_indices.extend(range(loaded_start, loaded_start + n))
        global_ids.extend(_file_sample_ids(path, offset, n))
        loaded_start += n
        offset = 0
        if remaining is not None:
            remaining -= n
            if remaining <= 0:
                break
    if not parts:
        raise _missing_error(root, "darcy", active_split, patterns)
    raw = {
        "full_tensor": torch.cat(parts, dim=0),
        "channel_names": ["a", "p"],
        "input_channel_names": ["a"],
        "target_channel_names": ["p"],
        "split": split,
        "file_paths": [str(p) for p in files],
        "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
        "global_sample_ids": global_ids,
        "metadata": {"files": [str(p) for p in files], "canonical_layout": "NCHW", "split": split},
    }
    return _finalize_loaded_raw(raw, max_samples, strict_size)


def _load_poisson(
    root: Path,
    split: str,
    max_samples: int | None,
    train_shards: int = 5,
    sample_offset: int = 0,
    val_from_train_offset: int | None = None,
    prefer_test: bool = False,
    scalar_param_mode: str = "metadata",
    load_full_trajectory: bool = True,
    strict_size: bool = False,
) -> dict[str, Any]:
    active_split = "test" if prefer_test else _validate_split(split)
    return _load_static_mat(
        root,
        "poisson",
        ("f_data", "phi_data"),
        _static_patterns("poisson", active_split),
        split,
        max_samples,
        ["f", "phi"],
        train_shards=train_shards,
        sample_offset=sample_offset,
        active_split=active_split,
        strict_size=strict_size,
    )


def _load_helmholtz(
    root: Path,
    split: str,
    max_samples: int | None,
    train_shards: int = 5,
    sample_offset: int = 0,
    val_from_train_offset: int | None = None,
    prefer_test: bool = False,
    scalar_param_mode: str = "metadata",
    load_full_trajectory: bool = True,
    strict_size: bool = False,
) -> dict[str, Any]:
    active_split = "test" if prefer_test else _validate_split(split)
    return _load_static_mat(
        root,
        "helmholtz",
        ("f_data", "psi_data"),
        _static_patterns("helmholtz", active_split),
        split,
        max_samples,
        ["f", "psi"],
        metadata={"k": 1.0},
        train_shards=train_shards,
        sample_offset=sample_offset,
        active_split=active_split,
        strict_size=strict_size,
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
    train_shards: int = 5,
    sample_offset: int = 0,
    active_split: str | None = None,
    strict_size: bool = False,
) -> dict[str, Any]:
    split = _validate_split(split)
    active_split = active_split or split
    files = _train_limited(_candidate_files(root, pde, active_split, patterns), active_split, train_shards)
    if not files:
        raise _missing_error(root, pde, active_split, patterns)
    parts = []
    remaining = max_samples
    offset = max(int(sample_offset), 0)
    global_ids: list[str] = []
    sample_indices: list[int] = []
    loaded_start = offset
    for path in files:
        raw = _safe_loadmat(path, keys)
        n_total = raw[keys[0]].shape[0]
        if offset >= n_total:
            offset -= n_total
            continue
        n = n_total - offset if remaining is None else min(remaining, n_total - offset)
        x = _as_float_tensor(raw[keys[0]][offset : offset + n])
        y = _as_float_tensor(raw[keys[1]][offset : offset + n])
        parts.append(torch.stack((x, y), dim=1))
        sample_indices.extend(range(loaded_start, loaded_start + n))
        global_ids.extend(_file_sample_ids(path, offset, n))
        loaded_start += n
        offset = 0
        if remaining is not None:
            remaining -= n
            if remaining <= 0:
                break
    if not parts:
        raise _missing_error(root, pde, active_split, patterns)
    meta = {"files": [str(p) for p in files], "canonical_layout": "NCHW"}
    if metadata:
        meta.update(metadata)
    meta.update({"split": split})
    raw = {
        "full_tensor": torch.cat(parts, dim=0),
        "channel_names": channel_names,
        "input_channel_names": [channel_names[0]],
        "target_channel_names": [channel_names[1]],
        "metadata": meta,
        "split": split,
        "file_paths": [str(p) for p in files],
        "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
        "global_sample_ids": global_ids,
    }
    return _finalize_loaded_raw(raw, max_samples, strict_size)


def _static_patterns(pde: str, split: str) -> list[str]:
    if split == "test":
        if pde == "helmholtz":
            return ["helmholtz_test_*-128-128*.mat", "helmholtz_*-128-128_test.mat"]
        return [f"{pde}_test_*-128-128.mat", f"{pde}_*-128-128_test.mat"]
    if split == "val":
        return [f"{pde}_val_*-128-128*.mat", f"{pde}_*-128-128_val.mat"]
    return [f"{pde}_10000-128-128_*.mat"]


def _load_nsnonbounded(
    root: Path,
    split: str,
    max_samples: int | None,
    train_shards: int = 5,
    sample_offset: int = 0,
    val_from_train_offset: int | None = None,
    prefer_test: bool = False,
    scalar_param_mode: str = "metadata",
    load_full_trajectory: bool = True,
    strict_size: bool = False,
) -> dict[str, Any]:
    split = _validate_split(split)
    active_split = "test" if prefer_test else split
    if active_split == "test":
        patterns = [
            "nsnonbounded_test_*-128-128-10*.mat",
            "nsnonbounded_1000-128-128-10*.mat",
            "nsnonbounded_10000-128-128-10_test*.mat",
            "nsnonbounded_*_test*.mat",
        ]
    elif active_split == "val":
        patterns = ["nsnonbounded_val_*-128-128-10_*.mat", "nsnonbounded_*-128-128-10_val*.mat"]
    else:
        patterns = ["nsnonbounded_10000-128-128-10_*_new.mat"]
    files = _train_limited(_candidate_files(root, "nsnonbounded", active_split, patterns), active_split, train_shards)
    if active_split == "test":
        files = _filter_nsnonbounded_test_files(files)
    if not files:
        raise _missing_error(root, "nsnonbounded", active_split, patterns)
    parts = []
    remaining = max_samples
    offset = max(int(sample_offset), 0)
    global_ids: list[str] = []
    sample_indices: list[int] = []
    loaded_start = offset
    for path in files:
        with h5py.File(path, "r") as f:
            n_total = f["w0"].shape[0]
            if offset >= n_total:
                offset -= n_total
                continue
            n = n_total - offset if remaining is None else min(remaining, n_total - offset)
            w0 = f["w0"][offset : offset + n]
            if load_full_trajectory:
                w = f["w"][offset : offset + n]
                traj = np.concatenate([w0[:, None, :, :], np.moveaxis(w, -1, 1)], axis=1)
                parts.append(_as_float_tensor(traj).unsqueeze(1))
            else:
                wT = f["w"][offset : offset + n, :, :, -1]
                parts.append(torch.stack((_as_float_tensor(w0), _as_float_tensor(wT)), dim=1))
        sample_indices.extend(range(loaded_start, loaded_start + n))
        global_ids.extend(_file_sample_ids(path, offset, n))
        loaded_start += n
        offset = 0
        if remaining is not None:
            remaining -= n
            if remaining <= 0:
                break
    if not parts:
        raise _missing_error(root, "nsnonbounded", active_split, patterns)
    raw = {
        "full_tensor": torch.cat(parts, dim=0),
        "channel_names": ["w"] if load_full_trajectory else ["w0", "wT"],
        "input_channel_names": ["w0"],
        "target_channel_names": [f"w_t{i}" for i in range(1, 11)] if load_full_trajectory else ["wT"],
        "split": split,
        "file_paths": [str(p) for p in files],
        "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
        "global_sample_ids": global_ids,
        "metadata": {
            "files": [str(p) for p in files],
            "canonical_layout": "NCTHW" if load_full_trajectory else "NCHW",
            "time_values": [i / 10 for i in range(11)],
            "final_time": 1.0,
            "nu": 1e-3,
            "split": split,
            "input_indices": [0] if not load_full_trajectory else None,
            "target_indices": [1] if not load_full_trajectory else None,
            "load_full_trajectory": bool(load_full_trajectory),
            "loaded_full_trajectory": bool(load_full_trajectory),
        },
    }
    return _finalize_loaded_raw(raw, max_samples, strict_size)


def _load_burger(
    root: Path,
    split: str,
    max_samples: int | None,
    train_shards: int = 5,
    sample_offset: int = 0,
    val_from_train_offset: int | None = None,
    prefer_test: bool = False,
    scalar_param_mode: str = "metadata",
    load_full_trajectory: bool = True,
    strict_size: bool = False,
) -> dict[str, Any]:
    split = _validate_split(split)
    active_split = "test" if prefer_test else split
    if active_split == "test":
        patterns = ["burger_test_*-128-128.mat", "burger_*-128-128_test.mat"]
    elif active_split == "val":
        patterns = ["burger_val_*-128-128.mat", "burger_*-128-128_val.mat"]
    else:
        patterns = ["burger_10000-128-128_*.mat"]
    files = _train_limited(_candidate_files(root, "burger", active_split, patterns, aliases=("burgers",)), active_split, train_shards)
    if not files:
        raise _missing_error(root, "burger", active_split, patterns, aliases=("burgers",))
    parts = []
    initials = []
    remaining = max_samples
    offset = max(int(sample_offset), 0)
    global_ids: list[str] = []
    sample_indices: list[int] = []
    loaded_start = offset
    for path in files:
        raw = _safe_loadmat(path, ("input", "output"))
        n_total = raw["output"].shape[0]
        if offset >= n_total:
            offset -= n_total
            continue
        n = n_total - offset if remaining is None else min(remaining, n_total - offset)
        parts.append(_as_float_tensor(raw["output"][offset : offset + n]).unsqueeze(1))
        if "input" in raw:
            initials.append(_as_float_tensor(raw["input"][offset : offset + n]))
        sample_indices.extend(range(loaded_start, loaded_start + n))
        global_ids.extend(_file_sample_ids(path, offset, n))
        loaded_start += n
        offset = 0
        if remaining is not None:
            remaining -= n
            if remaining <= 0:
                break
    if not parts:
        raise _missing_error(root, "burger", active_split, patterns, aliases=("burgers",))
    t_steps = parts[0].shape[-2] if parts else 0
    meta = {
        "files": [str(p) for p in files],
        "canonical_layout": "NCTX",
        "axes": ["time", "x"],
        "time_values": [i / max(t_steps - 1, 1) for i in range(t_steps)],
        "final_time": 1.0,
        "nu": 0.01,
        "split": split,
    }
    if initials:
        meta["initial_1d"] = torch.cat(initials, dim=0)
    raw = {
        "full_tensor": torch.cat(parts, dim=0),
        "channel_names": ["u"],
        "input_channel_names": ["u0"],
        "target_channel_names": ["u"],
        "metadata": meta,
        "split": split,
        "file_paths": [str(p) for p in files],
        "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
        "global_sample_ids": global_ids,
    }
    return _finalize_loaded_raw(raw, max_samples, strict_size)


def _load_reaction_diffusion(
    root: Path,
    split: str,
    max_samples: int | None,
    train_shards: int = 5,
    sample_offset: int = 0,
    val_from_train_offset: int | None = None,
    prefer_test: bool = False,
    scalar_param_mode: str = "metadata",
    load_full_trajectory: bool = True,
    strict_size: bool = False,
) -> dict[str, Any]:
    split = _validate_split(split)
    active_split = "test" if prefer_test else split
    if active_split == "test":
        patterns = ["reaction_diffusion_test_*-128-128-*.h5"]
    elif active_split == "val":
        patterns = ["reaction_diffusion_val_*-128-128-*.h5", "reaction_diffusion-128-128-*_val*.h5"]
    else:
        patterns = ["reaction_diffusion-128-128-*_*.h5", "2D_diff-react_NA_NA.h5"]
    files = _train_limited(_candidate_files(root, "reaction_diffusion", active_split, patterns), active_split, train_shards)
    if not files:
        raise _missing_error(root, "reaction_diffusion", active_split, patterns)
    parts = []
    remaining = max_samples
    skip = max(int(sample_offset), 0)
    sample_indices: list[int] = []
    global_ids: list[str] = []
    seen = 0
    for path in files:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            for key in keys:
                if skip > 0:
                    skip -= 1
                    seen += 1
                    continue
                data = np.asarray(f[key]["data"][:])
                # raw [T,H,W,2] -> canonical [2,T,H,W]
                if load_full_trajectory:
                    parts.append(_as_float_tensor(np.moveaxis(data, -1, 0)))
                else:
                    input_idx_local = 50 if data.shape[0] > 50 else 0
                    endpoints = np.concatenate([data[input_idx_local], data[-1]], axis=-1)
                    parts.append(_as_float_tensor(np.moveaxis(endpoints, -1, 0)))
                sample_indices.append(seen)
                global_ids.append(f"{path.name}:{key}")
                seen += 1
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        break
        if remaining is not None and remaining <= 0:
            break
    if not parts:
        raise _missing_error(root, "reaction_diffusion", active_split, patterns)
    full = torch.stack(parts, dim=0)
    input_idx = 50 if load_full_trajectory and full.shape[2] > 50 else 0
    is_test = active_split == "test"
    raw = {
        "full_tensor": full,
        "channel_names": ["u", "v"] if load_full_trajectory else ["u0", "v0", "uT", "vT"],
        "input_channel_names": ["u0", "v0"],
        "target_channel_names": ["uT", "vT"],
        "split": split,
        "file_paths": [str(p) for p in files],
        "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
        "global_sample_ids": global_ids,
        "metadata": {
            "files": [str(p) for p in files],
            "canonical_layout": "NCTHW" if load_full_trajectory else "NCHW",
            "input_time_index": input_idx,
            "input_indices": [0, 1] if not load_full_trajectory else None,
            "target_indices": [2, 3] if not load_full_trajectory else None,
            "final_time": 5.0,
            "D_u": 2e-3 if is_test else 1e-3,
            "D_v": 4e-3 if is_test else 5e-3,
            "k": 3e-3 if is_test else 5e-3,
            "split": split,
            "load_full_trajectory": bool(load_full_trajectory),
            "loaded_full_trajectory": bool(load_full_trajectory),
        },
    }
    return _finalize_loaded_raw(raw, max_samples, strict_size)


def _load_shallow_water(
    root: Path,
    split: str,
    max_samples: int | None,
    train_shards: int = 5,
    sample_offset: int = 0,
    val_from_train_offset: int | None = None,
    prefer_test: bool = False,
    scalar_param_mode: str = "metadata",
    load_full_trajectory: bool = True,
    strict_size: bool = False,
) -> dict[str, Any]:
    split = _validate_split(split)
    active_split = "test" if prefer_test else split
    if active_split == "test":
        patterns = ["swe_test_*-128-128-*.h5", "2d_swe_test*.h5"]
    elif active_split == "val":
        patterns = ["swe_val_*-128-128-*.h5", "2d_swe_val*.h5"]
    else:
        patterns = ["2d_swe_128_128_10_*.h5"]
    files = _train_limited(_candidate_files(root, "shallow_water", active_split, patterns), active_split, train_shards)
    if not files:
        raise _missing_error(root, "shallow_water", active_split, patterns)
    parts = []
    remaining = max_samples
    skip = max(int(sample_offset), 0)
    sample_indices: list[int] = []
    global_ids: list[str] = []
    seen = 0
    for path in files:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            for key in keys:
                if skip > 0:
                    skip -= 1
                    seen += 1
                    continue
                group = f[key]["data"]
                channels = []
                for field in ("h", "hu", "hv"):
                    arr = np.asarray(group[field][:])
                    if arr.ndim == 4 and arr.shape[-1] == 1:
                        arr = arr[..., 0]
                    channels.append(arr)
                if load_full_trajectory:
                    parts.append(_as_float_tensor(np.stack(channels, axis=0)))
                else:
                    endpoints = [arr[0] for arr in channels] + [arr[-1] for arr in channels]
                    parts.append(_as_float_tensor(np.stack(endpoints, axis=0)))
                sample_indices.append(seen)
                global_ids.append(f"{path.name}:{key}")
                seen += 1
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        break
        if remaining is not None and remaining <= 0:
            break
    if not parts:
        raise _missing_error(root, "shallow_water", active_split, patterns)
    raw = {
        "full_tensor": torch.stack(parts, dim=0),
        "channel_names": ["h", "hu", "hv"] if load_full_trajectory else ["h0", "hu0", "hv0", "hT", "huT", "hvT"],
        "input_channel_names": ["h0", "hu0", "hv0"],
        "target_channel_names": ["hT", "huT", "hvT"],
        "split": split,
        "file_paths": [str(p) for p in files],
        "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
        "global_sample_ids": global_ids,
        "metadata": {
            "files": [str(p) for p in files],
            "canonical_layout": "NCTHW" if load_full_trajectory else "NCHW",
            "input_indices": [0, 1, 2] if not load_full_trajectory else None,
            "target_indices": [3, 4, 5] if not load_full_trajectory else None,
            "final_time": 1.0,
            "g": 1.0,
            "domain_length": 5.0,
            "split": split,
            "load_full_trajectory": bool(load_full_trajectory),
            "loaded_full_trajectory": bool(load_full_trajectory),
        },
    }
    return _finalize_loaded_raw(raw, max_samples, strict_size)


def _load_heat(
    root: Path,
    split: str,
    max_samples: int | None,
    train_shards: int = 5,
    sample_offset: int = 0,
    val_from_train_offset: int | None = None,
    prefer_test: bool = False,
    scalar_param_mode: str = "metadata",
    load_full_trajectory: bool = True,
    strict_size: bool = False,
) -> dict[str, Any]:
    split = _validate_split(split)
    active_split = "test" if prefer_test else split
    files = _candidate_files(
        root,
        "heat",
        active_split,
        _future_patterns("heat", active_split),
    )
    files = _train_limited(files, active_split, train_shards)
    if not files:
        raise _missing_error(root, "heat", active_split, _future_patterns("heat", active_split))
    return _load_future_field_h5(files, max_samples, "heat", split, scalar_param_mode, sample_offset, strict_size, load_full_trajectory)


def _load_wave(
    root: Path,
    split: str,
    max_samples: int | None,
    train_shards: int = 5,
    sample_offset: int = 0,
    val_from_train_offset: int | None = None,
    prefer_test: bool = False,
    scalar_param_mode: str = "metadata",
    load_full_trajectory: bool = True,
    strict_size: bool = False,
) -> dict[str, Any]:
    split = _validate_split(split)
    active_split = "test" if prefer_test else split
    files = _candidate_files(
        root,
        "wave",
        active_split,
        _future_patterns("wave", active_split),
    )
    files = _train_limited(files, active_split, train_shards)
    if not files:
        raise _missing_error(root, "wave", active_split, _future_patterns("wave", active_split))
    return _load_future_field_h5(files, max_samples, "wave", split, scalar_param_mode, sample_offset, strict_size, load_full_trajectory)


def _load_advection_diffusion(
    root: Path,
    split: str,
    max_samples: int | None,
    train_shards: int = 5,
    sample_offset: int = 0,
    val_from_train_offset: int | None = None,
    prefer_test: bool = False,
    scalar_param_mode: str = "metadata",
    load_full_trajectory: bool = True,
    strict_size: bool = False,
) -> dict[str, Any]:
    split = _validate_split(split)
    active_split = "test" if prefer_test else split
    files = _candidate_files(
        root,
        "advection_diffusion",
        active_split,
        _future_patterns("advection_diffusion", active_split),
    )
    files = _train_limited(files, active_split, train_shards)
    if not files:
        raise _missing_error(root, "advection_diffusion", active_split, _future_patterns("advection_diffusion", active_split))
    return _load_future_field_h5(files, max_samples, "advection_diffusion", split, scalar_param_mode, sample_offset, strict_size, load_full_trajectory)


def _load_steady_heat_conduction(
    root: Path,
    split: str,
    max_samples: int | None,
    train_shards: int = 5,
    sample_offset: int = 0,
    val_from_train_offset: int | None = None,
    prefer_test: bool = False,
    scalar_param_mode: str = "metadata",
    load_full_trajectory: bool = True,
    strict_size: bool = False,
) -> dict[str, Any]:
    split = _validate_split(split)
    active_split = "test" if prefer_test else split
    patterns = _future_patterns("steady_heat_conduction", active_split)
    files = _candidate_files(
        root,
        "steady_heat_conduction",
        active_split,
        patterns,
    )
    files = _train_limited(files, active_split, train_shards)
    if not files:
        raise _missing_error(root, "steady_heat_conduction", active_split, patterns)
    return _load_future_field_h5(files, max_samples, "steady_heat_conduction", split, scalar_param_mode, sample_offset, strict_size, load_full_trajectory)


def _future_patterns(pde: str, split: str) -> list[str]:
    if split == "test":
        return [f"{pde}_test*.h5"]
    if split == "val":
        return [f"{pde}_val*.h5"]
    return [f"{pde}_10000-128-128_*.h5", f"{pde}_*.h5"]


def _load_future_field_h5(
    files: list[Path],
    max_samples: int | None,
    pde: str,
    split: str,
    scalar_param_mode: str,
    sample_offset: int = 0,
    strict_size: bool = False,
    load_full_trajectory: bool = True,
) -> dict[str, Any]:
    parts: list[torch.Tensor] = []
    scalar_meta: dict[str, list[torch.Tensor]] = {}
    diagnostic_scalar_meta: dict[str, list[torch.Tensor]] = {}
    array_meta: dict[str, list[torch.Tensor]] = {}
    full_traj_parts: list[torch.Tensor] = []
    meta: dict[str, Any] = {"files": [str(p) for p in files], "canonical_layout": "NCHW", "split": split, "scalar_param_mode": scalar_param_mode}
    remaining = max_samples
    skip = max(int(sample_offset), 0)
    channel_names: list[str]
    input_names: list[str]
    target_names: list[str]
    sample_indices: list[int] = []
    global_ids: list[str] = []
    seen = 0
    for path in files:
        try:
            with h5py.File(path, "r") as f:
                if "input_data" not in f or "output_data" not in f:
                    raise KeyError(f"{path} must contain input_data and output_data")
                n_total = int(f["input_data"].shape[0])
                if skip >= n_total:
                    skip -= n_total
                    seen += n_total
                    continue
                local_start = skip
                n = n_total - local_start if remaining is None else min(remaining, n_total - local_start)
                inp = _as_float_tensor(f["input_data"][local_start : local_start + n])
                out = _as_float_tensor(f["output_data"][local_start : local_start + n])
                attrs = _h5_attrs_to_python(f)
                for key, value in attrs.items():
                    meta.setdefault(key, value)
                if "t" in f:
                    meta.setdefault("time_values", np.asarray(f["t"][:], dtype=np.float32).tolist())
                if "T" in attrs:
                    meta.setdefault("final_time", float(attrs["T"]))
                if "boundary_condition" in attrs:
                    meta.setdefault("bc", str(attrs["boundary_condition"]))
                if load_full_trajectory and "full_trajectory" in f:
                    full_traj_parts.append(_as_float_tensor(f["full_trajectory"][local_start : local_start + n]))

                if pde == "heat":
                    alpha = _scalar_or_attr(f, "alpha", "fixed_alpha", n, local_start, required=False)
                    if alpha is not None:
                        scalar_meta.setdefault("alpha", []).append(alpha)
                    if scalar_param_mode == "materialize":
                        alpha_field = _expand_scalar_to_field(alpha if alpha is not None else torch.full((n,), float("nan")), inp.shape[-2], inp.shape[-1])
                        parts.append(torch.cat([inp[:, :1], alpha_field, out[:, :1], alpha_field.clone()], dim=1))
                        channel_names = ["u0", "alpha", "uT", "alpha_T"]
                        input_names, target_names = ["u0", "alpha"], ["uT", "alpha_T"]
                        meta.update({"input_indices": [0, 1], "target_indices": [2, 3]})
                    else:
                        parts.append(torch.cat([inp[:, :1], out[:, :1]], dim=1))
                        channel_names = ["u0", "uT"]
                        input_names, target_names = ["u0"], ["uT"]
                        meta.update({"input_indices": [0], "target_indices": [1]})
                elif pde == "wave":
                    c = _scalar_or_attr(f, "c", "fixed_c", n, local_start, required=False)
                    if c is not None:
                        scalar_meta.setdefault("c", []).append(c)
                        meta.setdefault("fixed_c", float(c[0]) if torch.allclose(c, c[:1].expand_as(c)) else None)
                    elif "fixed_c" in attrs:
                        meta.setdefault("fixed_c", float(attrs["fixed_c"]))
                    if scalar_param_mode == "materialize" and c is not None:
                        c_field = _expand_scalar_to_field(c, inp.shape[-2], inp.shape[-1])
                        parts.append(torch.cat([inp[:, :2], c_field, out[:, :2], c_field.clone()], dim=1))
                        channel_names = ["u0", "v0", "c", "uT", "vT", "c_T"]
                        input_names, target_names = ["u0", "v0", "c"], ["uT", "vT", "c_T"]
                        meta.update({"input_indices": [0, 1, 2], "target_indices": [3, 4, 5]})
                    else:
                        parts.append(torch.cat([inp[:, :2], out[:, :2]], dim=1))
                        channel_names = ["u0", "v0", "uT", "vT"]
                        input_names, target_names = ["u0", "v0"], ["uT", "vT"]
                        meta.update({"input_indices": [0, 1], "target_indices": [2, 3]})
                elif pde == "advection_diffusion":
                    bx = _scalar_or_attr(f, "b_x", "b_x", n, local_start, required=True)
                    by = _scalar_or_attr(f, "b_y", "b_y", n, local_start, required=True)
                    kappa = _scalar_or_attr(f, "kappa", "kappa", n, local_start, required=True)
                    scalar_meta.setdefault("b_x", []).append(bx)
                    scalar_meta.setdefault("b_y", []).append(by)
                    scalar_meta.setdefault("kappa", []).append(kappa)
                    if scalar_param_mode == "materialize":
                        bx_f = _expand_scalar_to_field(bx, inp.shape[-2], inp.shape[-1])
                        by_f = _expand_scalar_to_field(by, inp.shape[-2], inp.shape[-1])
                        k_f = _expand_scalar_to_field(kappa, inp.shape[-2], inp.shape[-1])
                        parts.append(torch.cat([inp[:, :1], bx_f, by_f, k_f, out[:, :1], bx_f.clone(), by_f.clone(), k_f.clone()], dim=1))
                        channel_names = ["u0", "b_x", "b_y", "kappa", "uT", "b_x_T", "b_y_T", "kappa_T"]
                        input_names, target_names = ["u0", "b_x", "b_y", "kappa"], ["uT", "b_x_T", "b_y_T", "kappa_T"]
                        meta.update({"input_indices": [0, 1, 2, 3], "target_indices": [4, 5, 6, 7]})
                    else:
                        parts.append(torch.cat([inp[:, :1], out[:, :1]], dim=1))
                        channel_names = ["u0", "uT"]
                        input_names, target_names = ["u0"], ["uT"]
                        meta.update({"input_indices": [0], "target_indices": [1]})
                elif pde == "steady_heat_conduction":
                    u_d = _scalar_or_attr(f, "u_D", "u_D", n, local_start, required=True)
                    scalar_meta.setdefault("u_D", []).append(u_d)
                    for extra_key in ("picard_iters", "converged", "residual_norm", "n_sources"):
                        extra = _scalar_or_attr(f, extra_key, extra_key, n, local_start, required=False)
                        if extra is not None:
                            diagnostic_scalar_meta.setdefault(extra_key, []).append(extra)
                    for array_key in ("source_x", "source_y", "source_amp", "source_sigma"):
                        extra_array = _array_dataset_or_attr(f, array_key, n, local_start, required=False)
                        if extra_array is not None:
                            array_meta.setdefault(array_key, []).append(extra_array)
                    if scalar_param_mode == "materialize":
                        u_d_f = _expand_scalar_to_field(u_d, inp.shape[-2], inp.shape[-1])
                        parts.append(torch.cat([inp[:, :1], u_d_f, out[:, :1], u_d_f.clone()], dim=1))
                        channel_names = ["f", "u_D", "u", "u_D_T"]
                        input_names, target_names = ["f", "u_D"], ["u", "u_D_T"]
                        meta.update({"input_indices": [0, 1], "target_indices": [2, 3]})
                    else:
                        parts.append(torch.cat([inp[:, :1], out[:, :1]], dim=1))
                        channel_names = ["f", "u"]
                        input_names, target_names = ["f"], ["u"]
                        meta.update({"input_indices": [0], "target_indices": [1]})
                else:
                    raise ValueError(f"Unsupported future PDE '{pde}'")
                sample_indices.extend(range(seen + local_start, seen + local_start + n))
                global_ids.extend(_file_sample_ids(path, local_start, n))
                skip = 0
        except OSError as exc:
            raise OSError(f"Could not read {pde} HDF5 file {path}: {exc}") from exc

        if remaining is not None:
            remaining -= n
            if remaining <= 0:
                break
        seen += n_total

    if not parts:
        raise FileNotFoundError(f"No samples loaded for PDE '{pde}' from files: {[str(p) for p in files]}")
    pde_params: dict[str, torch.Tensor] = {}
    for key, values in scalar_meta.items():
        meta[key] = torch.cat(values, dim=0)
        pde_params[key] = meta[key]
    for key, values in diagnostic_scalar_meta.items():
        meta[key] = torch.cat(values, dim=0)
    if array_meta:
        meta["source_params"] = {key: torch.cat(values, dim=0) for key, values in array_meta.items()}
    meta["pde_params"] = pde_params
    meta["pde_params_available"] = sorted(pde_params)
    if full_traj_parts:
        full_traj = torch.cat(full_traj_parts, dim=0)
        meta["full_trajectory"] = full_traj
        meta["full_trajectory_shape"] = tuple(full_traj.shape)
    meta["load_full_trajectory"] = bool(load_full_trajectory)
    meta["loaded_full_trajectory"] = bool(full_traj_parts)
    raw = {
        "full_tensor": torch.cat(parts, dim=0),
        "channel_names": channel_names,
        "input_channel_names": input_names,
        "target_channel_names": target_names,
        "metadata": meta,
        "pde_params": pde_params,
        "split": split,
        "file_paths": [str(p) for p in files],
        "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
        "global_sample_ids": global_ids,
    }
    return _finalize_loaded_raw(raw, max_samples, strict_size)


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


def _scalar_or_attr(
    f: h5py.File,
    dataset_key: str,
    attr_key: str,
    n: int,
    local_start: int,
    required: bool,
) -> torch.Tensor | None:
    if dataset_key in f:
        dataset = f[dataset_key]
        if dataset.shape == ():
            return torch.full((n,), float(dataset[()]), dtype=torch.float32)
        return _as_float_tensor(dataset[local_start : local_start + n]).reshape(n)
    if attr_key in f.attrs:
        return torch.full((n,), float(f.attrs[attr_key]), dtype=torch.float32)
    if required:
        raise KeyError(f"Missing required scalar dataset or attr {dataset_key!r}/{attr_key!r} in {f.filename}")
    return None


def _array_dataset_or_attr(
    f: h5py.File,
    key: str,
    n: int,
    local_start: int,
    required: bool = False,
) -> torch.Tensor | None:
    if key in f:
        dataset = f[key]
        if dataset.shape == ():
            return torch.full((n,), float(dataset[()]), dtype=torch.float32)
        return _as_float_tensor(dataset[local_start : local_start + n])
    if key in f.attrs:
        value = np.asarray(f.attrs[key])
        if value.ndim == 0:
            return torch.full((n,), float(value), dtype=torch.float32)
        if value.shape[0] >= local_start + n:
            return _as_float_tensor(value[local_start : local_start + n])
        return _as_float_tensor(np.broadcast_to(value, (n, *value.shape)).copy())
    if required:
        raise KeyError(f"Missing required array dataset or attr {key!r} in {f.filename}")
    return None


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


def load_raw_split(
    pde: str,
    data_root: str | Path,
    split: str,
    max_samples: int | None = None,
    train_shards: int = 5,
    prefer_test: bool = False,
    scalar_param_mode: str = "metadata",
    load_full_trajectory: bool = True,
    strict_size: bool = False,
    sample_offset: int = 0,
    val_from_train_offset: int | None = None,
) -> dict[str, Any]:
    """Load one no-leakage split using the canonical baseline adapter.

    ``split`` is restricted to ``train``, ``val``, or ``test``. Validation
    falls back to a deterministic train subset when no independent val file is
    present; test files are never used for train or validation.
    """

    return build_default_registry().load_raw(
        pde,
        data_root,
        split=split,
        max_samples=max_samples,
        train_shards=train_shards,
        sample_offset=sample_offset,
        val_from_train_offset=val_from_train_offset,
        prefer_test=prefer_test and split == "test",
        scalar_param_mode=scalar_param_mode,
        load_full_trajectory=load_full_trajectory,
        strict_size=strict_size,
    )
