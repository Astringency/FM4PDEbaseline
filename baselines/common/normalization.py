from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from baselines.common.data_adapter import PDEBatch


@dataclass
class NormalizationStats:
    input_mean: torch.Tensor
    input_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    num_batches: int
    num_samples: int
    eps: float = 1e-6

    def to(self, device: torch.device | str) -> "NormalizationStats":
        return NormalizationStats(
            input_mean=self.input_mean.to(device),
            input_std=self.input_std.to(device),
            target_mean=self.target_mean.to(device),
            target_std=self.target_std.to(device),
            input_shape=self.input_shape,
            target_shape=self.target_shape,
            num_batches=self.num_batches,
            num_samples=self.num_samples,
            eps=self.eps,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "input_mean": self.input_mean.detach().cpu(),
            "input_std": self.input_std.detach().cpu(),
            "target_mean": self.target_mean.detach().cpu(),
            "target_std": self.target_std.detach().cpu(),
            "input_shape": list(self.input_shape),
            "target_shape": list(self.target_shape),
            "num_batches": int(self.num_batches),
            "num_samples": int(self.num_samples),
            "eps": float(self.eps),
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any] | None) -> "NormalizationStats | None":
        if not payload:
            return None
        return cls(
            input_mean=torch.as_tensor(payload["input_mean"], dtype=torch.float32),
            input_std=torch.as_tensor(payload["input_std"], dtype=torch.float32),
            target_mean=torch.as_tensor(payload["target_mean"], dtype=torch.float32),
            target_std=torch.as_tensor(payload["target_std"], dtype=torch.float32),
            input_shape=tuple(int(x) for x in payload.get("input_shape", ())),
            target_shape=tuple(int(x) for x in payload.get("target_shape", ())),
            num_batches=int(payload.get("num_batches", 0)),
            num_samples=int(payload.get("num_samples", 0)),
            eps=float(payload.get("eps", 1e-6)),
        )

    def json_summary(self) -> dict[str, Any]:
        return {
            "input_mean": [float(x) for x in self.input_mean.detach().cpu().reshape(-1)],
            "input_std": [float(x) for x in self.input_std.detach().cpu().reshape(-1)],
            "target_mean": [float(x) for x in self.target_mean.detach().cpu().reshape(-1)],
            "target_std": [float(x) for x in self.target_std.detach().cpu().reshape(-1)],
            "input_shape": list(self.input_shape),
            "target_shape": list(self.target_shape),
            "num_batches": int(self.num_batches),
            "num_samples": int(self.num_samples),
            "eps": float(self.eps),
        }


def estimate_normalization_stats(loader, max_batches: int | None = None, eps: float = 1e-6) -> NormalizationStats:
    input_sum = input_sumsq = target_sum = target_sumsq = None
    input_count = target_count = 0
    num_batches = 0
    num_samples = 0
    input_shape: tuple[int, ...] = ()
    target_shape: tuple[int, ...] = ()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= int(max_batches):
                break
            x = batch.input_fields.detach().float().cpu()
            y = batch.target_fields.detach().float().cpu()
            if x.ndim < 2 or y.ndim < 2:
                raise ValueError(f"Expected input/target tensors with channel dimension, got {tuple(x.shape)} and {tuple(y.shape)}")
            input_shape = tuple(x.shape)
            target_shape = tuple(y.shape)
            reduce_dims_x = tuple(i for i in range(x.ndim) if i != 1)
            reduce_dims_y = tuple(i for i in range(y.ndim) if i != 1)
            x_sum = x.sum(dim=reduce_dims_x)
            x_sumsq = x.square().sum(dim=reduce_dims_x)
            y_sum = y.sum(dim=reduce_dims_y)
            y_sumsq = y.square().sum(dim=reduce_dims_y)
            input_sum = x_sum if input_sum is None else input_sum + x_sum
            input_sumsq = x_sumsq if input_sumsq is None else input_sumsq + x_sumsq
            target_sum = y_sum if target_sum is None else target_sum + y_sum
            target_sumsq = y_sumsq if target_sumsq is None else target_sumsq + y_sumsq
            input_count += int(x.numel() // max(x.shape[1], 1))
            target_count += int(y.numel() // max(y.shape[1], 1))
            num_batches += 1
            num_samples += int(x.shape[0])
    if num_batches == 0 or input_sum is None or target_sum is None or input_sumsq is None or target_sumsq is None:
        raise ValueError("Cannot estimate normalization stats from an empty loader")
    input_mean = input_sum / max(input_count, 1)
    target_mean = target_sum / max(target_count, 1)
    input_var = (input_sumsq / max(input_count, 1) - input_mean.square()).clamp_min(float(eps) ** 2)
    target_var = (target_sumsq / max(target_count, 1) - target_mean.square()).clamp_min(float(eps) ** 2)
    return NormalizationStats(
        input_mean=input_mean,
        input_std=input_var.sqrt(),
        target_mean=target_mean,
        target_std=target_var.sqrt(),
        input_shape=input_shape,
        target_shape=target_shape,
        num_batches=num_batches,
        num_samples=num_samples,
        eps=float(eps),
    )


def normalize_batch_input_target(batch: PDEBatch, stats: NormalizationStats) -> PDEBatch:
    stats = stats.to(batch.input_fields.device)
    input_fields = _normalize_channels(batch.input_fields, stats.input_mean, stats.input_std)
    target_fields = _normalize_channels(batch.target_fields, stats.target_mean.to(batch.target_fields.device), stats.target_std.to(batch.target_fields.device))
    metadata = dict(batch.metadata)
    normalized_sources: dict[str, str] = {}
    for key in ("masked_grid", "voronoi_grid", "original_input_fields", "observed_solution_fields", "observation_source_fields"):
        value = metadata.get(key)
        if isinstance(value, torch.Tensor):
            source = _metadata_normalization_source(batch, key, value, stats)
            if source == "input":
                metadata[key] = _normalize_channels(value, stats.input_mean.to(value.device), stats.input_std.to(value.device))
                normalized_sources[key] = "input"
            elif source == "target":
                metadata[key] = _normalize_channels(value, stats.target_mean.to(value.device), stats.target_std.to(value.device))
                normalized_sources[key] = "target"
    obs_values = batch.obs_values
    if isinstance(obs_values, torch.Tensor):
        obs_source = _obs_value_normalization_source(batch)
        if obs_source == "input":
            obs_values = _normalize_last_channel(obs_values, stats.input_mean.to(obs_values.device), stats.input_std.to(obs_values.device))
        else:
            obs_values = _normalize_last_channel(obs_values, stats.target_mean.to(obs_values.device), stats.target_std.to(obs_values.device))
        normalized_sources["obs_values"] = obs_source
    metadata["normalization_applied"] = True
    metadata["normalization_scale"] = "mean_std"
    metadata["normalization_metadata_sources"] = normalized_sources
    return PDEBatch(
        pde_name=batch.pde_name,
        task=batch.task,
        full_tensor=batch.full_tensor,
        input_fields=input_fields,
        target_fields=target_fields,
        coords=batch.coords,
        mask=batch.mask,
        obs_values=obs_values,
        obs_coords=batch.obs_coords,
        channel_names=batch.channel_names,
        input_channel_names=batch.input_channel_names,
        target_channel_names=batch.target_channel_names,
        metadata=metadata,
        pde_params=batch.pde_params,
        split=batch.split,
        sample_indices=batch.sample_indices,
        global_sample_ids=list(batch.global_sample_ids),
        file_paths=list(batch.file_paths),
    )


def denormalize_prediction(pred: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    return _denormalize_channels(pred, stats.target_mean.to(pred.device), stats.target_std.to(pred.device))


def _normalize_channels(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    c = min(int(x.shape[1]), int(mean.numel()))
    view = _channel_view(mean[:c], x.ndim)
    scale = _channel_view(std[:c].clamp_min(1e-12), x.ndim)
    out = x.clone()
    out[:, :c] = (out[:, :c] - view) / scale
    return out


def _denormalize_channels(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    c = min(int(x.shape[1]), int(mean.numel()))
    out = x.clone()
    out[:, :c] = out[:, :c] * _channel_view(std[:c], x.ndim) + _channel_view(mean[:c], x.ndim)
    return out


def _normalize_last_channel(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    c = min(int(x.shape[-1]), int(mean.numel()))
    out = x.clone()
    view_shape = [1] * x.ndim
    view_shape[-1] = c
    out[..., :c] = (out[..., :c] - mean[:c].reshape(view_shape)) / std[:c].clamp_min(1e-12).reshape(view_shape)
    return out


def _channel_view(values: torch.Tensor, ndim: int) -> torch.Tensor:
    return values.reshape(1, -1, *([1] * (ndim - 2)))


def _obs_value_normalization_source(batch: PDEBatch) -> str:
    return "input" if batch.task == "sparse_inverse" else "target"


def _metadata_normalization_source(batch: PDEBatch, key: str, value: torch.Tensor, stats: NormalizationStats) -> str:
    if key in {"masked_grid", "voronoi_grid", "observed_solution_fields", "observation_source_fields"}:
        return "input" if batch.task == "sparse_inverse" else "target"
    if key == "original_input_fields":
        return "input"
    if tuple(value.shape) == tuple(batch.input_fields.shape):
        return "input"
    if value.ndim >= 2 and value.shape[1] == stats.target_mean.numel():
        return "target"
    return "input"
