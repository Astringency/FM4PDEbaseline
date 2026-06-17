from __future__ import annotations

import json
import math
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

from .physics import residual_loss


class NotImplementedWarning(Warning):
    """Warning used for intentionally unimplemented physics residuals."""


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred, target = _align(pred, target)
    diff = torch.linalg.vector_norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    denom = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(eps)
    return (diff / denom).mean()


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred, target = _align(pred, target)
    return F.mse_loss(pred, target)


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred, target = _align(pred, target)
    return F.l1_loss(pred, target)


def obs_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    pred, target = _align(pred, target)
    if mask is None:
        return torch.tensor(float("nan"), device=pred.device)
    if tuple(mask.shape) != tuple(pred.shape[1:]):
        # Allow masks generated for the full target to be truncated to pred channels.
        mask = mask[: pred.shape[1]]
    denom = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask.unsqueeze(0)).sum() / (denom * pred.shape[0])


def pde_residual_metric(pred: torch.Tensor, pde_name: str, metadata: dict | None = None) -> torch.Tensor:
    metadata = metadata or {}
    try:
        return residual_loss(pred, pde_name, metadata)
    except NotImplementedError:
        warnings.warn(f"No PDE residual implemented for '{pde_name}'. Returning nan.", NotImplementedWarning, stacklevel=2)
        return _nan(pred)
    except Exception as exc:
        warnings.warn(f"Could not compute {pde_name} PDE residual: {exc}", RuntimeWarning, stacklevel=2)
        return _nan(pred)


def bc_residual_metric(*args, **kwargs) -> torch.Tensor:
    ref = _first_tensor(args, kwargs)
    warnings.warn("Boundary-condition residual metric is not implemented for this PDE.", NotImplementedWarning, stacklevel=2)
    return _nan(ref)


def ic_residual_metric(*args, **kwargs) -> torch.Tensor:
    ref = _first_tensor(args, kwargs)
    warnings.warn("Initial-condition residual metric is not implemented for this PDE.", NotImplementedWarning, stacklevel=2)
    return _nan(ref)


@contextmanager
def runtime_meter() -> Iterator[dict[str, float]]:
    record: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["seconds"] = time.perf_counter() - start


def append_result_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def append_result_csv(path: str | Path, row: dict) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def num_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _align(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.shape == target.shape:
        return pred, target
    if pred.ndim == target.ndim and pred.shape[0] == target.shape[0]:
        slices = [slice(None)]
        for a, b in zip(pred.shape[1:], target.shape[1:]):
            slices.append(slice(0, min(a, b)))
        return pred[tuple(slices)], target[tuple(slices)]
    raise ValueError(f"Cannot align prediction shape {tuple(pred.shape)} with target shape {tuple(target.shape)}")


def _nan(ref: torch.Tensor | None = None) -> torch.Tensor:
    device = ref.device if isinstance(ref, torch.Tensor) else None
    return torch.tensor(float("nan"), device=device)


def _first_tensor(args, kwargs) -> torch.Tensor | None:
    for x in list(args) + list(kwargs.values()):
        if isinstance(x, torch.Tensor):
            return x
    return None
