from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.data_adapter import PDEBatch


class BaselineModel(nn.Module):
    name: str = "base"

    def __init__(self) -> None:
        super().__init__()
        self.config: dict[str, Any] = {}
        self.data_spec: dict[str, Any] = {}
        self.backend_used: str = "local"
        self.official_backend: str = "local"
        self.fallback_used: bool = False
        self.backend_warning: str = ""

    def build(self, config, data_spec):
        self.config = dict(config or {})
        self.data_spec = dict(data_spec or {})
        return self

    def set_backend(
        self,
        backend_used: str,
        official_backend: str | None = None,
        fallback_used: bool = False,
        warning: str = "",
    ) -> None:
        self.backend_used = str(backend_used)
        self.official_backend = str(official_backend or backend_used)
        self.fallback_used = bool(fallback_used)
        self.backend_warning = str(warning or "")

    def fit(self, train_loader, val_loader=None):
        return {}

    def predict(self, batch: PDEBatch):
        raise NotImplementedError

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": self.config, "data_spec": self.data_spec}, path)

    def load(self, path):
        payload = torch.load(path, map_location="cpu")
        self.load_state_dict(payload["state_dict"])
        self.config = payload.get("config", {})
        self.data_spec = payload.get("data_spec", {})
        return self

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))


def run_supervised_fit(model: BaselineModel, train_loader, val_loader=None):
    device = torch.device(model.config.get("device", "cpu"))
    epochs = int(model.config.get("epochs", 1))
    lr = float(model.config.get("lr", 1e-3))
    max_steps = model.config.get("max_steps")
    max_val_steps = model.config.get("max_val_steps")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = {"train_loss": [], "val_loss": [], "best_epoch": None, "best_val_loss": None}
    best_state = None
    best_val = None
    for epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for step, batch in enumerate(train_loader):
            if max_steps is not None and step >= int(max_steps):
                break
            target = batch.target_fields.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model.predict(_to_device_batch(batch, device))
            loss = loss_fn(pred, target)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            count += 1
        history["train_loss"].append(total / max(count, 1))
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_count = 0
            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    if max_val_steps is not None and step >= int(max_val_steps):
                        break
                    batch = _to_device_batch(batch, device)
                    pred = model.predict(batch)
                    val_total += float(F.mse_loss(pred, batch.target_fields).detach().cpu())
                    val_count += 1
            val_loss = val_total / max(val_count, 1)
            history["val_loss"].append(val_loss)
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                history["best_epoch"] = epoch
                history["best_val_loss"] = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _to_device_batch(batch: PDEBatch, device: torch.device) -> PDEBatch:
    def move(x):
        return x.to(device) if isinstance(x, torch.Tensor) else x

    meta = _move_metadata(batch.metadata, device)
    return PDEBatch(
        pde_name=batch.pde_name,
        task=batch.task,
        full_tensor=batch.full_tensor.to(device),
        input_fields=batch.input_fields.to(device),
        target_fields=batch.target_fields.to(device),
        coords=move(batch.coords),
        mask=move(batch.mask),
        obs_values=move(batch.obs_values),
        obs_coords=move(batch.obs_coords),
        channel_names=batch.channel_names,
        input_channel_names=batch.input_channel_names,
        target_channel_names=batch.target_channel_names,
        metadata=meta,
        pde_params=_move_metadata(batch.pde_params, device),
        split=batch.split,
        sample_indices=batch.sample_indices.to(device) if isinstance(batch.sample_indices, torch.Tensor) else batch.sample_indices,
        global_sample_ids=list(batch.global_sample_ids),
        file_paths=list(batch.file_paths),
    )


def _move_metadata(metadata: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif isinstance(value, dict):
            out[key] = _move_metadata(value, device)
        else:
            out[key] = value
    return out
