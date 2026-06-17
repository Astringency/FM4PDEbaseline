from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from baselines.common.data_adapter import PDEBatch


class BaselineModel(nn.Module):
    name: str = "base"

    def __init__(self) -> None:
        super().__init__()
        self.config: dict[str, Any] = {}
        self.data_spec: dict[str, Any] = {}

    def build(self, config, data_spec):
        self.config = dict(config or {})
        self.data_spec = dict(data_spec or {})
        return self

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
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = {"train_loss": []}
    for _epoch in range(epochs):
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
    return history


def _to_device_batch(batch: PDEBatch, device: torch.device) -> PDEBatch:
    def move(x):
        return x.to(device) if isinstance(x, torch.Tensor) else x

    meta = dict(batch.metadata)
    for key, value in list(meta.items()):
        if isinstance(value, torch.Tensor):
            meta[key] = value.to(device)
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
        metadata=meta,
    )
