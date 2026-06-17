from __future__ import annotations

import torch

from baselines.common.data_adapter import PDEBatch

from .base import BaselineModel, run_supervised_fit
from .shared import MLP, flatten_grid


class DeepONetBaseline(BaselineModel):
    name = "deeponet"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        self.out_shape = tuple(data_spec["target_shape"][1:])
        basis = int(self.config.get("basis", 64))
        hidden = int(self.config.get("hidden", 128))
        branch_in = int(data_spec.get("branch_numel", data_spec["input_numel"]))
        coord_dim = len(self.out_shape[1:])
        out_channels = int(data_spec["target_channels"])
        self.branch = MLP(branch_in, basis * out_channels, hidden=hidden, depth=3)
        self.trunk = MLP(coord_dim, basis * out_channels, hidden=hidden, depth=3)
        self.basis = basis
        self.out_channels = out_channels
        self.branch_in = branch_in
        return self

    def fit(self, train_loader, val_loader=None):
        return run_supervised_fit(self, train_loader, val_loader)

    def predict(self, batch: PDEBatch):
        b = batch.input_fields.shape[0]
        branch_input = flatten_grid(batch.input_fields)
        if "sparse" in batch.task and batch.obs_values is not None:
            branch_input = batch.obs_values.reshape(b, -1)
        if branch_input.shape[1] < self.branch_in:
            pad = torch.zeros(b, self.branch_in - branch_input.shape[1], device=branch_input.device, dtype=branch_input.dtype)
            branch_input = torch.cat([branch_input, pad], dim=1)
        elif branch_input.shape[1] > self.branch_in:
            branch_input = branch_input[:, : self.branch_in]
        coeff = self.branch(branch_input).reshape(b, self.out_channels, self.basis)
        coords = batch.coords
        if coords is None:
            raise ValueError("DeepONet requires dense query coordinates in batch.coords")
        coords = coords.to(batch.input_fields.device, batch.input_fields.dtype)
        trunk = self.trunk(coords).reshape(b, coords.shape[1], self.out_channels, self.basis)
        values = torch.einsum("bck,bqck->bqc", coeff, trunk) / (self.basis ** 0.5)
        spatial = tuple(self.out_shape[1:])
        return values.permute(0, 2, 1).reshape(b, self.out_channels, *spatial)
