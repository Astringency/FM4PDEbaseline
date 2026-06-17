from __future__ import annotations

import torch
import torch.nn as nn

from baselines.common.data_adapter import PDEBatch

from .base import BaselineModel, run_supervised_fit
from .shared import MLP


class SenseiverBaseline(BaselineModel):
    name = "senseiver"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        self.out_shape = tuple(data_spec["target_shape"][1:])
        self.out_channels = int(data_spec["target_channels"])
        coord_dim = len(self.out_shape[1:])
        token_dim = int(self.config.get("token_dim", 64))
        num_latents = int(self.config.get("num_latents", 64))
        heads = int(self.config.get("heads", 4))
        self.sensor_proj = MLP(coord_dim + self.out_channels, token_dim, hidden=token_dim, depth=2)
        self.query_proj = MLP(coord_dim, token_dim, hidden=token_dim, depth=2)
        self.latents = nn.Parameter(torch.randn(num_latents, token_dim) * 0.02)
        self.enc_attn = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.dec_attn = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, self.out_channels))
        return self

    def fit(self, train_loader, val_loader=None):
        return run_supervised_fit(self, train_loader, val_loader)

    def predict(self, batch: PDEBatch):
        if batch.obs_values is None or batch.obs_coords is None:
            # Dense fallback: use all target-grid input values as pseudo sensors.
            b, c, h, w = batch.input_fields.shape
            values = batch.input_fields.reshape(b, c, -1).permute(0, 2, 1)
            coords = batch.coords.to(batch.input_fields.device, batch.input_fields.dtype)
        else:
            values = batch.obs_values.to(batch.input_fields.device, batch.input_fields.dtype)
            coords = batch.obs_coords.to(batch.input_fields.device, batch.input_fields.dtype)
            if values.shape[-1] != self.out_channels:
                values = values[..., : self.out_channels]
        b = values.shape[0]
        tokens = self.sensor_proj(torch.cat([coords, values], dim=-1))
        latents = self.latents.unsqueeze(0).repeat(b, 1, 1)
        latents = latents + self.enc_attn(latents, tokens, tokens)[0]
        latents = latents + self.self_attn(latents, latents, latents)[0]
        query = batch.coords.to(batch.input_fields.device, batch.input_fields.dtype)
        q = self.query_proj(query)
        decoded = q + self.dec_attn(q, latents, latents)[0]
        values = self.out(decoded)
        spatial = tuple(self.out_shape[1:])
        return values.permute(0, 2, 1).reshape(b, self.out_channels, *spatial)

