from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from baselines.common.data_adapter import PDEBatch

from .base import BaselineModel, run_supervised_fit
from .official import OfficialImportError, get_senseiver_classes, official_source_info, requested_implementation_mode, wrap_official_adapter_error
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
        self.official_encoder = None
        self.official_decoder = None
        backend = str(self.config.get("official_backend", "auto")).lower()
        implementation_mode = requested_implementation_mode(self.config)
        fallback_reason = ""
        if implementation_mode != "adapted" and backend in {"auto", "senseiver", "official"}:
            try:
                encoder_cls, decoder_cls = get_senseiver_classes()
                self.official_encoder = encoder_cls(
                    input_ch=coord_dim + self.out_channels,
                    preproc_ch=token_dim,
                    num_latents=num_latents,
                    num_latent_channels=token_dim,
                    num_layers=int(self.config.get("num_layers", 1)),
                    num_cross_attention_heads=heads,
                    num_self_attention_heads=heads,
                    num_self_attention_layers_per_block=int(self.config.get("self_attention_layers", 1)),
                    dropout=float(self.config.get("dropout", 0.0)),
                )
                self.official_decoder = decoder_cls(
                    ff_channels=coord_dim,
                    preproc_ch=token_dim,
                    num_latent_channels=token_dim,
                    latent_size=1,
                    num_output_channels=self.out_channels,
                    num_cross_attention_heads=heads,
                    dropout=float(self.config.get("dropout", 0.0)),
                )
                self.set_backend(
                    "senseiver",
                    "senseiver",
                    fallback_used=False,
                    implementation_mode_effective="official",
                    implementation_source="senseiver",
                    official_import_success=True,
                    adapter_status="official_code_adapter",
                    **official_source_info("senseiver"),
                )
                return self
            except Exception as exc:
                exc = wrap_official_adapter_error("Senseiver", exc)
                fallback_reason = f"senseiver unavailable: {exc}"
                if backend in {"senseiver", "official"}:
                    warnings.warn(f"Senseiver official modules unavailable, using local fallback: {exc}", RuntimeWarning, stacklevel=2)
        if implementation_mode == "official" and self.official_encoder is None and fallback_reason:
            raise OfficialImportError(fallback_reason)
        self.sensor_proj = MLP(coord_dim + self.out_channels, token_dim, hidden=token_dim, depth=2)
        self.query_proj = MLP(coord_dim, token_dim, hidden=token_dim, depth=2)
        self.latents = nn.Parameter(torch.randn(num_latents, token_dim) * 0.02)
        self.enc_attn = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.dec_attn = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, self.out_channels))
        requested_local = backend in {"local", "none"} or implementation_mode == "adapted"
        self.set_backend(
            "local",
            "local" if requested_local else ("official" if backend == "official" else backend),
            fallback_used=not requested_local,
            warning=fallback_reason,
            implementation_mode_effective="adapted",
            implementation_source="local_senseiver_style",
            official_import_success=False,
            adapter_status="local_adapted" if requested_local else "fallback_adapted",
        )
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
        query = batch.coords.to(batch.input_fields.device, batch.input_fields.dtype)
        if self.official_encoder is not None and self.official_decoder is not None:
            latents = self.official_encoder(torch.cat([coords, values], dim=-1))
            decoded = self.official_decoder(latents, query)
            spatial = tuple(self.out_shape[1:])
            return decoded.permute(0, 2, 1).reshape(b, self.out_channels, *spatial)
        tokens = self.sensor_proj(torch.cat([coords, values], dim=-1))
        latents = self.latents.unsqueeze(0).repeat(b, 1, 1)
        latents = latents + self.enc_attn(latents, tokens, tokens)[0]
        latents = latents + self.self_attn(latents, latents, latents)[0]
        q = self.query_proj(query)
        decoded = q + self.dec_attn(q, latents, latents)[0]
        values = self.out(decoded)
        spatial = tuple(self.out_shape[1:])
        return values.permute(0, 2, 1).reshape(b, self.out_channels, *spatial)
