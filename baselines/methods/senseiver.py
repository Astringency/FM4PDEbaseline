from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from baselines.common.data_adapter import PDEBatch

from .base import BaselineModel, run_supervised_fit
from .official import OfficialImportError, get_senseiver_classes, official_source_info, requested_implementation_mode, wrap_official_adapter_error
from .shared import MLP
from .senseiver_positional import senseiver_fourier_features


class SenseiverBaseline(BaselineModel):
    name = "senseiver"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        self.out_shape = tuple(data_spec["target_shape"][1:])
        self.in_channels = int(data_spec["input_channels"])
        self.out_channels = int(data_spec["target_channels"])
        self.spatial_shape = tuple(int(size) for size in self.out_shape[1:])
        coord_dim = len(self.spatial_shape)
        self.space_bands = int(self.config.get("space_bands", 32))
        if self.space_bands < 1:
            raise ValueError(f"space_bands must be positive, got {self.space_bands}")
        self.batch_pixels = int(self.config.get("batch_pixels", 2048))
        if self.batch_pixels < 1:
            raise ValueError(f"batch_pixels must be positive, got {self.batch_pixels}")
        self.position_channels = 2 * coord_dim * self.space_bands

        for removed_setting in ("token_dim", "heads"):
            if removed_setting in self.config:
                raise ValueError(
                    f"Senseiver setting {removed_setting!r} has been removed; use the explicit Senseiver architecture settings"
                )
        enc_preproc_ch = int(self.config.get("enc_preproc_ch", 64))
        num_latents = int(self.config.get("num_latents", 4))
        latent_channels = int(self.config.get("enc_num_latent_channels", 16))
        num_layers = int(self.config.get("num_layers", 3))
        cross_heads = int(self.config.get("num_cross_attention_heads", 2))
        self_heads = int(self.config.get("enc_num_self_attention_heads", 2))
        self_attention_layers = int(self.config.get("num_self_attention_layers_per_block", 3))
        dec_preproc_ch_value = self.config.get("dec_preproc_ch", None)
        dec_preproc_ch = None if dec_preproc_ch_value is None else int(dec_preproc_ch_value)
        decoder_latent_channels = int(self.config.get("dec_num_latent_channels", latent_channels))
        decoder_heads = int(self.config.get("dec_num_cross_attention_heads", 1))
        if decoder_latent_channels != latent_channels:
            raise ValueError(
                "Senseiver decoder latent channels must match encoder latent channels: "
                f"{decoder_latent_channels} != {latent_channels}"
            )
        dropout = float(self.config.get("dropout", 0.0))
        self.config.update(
            {
                "batch_pixels": self.batch_pixels,
                "space_bands": self.space_bands,
                "enc_preproc_ch": enc_preproc_ch,
                "num_latents": num_latents,
                "enc_num_latent_channels": latent_channels,
                "num_layers": num_layers,
                "num_cross_attention_heads": cross_heads,
                "enc_num_self_attention_heads": self_heads,
                "num_self_attention_layers_per_block": self_attention_layers,
                "dec_preproc_ch": dec_preproc_ch,
                "dec_num_latent_channels": decoder_latent_channels,
                "dec_num_cross_attention_heads": decoder_heads,
            }
        )
        self.official_encoder = None
        self.official_decoder = None
        backend = str(self.config.get("official_backend", "auto")).lower()
        implementation_mode = requested_implementation_mode(self.config)
        fallback_reason = ""
        if implementation_mode != "adapted" and backend in {"auto", "senseiver", "official"}:
            try:
                encoder_cls, decoder_cls = get_senseiver_classes()
                self.official_encoder = encoder_cls(
                    input_ch=self.position_channels + self.in_channels,
                    preproc_ch=enc_preproc_ch,
                    num_latents=num_latents,
                    num_latent_channels=latent_channels,
                    num_layers=num_layers,
                    num_cross_attention_heads=cross_heads,
                    num_self_attention_heads=self_heads,
                    num_self_attention_layers_per_block=self_attention_layers,
                    dropout=dropout,
                )
                self.official_decoder = decoder_cls(
                    ff_channels=self.position_channels,
                    preproc_ch=dec_preproc_ch,
                    num_latent_channels=decoder_latent_channels,
                    latent_size=1,
                    num_output_channels=self.out_channels,
                    num_cross_attention_heads=decoder_heads,
                    dropout=dropout,
                )
                self.set_backend(
                    "senseiver",
                    "senseiver",
                    fallback_used=False,
                    implementation_mode_effective="adapted",
                    implementation_source="vendored_senseiver_sum_mse_adam_training_adapter",
                    official_import_success=True,
                    official_reimplementation_success=False,
                    official_alignment_level="algorithm_training",
                    official_alignment_notes=(
                        "Directly imports the vendored Senseiver Encoder and Decoder, ports its Fourier "
                        "positions, and preserves pre-decoder query sampling, the official query-batch "
                        "volume, sum-MSE, Adam, train-loss monitoring, and patience=100; the Lightning "
                        "data interface is adapted to PDEBatch."
                    ),
                    adapter_status="official_training_flow_adapter",
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
        local_channels = latent_channels
        self.sensor_proj = MLP(
            self.position_channels + self.in_channels,
            local_channels,
            hidden=max(enc_preproc_ch, local_channels),
            depth=2,
        )
        self.query_proj = MLP(self.position_channels, local_channels, hidden=local_channels, depth=2)
        self.latents = nn.Parameter(torch.randn(num_latents, local_channels) * 0.02)
        self.enc_attn = nn.MultiheadAttention(local_channels, cross_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(local_channels, self_heads, batch_first=True)
        self.dec_attn = nn.MultiheadAttention(local_channels, decoder_heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(local_channels), nn.Linear(local_channels, self.out_channels))
        requested_local = backend in {"local", "none"} or implementation_mode == "adapted"
        self.set_backend(
            "local",
            "local" if requested_local else ("official" if backend == "official" else backend),
            fallback_used=not requested_local,
            warning=fallback_reason,
            implementation_mode_effective="adapted",
            implementation_source="local_senseiver_architecture_adaptation",
            official_import_success=False,
            official_reimplementation_success=False,
            official_alignment_level="adapted",
            official_alignment_notes=(
                "Local Perceiver-style fallback with Senseiver Fourier positional features; it does not "
                "import the official Encoder or Decoder components."
            ),
            adapter_status="local_adapted" if requested_local else "fallback_adapted",
        )
        return self

    def fit(self, train_loader, val_loader=None):
        return run_supervised_fit(self, train_loader, val_loader)

    def _inputs_and_queries(self, batch: PDEBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch.coords is None:
            raise ValueError("Senseiver requires target-grid coordinates")
        if batch.obs_values is None or batch.obs_coords is None:
            if batch.metadata.get("deferred_dynamic_sensors"):
                raise ValueError(
                    "Senseiver received an unmaterialized random_per_sample batch; access it through "
                    "PDEBatchDataset so observations are generated per item"
                )
            # Dense fallback: use all target-grid input values as pseudo sensors.
            b, c, h, w = batch.input_fields.shape
            values = batch.input_fields.reshape(b, c, -1).permute(0, 2, 1)
            coords = batch.coords.to(batch.input_fields.device, batch.input_fields.dtype)
        else:
            values = batch.obs_values.to(batch.input_fields.device, batch.input_fields.dtype)
            coords = batch.obs_coords.to(batch.input_fields.device, batch.input_fields.dtype)
        if values.shape[-1] != self.in_channels:
            raise ValueError(
                f"Senseiver observations have {values.shape[-1]} channels, expected {self.in_channels}; "
                "input and target channels must not be truncated or broadcast"
            )
        b = values.shape[0]
        query = batch.coords.to(batch.input_fields.device, batch.input_fields.dtype)
        if coords.shape[0] == 1 and b != 1:
            coords = coords.expand(b, -1, -1)
        if query.shape[0] == 1 and b != 1:
            query = query.expand(b, -1, -1)
        return values, coords, query

    def _decode_queries(
        self,
        values: torch.Tensor,
        coords: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        sensor_positions = senseiver_fourier_features(coords, self.spatial_shape, self.space_bands)
        query_positions = senseiver_fourier_features(query, self.spatial_shape, self.space_bands)
        if self.official_encoder is not None and self.official_decoder is not None:
            latents = self.official_encoder(torch.cat([values, sensor_positions], dim=-1))
            decoded = self.official_decoder(latents, query_positions)
            return decoded.permute(0, 2, 1)
        tokens = self.sensor_proj(torch.cat([values, sensor_positions], dim=-1))
        b = values.shape[0]
        latents = self.latents.unsqueeze(0).repeat(b, 1, 1)
        latents = latents + self.enc_attn(latents, tokens, tokens)[0]
        latents = latents + self.self_attn(latents, latents, latents)[0]
        q = self.query_proj(query_positions)
        decoded = q + self.dec_attn(q, latents, latents)[0]
        values = self.out(decoded)
        return values.permute(0, 2, 1)

    def supervised_training_pair(self, batch: PDEBatch) -> tuple[torch.Tensor, torch.Tensor]:
        values, coords, query = self._inputs_and_queries(batch)
        target = batch.target_fields.flatten(start_dim=2)
        spatial_points = int(query.shape[1])
        if int(target.shape[2]) != spatial_points:
            raise ValueError(
                "Senseiver query count must match flattened target pixels: "
                f"{spatial_points} != {int(target.shape[2])}"
            )
        sampled_points = min(self.batch_pixels, spatial_points)
        if sampled_points < spatial_points:
            indices = torch.randperm(spatial_points, device=query.device)[:sampled_points]
            query = query.index_select(1, indices)
            target = target.index_select(2, indices)
        return self._decode_queries(values, coords, query), target

    def supervised_training_updates_per_batch(self, batch: PDEBatch) -> int:
        """Match the official loader's query-batch count for each field batch."""
        spatial_points = int(batch.target_fields.flatten(start_dim=2).shape[2])
        return max(spatial_points // min(self.batch_pixels, spatial_points), 1)

    def predict(self, batch: PDEBatch):
        values, coords, query = self._inputs_and_queries(batch)
        decoded = self._decode_queries(values, coords, query)
        b = values.shape[0]
        spatial = tuple(self.out_shape[1:])
        return decoded.reshape(b, self.out_channels, *spatial)
