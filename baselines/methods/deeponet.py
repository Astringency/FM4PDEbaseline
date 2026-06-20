from __future__ import annotations

import warnings

import torch

from baselines.common.data_adapter import PDEBatch

from .base import BaselineModel, run_supervised_fit
from .official import OfficialImportError, get_deepxde_deeponet_class, official_source_info, requested_implementation_mode, wrap_official_adapter_error
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
        self.basis = basis
        self.out_channels = out_channels
        self.branch_in = branch_in
        self.official_net = None
        backend = str(self.config.get("official_backend", "auto")).lower()
        implementation_mode = requested_implementation_mode(self.config)
        fallback_reason = ""
        if implementation_mode != "adapted" and backend in {"auto", "deepxde", "official"}:
            try:
                deeponet = get_deepxde_deeponet_class()
                self.official_net = deeponet(
                    [branch_in, hidden, hidden, basis],
                    [coord_dim, hidden, hidden, basis],
                    activation=str(self.config.get("activation", "gelu")),
                    kernel_initializer=str(self.config.get("kernel_initializer", "Glorot normal")),
                    num_outputs=out_channels,
                    multi_output_strategy="independent" if out_channels > 1 else None,
                )
                self._official_smoke_check(branch_in, coord_dim, out_channels)
                self.set_backend(
                    "deepxde",
                    "deepxde",
                    fallback_used=False,
                    implementation_mode_effective="official",
                    implementation_source="deepxde",
                    official_import_success=True,
                    adapter_status="official_code_adapter",
                    **official_source_info("deepxde"),
                )
            except Exception as exc:
                exc = wrap_official_adapter_error("DeepXDE DeepONet", exc)
                fallback_reason = f"deepxde unavailable: {exc}"
                if backend in {"deepxde", "official"}:
                    warnings.warn(f"DeepXDE DeepONet unavailable, using local fallback: {exc}", RuntimeWarning, stacklevel=2)
        if implementation_mode == "official" and self.official_net is None and fallback_reason:
            raise OfficialImportError(fallback_reason)
        if self.official_net is None:
            requested_local = backend in {"local", "none"} or implementation_mode == "adapted"
            self.set_backend(
                "local",
                "local" if requested_local else ("official" if backend == "official" else backend),
                fallback_used=not requested_local,
                warning=fallback_reason,
                implementation_mode_effective="adapted",
                implementation_source="local_deeponet",
                official_import_success=False,
                adapter_status="local_adapted" if requested_local else "fallback_adapted",
            )
            self.branch = MLP(branch_in, basis * out_channels, hidden=hidden, depth=3)
            self.trunk = MLP(coord_dim, basis * out_channels, hidden=hidden, depth=3)
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
        coords = batch.coords
        if coords is None:
            raise ValueError("DeepONet requires dense query coordinates in batch.coords")
        coords = coords.to(batch.input_fields.device, batch.input_fields.dtype)
        if self.official_net is not None:
            values = self.official_net((branch_input, coords[0]))
            if values.ndim == 2:
                values = values.unsqueeze(-1)
            spatial = tuple(self.out_shape[1:])
            return values.permute(0, 2, 1).reshape(b, self.out_channels, *spatial)
        coeff = self.branch(branch_input).reshape(b, self.out_channels, self.basis)
        trunk = self.trunk(coords).reshape(b, coords.shape[1], self.out_channels, self.basis)
        values = torch.einsum("bck,bqck->bqc", coeff, trunk) / (self.basis ** 0.5)
        spatial = tuple(self.out_shape[1:])
        return values.permute(0, 2, 1).reshape(b, self.out_channels, *spatial)

    def _official_smoke_check(self, branch_in: int, coord_dim: int, out_channels: int) -> None:
        if self.official_net is None:
            return
        coords = torch.zeros(1, 1, max(coord_dim, 1))
        if coord_dim == 0:
            coords = coords[..., :0]
        branch = torch.zeros(1, branch_in)
        with torch.no_grad():
            values = self.official_net((branch, coords[0]))
        if values.shape[0] != 1:
            raise RuntimeError(f"DeepXDE smoke output batch mismatch: {tuple(values.shape)}")
        if out_channels > 1 and values.reshape(1, -1).numel() < out_channels:
            raise RuntimeError(f"DeepXDE smoke output too small for {out_channels} outputs: {tuple(values.shape)}")
