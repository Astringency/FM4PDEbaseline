from __future__ import annotations

from baselines.common.data_adapter import PDEBatch

from .base import BaselineModel, run_supervised_fit
from .shared import FNO2dNet


class FNOBaseline(BaselineModel):
    name = "fno"

    def build(self, config, data_spec):
        super().build(config, data_spec)
        self.net = FNO2dNet(
            in_channels=int(data_spec["input_channels"]),
            out_channels=int(data_spec["target_channels"]),
            width=int(self.config.get("width", 24)),
            modes1=int(self.config.get("modes1", 12)),
            modes2=int(self.config.get("modes2", 12)),
            layers=int(self.config.get("layers", 4)),
        )
        return self

    def fit(self, train_loader, val_loader=None):
        return run_supervised_fit(self, train_loader, val_loader)

    def predict(self, batch: PDEBatch):
        return self.net(batch.input_fields)

