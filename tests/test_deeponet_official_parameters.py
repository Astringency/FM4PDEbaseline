from __future__ import annotations

import torch

import baselines.methods.deeponet as deeponet_module
from baselines.common.data_adapter import build_default_registry
from baselines.methods.deeponet import DeepONetBaseline
from baselines.run import build_data_spec


class _DummyDeepXDEDeepONet(torch.nn.Module):
    def __init__(self, branch_layers, trunk_layers, *args, num_outputs=1, **kwargs):
        super().__init__()
        self.branch_in = int(branch_layers[0])
        self.coord_dim = int(trunk_layers[0])
        self.num_outputs = int(num_outputs)
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, inputs):
        branch, coords = inputs
        return self.weight * torch.ones(branch.shape[0], coords.shape[0], self.num_outputs, device=branch.device, dtype=branch.dtype)


def test_deeponet_official_backend_does_not_register_unused_local_branch_trunk(monkeypatch):
    monkeypatch.setattr(deeponet_module, "get_deepxde_deeponet_class", lambda: _DummyDeepXDEDeepONet)
    registry = build_default_registry()
    batch = registry.make_task(registry.synthetic_raw("poisson", n=2, resolution=8), "poisson", "forward")
    model = DeepONetBaseline().build({"implementation_mode": "official_or_skip", "official_backend": "deepxde"}, build_data_spec(batch))

    assert model.official_net is not None
    assert not hasattr(model, "branch")
    assert not hasattr(model, "trunk")
    assert model.parameter_count() == sum(p.numel() for p in model.official_net.parameters() if p.requires_grad)
