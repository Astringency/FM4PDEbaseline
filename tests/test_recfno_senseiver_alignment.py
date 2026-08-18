from __future__ import annotations

import importlib.util
from pathlib import Path
import warnings

import pytest
import torch
import yaml

import baselines.methods.recfno as recfno_module
import baselines.methods.senseiver as senseiver_module
from baselines.common.data_adapter import build_default_registry
from baselines.methods.recfno import RecFNOBaseline
from baselines.methods.senseiver import SenseiverBaseline
from baselines.methods.shared import grid_channels
from baselines.run import build_data_spec


class _CaptureRecFNONet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.last_input = None

    def forward(self, x):
        self.last_input = x.detach().clone()
        return x[:, : self.out_channels]


def _poisson_sparse_batch():
    registry = build_default_registry()
    raw = registry.synthetic_raw("poisson", n=2, resolution=4)
    return registry.make_task(raw, "poisson", "sparse_solution", num_sensors=3, sensor_mode="fixed", seed=5)


def test_recfno_official_component_receives_voronoi_mask_and_coordinates(monkeypatch):
    batch = _poisson_sparse_batch()
    batch.metadata["masked_grid"] = torch.full_like(batch.input_fields, -3.0)
    batch.metadata["voronoi_grid"] = torch.full_like(batch.input_fields, 7.0)
    monkeypatch.setattr(recfno_module, "OfficialRecFNOVoronoiFNO2dNet", _CaptureRecFNONet)

    model = RecFNOBaseline().build(
        {"implementation_mode": "official_or_skip", "official_backend": "recfno"},
        build_data_spec(batch),
    )
    model.predict(batch)

    actual = model.net.last_input
    assert actual is not None
    assert model.net.in_channels == 6
    assert torch.equal(actual[:, :2], batch.metadata["voronoi_grid"])
    assert torch.equal(actual[:, 2:4], batch.mask.unsqueeze(0).expand(batch.input_fields.shape[0], -1, -1, -1))
    assert torch.equal(actual[:, 4:], grid_channels(batch.input_fields))

    backend = model.backend_metadata()
    assert backend["official_import_success"] is True
    assert backend["implementation_mode_effective"] == "adapted"
    assert backend["official_alignment_level"] == "component"
    assert "component" in backend["adapter_status"]
    assert "unified" in backend["official_alignment_notes"].lower()


def test_recfno_rejects_a_misleading_non_voronoi_input_config():
    batch = _poisson_sparse_batch()
    with pytest.raises(ValueError, match="input_representation"):
        RecFNOBaseline().build(
            {"implementation_mode": "adapted", "input_representation": "masked_grid"},
            build_data_spec(batch),
        )


def test_recfno_requires_explicit_voronoi_grid_and_mask():
    batch = _poisson_sparse_batch()
    model = RecFNOBaseline().build(
        {"implementation_mode": "adapted", "input_representation": "voronoi_mask_coords"},
        build_data_spec(batch),
    )
    batch.metadata.pop("voronoi_grid")
    with pytest.raises(ValueError, match="voronoi_grid"):
        model.predict(batch)

    batch.metadata["voronoi_grid"] = torch.ones_like(batch.input_fields)
    batch.mask = None
    with pytest.raises(ValueError, match="mask"):
        model.predict(batch)


class _CaptureSenseiverEncoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.init_kwargs = kwargs
        self.last_input = None

    def forward(self, x):
        self.last_input = x.detach().clone()
        channels = int(self.init_kwargs["num_latent_channels"])
        return torch.zeros(x.shape[0], 1, channels, device=x.device, dtype=x.dtype)


class _CaptureSenseiverDecoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.init_kwargs = kwargs
        self.last_coords = None

    def forward(self, latents, coords):
        self.last_coords = coords.detach().clone()
        channels = int(self.init_kwargs["num_output_channels"])
        return torch.zeros(coords.shape[0], coords.shape[1], channels, device=coords.device, dtype=coords.dtype)


def _official_senseiver_positional_grid(shape: tuple[int, int], bands: int) -> torch.Tensor:
    path = Path(__file__).resolve().parents[1] / "offical" / "Senseiver" / "positional.py"
    spec = importlib.util.spec_from_file_location("_test_official_senseiver_positional", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")
        return module.PositionalEncoder((*shape, 1), bands)


def test_senseiver_official_components_receive_official_fourier_features(monkeypatch):
    batch = _poisson_sparse_batch()
    sensor_indices = torch.tensor([0, 5, 15])
    batch.obs_coords = batch.coords[:, sensor_indices]
    batch.obs_values = torch.tensor(
        [[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], [[40.0, 41.0], [50.0, 51.0], [60.0, 61.0]]]
    )
    monkeypatch.setattr(
        senseiver_module,
        "get_senseiver_classes",
        lambda: (_CaptureSenseiverEncoder, _CaptureSenseiverDecoder),
    )

    model = SenseiverBaseline().build(
        {
            "implementation_mode": "official_or_skip",
            "official_backend": "senseiver",
            "space_bands": 2,
            "enc_preproc_ch": 8,
            "num_latents": 4,
            "enc_num_latent_channels": 6,
            "num_layers": 3,
            "num_cross_attention_heads": 2,
            "enc_num_self_attention_heads": 2,
            "num_self_attention_layers_per_block": 3,
            "dec_preproc_ch": None,
            "dec_num_cross_attention_heads": 1,
        },
        build_data_spec(batch),
    )
    model.predict(batch)

    expected_grid = _official_senseiver_positional_grid((4, 4), bands=2)
    expected_sensor_features = expected_grid[sensor_indices].unsqueeze(0).expand(2, -1, -1)
    assert model.official_encoder.init_kwargs["input_ch"] == 10
    assert model.official_decoder.init_kwargs["ff_channels"] == 8
    assert torch.equal(model.official_encoder.last_input[..., :2], batch.obs_values)
    assert torch.allclose(model.official_encoder.last_input[..., 2:], expected_sensor_features)
    assert torch.allclose(model.official_decoder.last_coords, expected_grid.unsqueeze(0).expand(2, -1, -1))

    backend = model.backend_metadata()
    assert backend["official_import_success"] is True
    assert backend["implementation_mode_effective"] == "adapted"
    assert backend["official_alignment_level"] == "component"
    assert "component" in backend["adapter_status"]
    assert "unified" in backend["official_alignment_notes"].lower()


def test_paper_configs_disclose_recfno_input_and_senseiver_architecture():
    root = Path(__file__).resolve().parents[1]
    main = yaml.safe_load((root / "baselines" / "configs" / "paper.yaml").read_text(encoding="utf-8"))
    recfno = main["method_by_baseline"]["recfno"]
    senseiver = main["method_by_baseline"]["senseiver"]

    assert {"token_dim", "num_latents", "heads"}.isdisjoint(main["method"])
    assert recfno["input_representation"] == "voronoi_mask_coords"
    assert "embedding" not in recfno
    assert senseiver == {
        "space_bands": 32,
        "enc_preproc_ch": 64,
        "num_latents": 4,
        "enc_num_latent_channels": 16,
        "num_layers": 3,
        "num_cross_attention_heads": 2,
        "enc_num_self_attention_heads": 2,
        "num_self_attention_layers_per_block": 3,
        "dec_preproc_ch": None,
        "dec_num_latent_channels": 16,
        "dec_num_cross_attention_heads": 1,
    }

    method_recfno = yaml.safe_load(
        (root / "baselines" / "configs" / "paper" / "recfno.yaml").read_text(encoding="utf-8")
    )["method"]
    method_senseiver = yaml.safe_load(
        (root / "baselines" / "configs" / "paper" / "senseiver.yaml").read_text(encoding="utf-8")
    )["method"]
    assert method_recfno["input_representation"] == "voronoi_mask_coords"
    for key, value in senseiver.items():
        assert method_senseiver[key] == value
    assert method_recfno["epochs"] == 200
    assert method_senseiver["epochs"] == 200

    default = yaml.safe_load((root / "baselines" / "configs" / "default.yaml").read_text(encoding="utf-8"))
    assert {"token_dim", "num_latents", "heads"}.isdisjoint(default["method"])
    assert default["method_by_baseline"]["recfno"]["input_representation"] == "voronoi_mask_coords"
    assert default["method_by_baseline"]["senseiver"] == senseiver

    tuning = yaml.safe_load((root / "baselines" / "configs" / "tuning.yaml").read_text(encoding="utf-8"))[
        "tuning_grid"
    ]
    assert "token_dim" not in tuning["recfno"]
    assert tuning["recfno"]["input_representation"] == ["voronoi_mask_coords"]
    assert tuning["recfno"]["modes1"] == [12, 20]
    assert tuning["recfno"]["modes2"] == [12, 20]
    assert "token_dim" not in tuning["senseiver"]
    assert tuning["senseiver"]["space_bands"] == [16, 32]
    assert tuning["senseiver"]["num_latents"] == [4, 16]
    assert tuning["senseiver"]["enc_num_latent_channels"] == [16, 32]
