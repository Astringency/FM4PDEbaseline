from __future__ import annotations

from pathlib import Path

from baselines.common.data_adapter import build_default_registry


def test_future_materialize_adds_scalar_input_channels(tiny_data_root):
    registry = build_default_registry()
    cases = {
        "heat": 2,
        "advection_diffusion": 4,
        "steady_heat_conduction": 2,
    }
    for pde, expected_channels in cases.items():
        raw = registry.load_raw(pde, tiny_data_root, split="train", max_samples=2, scalar_param_mode="materialize")
        batch = registry.make_task(raw, pde, "forward")
        assert batch.input_fields.shape[1] == expected_channels
        assert any(name in batch.input_channel_names for name in batch.pde_params)


def test_full_operator_script_defaults_future_supervised_to_materialize():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/baselines/run_paper_full_operator.sh").read_text(encoding="utf-8")
    assert 'SUPERVISED_SCALAR_PARAM_MODE="${SUPERVISED_SCALAR_PARAM_MODE:-materialize}"' in text
    assert 'SCALAR_PARAM_MODE_WAS_SET="${SCALAR_PARAM_MODE+x}"' in text
    assert 'scalar_mode="$SUPERVISED_SCALAR_PARAM_MODE"' in text


def test_full_operator_script_allows_explicit_metadata_override():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/baselines/run_paper_full_operator.sh").read_text(encoding="utf-8")
    assert 'SCALAR_PARAM_MODE="${SCALAR_PARAM_MODE:-metadata}"' in text
    assert 'if is_future_pde "$pde" && [ -z "$SCALAR_PARAM_MODE_WAS_SET" ]; then' in text
