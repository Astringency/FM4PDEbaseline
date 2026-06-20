from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from baselines.capabilities import resolve_capability
from baselines.methods.official import OfficialImportError
import baselines.run as run_module
from baselines.run import _preflight_backend_availability, main


def test_unsupported_combination_skips_before_dataset_loading(tmp_path: Path):
    out = tmp_path / "out"
    main(
        [
            "--baseline",
            "ifno",
            "--pde",
            "darcy",
            "--task",
            "sparse_solution",
            "--experiment-mode",
            "paper",
            "--data-root",
            str(tmp_path / "missing_data_root"),
            "--output-dir",
            str(out),
        ]
    )
    rows = [json.loads(line) for line in (out / "skipped_combinations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["backend_used"] == "skipped"
    assert rows[-1]["adapter_status"] == "capability_unsupported"


def test_official_aligned_available_combinations_do_not_preflight_skip():
    args = Namespace(experiment_mode="paper", baseline="ifno", pde="darcy")
    cap = resolve_capability("ifno", "darcy", "forward")
    assert _preflight_backend_availability(args, cap, {"implementation_mode": "official_aligned"}) is None

    args = Namespace(experiment_mode="paper", baseline="vivid", pde="reaction_diffusion")
    cap = resolve_capability(
        "vivid",
        "reaction_diffusion",
        "sparse_solution",
        "time_varying",
        "time_varying",
        load_full_trajectory=True,
        train_inverse_operator=True,
        uses_official_inverse_observation_operator=True,
    )
    assert _preflight_backend_availability(args, cap, {"implementation_mode": "official_aligned"}) is None


def test_official_or_skip_uses_aligned_when_direct_official_missing():
    args = Namespace(experiment_mode="paper", baseline="ifno", pde="darcy")
    cap = resolve_capability("ifno", "darcy", "forward")
    assert _preflight_backend_availability(args, cap, {"implementation_mode": "official_or_skip"}) is None


def test_official_or_skip_skips_when_aligned_unavailable(monkeypatch):
    def missing_direct():
        raise OfficialImportError("missing direct")

    def missing_aligned():
        raise OfficialImportError("missing aligned")

    monkeypatch.setattr(run_module, "get_ifno_official_status", missing_direct)
    monkeypatch.setattr(run_module, "get_ifno_official_aligned_status", missing_aligned)
    args = Namespace(experiment_mode="paper", baseline="ifno", pde="darcy")
    cap = resolve_capability("ifno", "darcy", "forward")
    skip = _preflight_backend_availability(args, cap, {"implementation_mode": "official_or_skip"})
    assert skip is not None
    assert "missing aligned" in skip["reason"]
