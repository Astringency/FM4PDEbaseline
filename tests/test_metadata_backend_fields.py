from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from baselines.methods.official import official_source_info


def test_result_rows_include_capability_and_backend_fields(tmp_path: Path):
    out = tmp_path / "run"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "baselines.run",
            "--baseline",
            "pde_opt",
            "--pde",
            "poisson",
            "--task",
            "sparse_inverse",
            "--experiment-mode",
            "smoke",
            "--dry-run",
            "--synthetic-data",
            "--synthetic-resolution",
            "8",
            "--train-size",
            "2",
            "--test-size",
            "1",
            "--batch-size",
            "1",
            "--num-sensors",
            "4",
            "--steps",
            "1",
            "--output-dir",
            str(out),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    row = json.loads((out / "results_raw.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for field in (
        "implementation_mode_requested",
        "implementation_mode_effective",
        "implementation_source",
        "official_repo",
        "official_commit_or_version",
        "official_import_path",
        "official_vendored_path",
        "official_local_modifications",
        "official_metadata_note",
        "official_import_success",
        "official_reimplementation_success",
        "official_alignment_level",
        "official_alignment_notes",
        "adapter_status",
        "capability_status",
        "unsupported_reason",
        "paper_table_eligible",
        "raw_input_shape",
        "official_input_shape",
        "observation_field_name",
        "predicted_field_name",
    ):
        assert field in row


def test_official_source_info_records_unknown_commit_and_vendored_path():
    info = official_source_info("deepxde")
    assert info["official_repo"].startswith("https://github.com/")
    assert info["official_commit_or_version"] == "unknown"
    assert info["official_commit_or_version"] != "vendored"
    assert "offical/deepxde" in info["official_vendored_path"]
    assert info["official_local_modifications"] == "unknown"
    assert "record" in info["official_metadata_note"]


def test_official_source_metadata_doc_lists_manual_unknown_records():
    path = Path(__file__).resolve().parents[1] / "baselines" / "OFFICIAL_SOURCE_METADATA.md"
    text = path.read_text(encoding="utf-8")
    assert "official_commit_or_version" in text
    assert "official_local_modifications" in text
    assert "unknown" in text
    assert "offical/neuraloperator" in text
    assert "official-aligned" in text.lower()
