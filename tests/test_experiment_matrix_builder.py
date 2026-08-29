from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from baselines.experiment_matrix import compatibility_reason


ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_sanity_matrix_generation_unique_run_ids_and_skips(tmp_path: Path):
    out = tmp_path / "large"
    subprocess.run(
        [
            sys.executable,
            "scripts/experiments/build_matrix.py",
            "--config",
            "configs/experiments/sanity_main.yaml",
            "--output-root",
            str(out),
            "--matrix-name",
            "sanity_main",
        ],
        cwd=ROOT,
        check=True,
    )
    rows = _read_jsonl(out / "matrices" / "sanity_main.jsonl")
    assert rows
    run_ids = [row["run_id"] for row in rows]
    assert len(run_ids) == len(set(run_ids))
    skipped = _read_jsonl(out / "skipped_combinations.jsonl")
    assert all(not (row["task"] == "sparse_inverse" and row["baseline"] == "pde_opt") for row in skipped)
    assert any(row["task"] == "sparse_inverse" and row["baseline"] == "pde_opt" for row in rows)


def test_sanity_main_includes_main_table_ifno_inverse(tmp_path: Path):
    out = tmp_path / "large"
    subprocess.run(
        [
            sys.executable,
            "scripts/experiments/build_matrix.py",
            "--config",
            "configs/experiments/sanity_main.yaml",
            "--output-root",
            str(out),
            "--matrix-name",
            "sanity_main",
        ],
        cwd=ROOT,
        check=True,
    )
    rows = _read_jsonl(out / "matrices" / "sanity_main.jsonl")
    inverse_rows = [row for row in rows if row["task_group"] == "full_inverse_main"]

    assert inverse_rows
    assert {row["baseline"] for row in inverse_rows} == {"ifno"}
    assert {row["task"] for row in inverse_rows} == {"inverse"}
    assert {row["execution_mode"] for row in inverse_rows} == {"eval_only"}
    assert all(row["dependency_pending"] for row in inverse_rows)
    assert all(not row.get("skip_reason") for row in inverse_rows)
    assert not any(row["task_group"] == "full_inverse_main" and row["baseline"] == "deeponet" for row in rows)


def test_sanity_matrix_only_uses_documented_pdes(tmp_path: Path):
    out = tmp_path / "large"
    subprocess.run(
        [
            sys.executable,
            "scripts/experiments/build_matrix.py",
            "--config",
            "configs/experiments/sanity_main.yaml",
            "--output-root",
            str(out),
            "--matrix-name",
            "sanity_main",
        ],
        cwd=ROOT,
        check=True,
    )
    rows = _read_jsonl(out / "matrices" / "sanity_main.jsonl")
    assert {row["pde"] for row in rows}.issubset({"poisson", "helmholtz", "darcy", "burger", "nsnonbounded"})


def test_sparse_inverse_per_instance_and_time_varying_unsupported_are_skipped():
    assert compatibility_reason("pde_opt", "poisson", "sparse_inverse") == ""
    assert "time-varying" in compatibility_reason(
        "fno",
        "reaction_diffusion",
        "sparse_solution",
        sensor_mode="time_varying",
        task_group="time_varying",
    )


def test_compatibility_filter_has_no_main_or_supplement_tier():
    assert compatibility_reason("fno", "poisson", "inverse") == ""
    assert compatibility_reason("deeponet", "darcy", "sparse_solution", "random") == ""
    assert compatibility_reason("recfno", "poisson", "sparse_inverse", "random") == ""
    assert compatibility_reason("senseiver", "poisson", "sparse_inverse", "random") == ""
    assert compatibility_reason("voronoicnn", "poisson", "sparse_inverse", "random") == ""


def test_diffusionpde_and_fm4pde_are_outside_matrix():
    assert "DiffusionPDE" in compatibility_reason("fno", "DiffusionPDE", "forward")
    assert "FM4PDE" in compatibility_reason("FM4PDE", "darcy", "forward")
