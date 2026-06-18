from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.experiments import status


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _row(out_root: Path, matrix_name: str, run_id: str) -> dict:
    return {
        "matrix_name": matrix_name,
        "task_group": matrix_name,
        "pde": "poisson",
        "baseline": "pde_opt",
        "run_id": run_id,
        "output_dir": str(out_root / "runs" / matrix_name / run_id),
    }


def _status_root(tmp_path: Path) -> Path:
    out_root = tmp_path / "outputs"
    _write_jsonl(out_root / "matrices" / "main_results.jsonl", [_row(out_root, "main_results", "main")])
    _write_jsonl(out_root / "matrices" / "noise_ablation.jsonl", [_row(out_root, "noise_ablation", "noise")])
    _write_jsonl(out_root / "matrices" / "sensor_count_ablation.jsonl", [_row(out_root, "sensor_count_ablation", "sensor")])
    _write_jsonl(out_root / "matrices" / "noise_ablation_skipped.jsonl", [_row(out_root, "noise_ablation_skipped", "skip")])
    _write_jsonl(
        out_root / "skipped_combinations.jsonl",
        [
            {"matrix_name": "main_results", "task_group": "main_results", "pde": "poisson", "baseline": "pde_opt"},
            {"matrix_name": "noise_ablation", "task_group": "noise_ablation", "pde": "poisson", "baseline": "pde_opt"},
        ],
    )
    return out_root


def _run_status(out_root: Path, *args: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "scripts/experiments/status.py", "--output-root", str(out_root), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout)


def test_default_scope_main_selects_only_main_results(tmp_path: Path):
    out_root = _status_root(tmp_path)
    result = _run_status(out_root)

    assert result["scope"] == "main"
    assert result["matrices"] == [str(out_root / "matrices" / "main_results.jsonl")]
    assert result["status_csv"] == str(out_root / "status_main.csv")
    assert result["status_md"] == str(out_root / "status_main.md")
    assert Path(result["latest_status_csv"]) == out_root / "status.csv"
    assert Path(result["latest_status_md"]) == out_root / "status.md"
    assert (out_root / "status_main.csv").exists()
    assert "Scope: main" in (out_root / "status_main.md").read_text(encoding="utf-8")


def test_scope_all_selects_all_non_skipped_matrices(tmp_path: Path):
    out_root = _status_root(tmp_path)
    result = _run_status(out_root, "--scope", "all")

    assert result["scope"] == "all"
    assert result["status_csv"] == str(out_root / "status_all.csv")
    assert result["status_md"] == str(out_root / "status_all.md")
    assert {Path(path).name for path in result["matrices"]} == {
        "main_results.jsonl",
        "noise_ablation.jsonl",
        "sensor_count_ablation.jsonl",
    }
    assert "noise_ablation_skipped.jsonl" not in {Path(path).name for path in result["matrices"]}
    assert "Scope: all" in (out_root / "status_all.md").read_text(encoding="utf-8")


def test_matrix_arg_auto_selects_matrix_scope_and_matrix_filename(tmp_path: Path):
    out_root = _status_root(tmp_path)
    matrix = out_root / "matrices" / "noise_ablation.jsonl"
    result = _run_status(out_root, "--matrix", str(matrix))

    assert result["scope"] == "matrix"
    assert result["matrices"] == [str(matrix)]
    assert result["status_csv"] == str(out_root / "status_noise_ablation.csv")
    assert result["status_md"] == str(out_root / "status_noise_ablation.md")
    assert "Scope: matrix" in (out_root / "status_noise_ablation.md").read_text(encoding="utf-8")


def test_scope_matrix_requires_matrix(tmp_path: Path):
    out_root = _status_root(tmp_path)
    proc = subprocess.run(
        [sys.executable, "scripts/experiments/status.py", "--output-root", str(out_root), "--scope", "matrix"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "--scope matrix requires --matrix" in proc.stderr


def test_all_matrices_legacy_env_uses_all_scope(tmp_path: Path):
    out_root = _status_root(tmp_path)
    env = os.environ.copy()
    env.pop("MATRIX", None)
    env.pop("STATUS_SCOPE", None)
    env["OUT_ROOT"] = str(out_root)
    env["ALL_MATRICES"] = "1"
    proc = subprocess.run(
        ["bash", "scripts/experiments/08_status.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(proc.stdout)

    assert result["scope"] == "all"
    assert result["status_csv"] == str(out_root / "status_all.csv")
    assert (out_root / "status_all.csv").exists()


def test_matrix_scope_env_uses_matrix_status_filename(tmp_path: Path):
    out_root = _status_root(tmp_path)
    matrix = out_root / "matrices" / "noise_ablation.jsonl"
    env = os.environ.copy()
    env["OUT_ROOT"] = str(out_root)
    env["MATRIX"] = str(matrix)
    env["STATUS_SCOPE"] = "matrix"
    proc = subprocess.run(
        ["bash", "scripts/experiments/08_status.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(proc.stdout)

    assert result["scope"] == "matrix"
    assert result["matrices"] == [str(matrix)]
    assert result["status_csv"] == str(out_root / "status_noise_ablation.csv")


def test_all_matrices_legacy_cli_flag_uses_all_scope(tmp_path: Path):
    out_root = _status_root(tmp_path)
    args = status.parse_args(["--output-root", str(out_root), "--all-matrices"])

    assert args.scope == "all"
