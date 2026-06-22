from __future__ import annotations

import sys
from pathlib import Path

from scripts.experiments import run_one


def _row(tmp_path: Path, run_id: str = "progress_run") -> dict:
    return {
        "run_id": run_id,
        "run_name": run_id,
        "experiment_kind": "main",
        "ablation_factor": "",
        "task_group": "sparse_solution_main_amortized",
        "task": "sparse_solution",
        "pde": "poisson",
        "baseline": "recfno",
        "seed": 1,
        "train_size": 8,
        "val_size": 0,
        "test_size": 4,
        "train_shards": 1,
        "num_sensors": 16,
        "sensor_mode": "random",
        "noise_level": 0.0,
        "scalar_param_mode": "metadata",
        "data_loading_mode": "eager",
        "load_full_trajectory": False,
        "batch_size": 1,
        "epochs": 1,
        "steps": 0,
        "refine_steps": 0,
        "particles": 0,
        "device": "cpu",
        "config": "baselines/configs/paper.yaml",
        "output_dir": str(tmp_path / run_id),
        "log_dir": str(tmp_path / "logs" / run_id),
        "status_file": str(tmp_path / run_id / "run.status.json"),
        "skip_reason": "",
    }


def test_progress_can_be_disabled(monkeypatch, capsys):
    monkeypatch.setenv("RUN_PROGRESS", "0")

    run_one.progress("[run start] index=1/1 run_id=hidden output_dir=/tmp/hidden")

    captured = capsys.readouterr()
    assert captured.err == ""


def test_default_progress_reports_skip_context(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv("RUN_PROGRESS", raising=False)
    row = _row(tmp_path, "skip_context")
    output_dir = Path(row["output_dir"])
    output_dir.mkdir(parents=True)
    (output_dir / "run.done").write_text("{}", encoding="utf-8")

    rc = run_one.run_one(row, [sys.executable, "-c", "raise SystemExit(99)"], index=2, total=5)

    assert rc == 0
    captured = capsys.readouterr()
    assert "[run skip]" in captured.err
    assert "index=3/5" in captured.err
    assert "run_id=skip_context" in captured.err
    assert f"output_dir={output_dir}" in captured.err


def test_heartbeat_uses_short_interval_without_slow_test(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv("RUN_PROGRESS", raising=False)
    monkeypatch.delenv("RUN_TAIL_LOGS", raising=False)
    monkeypatch.setenv("PROGRESS_INTERVAL_SECONDS", "0.01")
    row = _row(tmp_path, "heartbeat_context")
    cmd = [
        sys.executable,
        "-c",
        "import sys, time; print('child stdout'); print('child stderr', file=sys.stderr); time.sleep(0.08)",
    ]

    rc = run_one.run_one(row, cmd, index=0, total=1)

    assert rc == 0
    captured = capsys.readouterr()
    assert "[run start] index=1/1 run_id=heartbeat_context" in captured.err
    assert "[run heartbeat] index=1/1 run_id=heartbeat_context" in captured.err
    assert "stdout=" in captured.err
    assert "stderr=" in captured.err
    assert "child stdout" in (Path(row["output_dir"]) / "stdout.log").read_text(encoding="utf-8")
    assert "child stderr" in (Path(row["output_dir"]) / "stderr.log").read_text(encoding="utf-8")
