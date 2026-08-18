from __future__ import annotations

import json
import os
from types import SimpleNamespace
from pathlib import Path

import pytest

from scripts import build_experiment_matrix, collect_results, plot_results, run_experiments


def _write_matrix(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_matrix_cli_defaults_to_formal_main_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: list[str] = []
    monkeypatch.setattr(build_experiment_matrix, "build_main", lambda argv: captured.extend(argv))

    assert build_experiment_matrix.main(
        ["--output-root", str(tmp_path), "--data-manifest", "report.json"]
    ) == 0

    assert captured == [
        "--config",
        "configs/experiments/main_results.yaml",
        "--output-root",
        str(tmp_path),
        "--matrix-name",
        "main_results",
        "--data-manifest",
        "report.json",
    ]


def test_runner_selects_pending_rows_and_assigns_gpu_slots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    done = tmp_path / "done"
    done.mkdir()
    (done / "run.done").touch()
    matrix = _write_matrix(
        tmp_path / "matrices" / "main.jsonl",
        [
            {"run_id": "done", "output_dir": str(done)},
            {"run_id": "a", "output_dir": str(tmp_path / "a")},
            {"run_id": "b", "output_dir": str(tmp_path / "b")},
            {"run_id": "c", "output_dir": str(tmp_path / "c")},
        ],
    )
    launched: list[tuple[int, str]] = []

    def fake_run(matrix_path: Path, index: int, gpu: str, data_root: Path) -> int:
        launched.append((index, gpu))
        return 0

    monkeypatch.setattr(run_experiments, "run_index", fake_run)
    monkeypatch.setattr(run_experiments, "is_complete", lambda row: row["run_id"] == "done")
    rc = run_experiments.main(
        [str(matrix), "--data-root", str(tmp_path), "--gpus", "2,5", "--jobs-per-gpu", "2"]
    )

    assert rc == 0
    assert sorted(launched) == [(1, "2"), (2, "2"), (3, "5")]


def test_runner_does_not_duplicate_an_active_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    running = tmp_path / "running"
    running.mkdir()
    (running / "run.running").touch()
    matrix = _write_matrix(
        tmp_path / "matrix.jsonl",
        [{"run_id": "active", "output_dir": str(running)}],
    )
    launched: list[int] = []
    monkeypatch.setattr(
        run_experiments,
        "run_index",
        lambda _matrix, index, _gpu, _data_root: launched.append(index) or 0,
    )

    assert run_experiments.main([str(matrix), "--data-root", str(tmp_path)]) == 0
    assert launched == []


def test_runner_recovery_restarts_only_dead_running_processes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    active = tmp_path / "active"
    stale = tmp_path / "stale"
    active.mkdir()
    stale.mkdir()
    (active / "run.running").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "pid_start_ticks": run_experiments.process_start_ticks(os.getpid()),
            }
        ),
        encoding="utf-8",
    )
    (stale / "run.running").write_text(json.dumps({"pid": 999_999_999}), encoding="utf-8")
    matrix = _write_matrix(
        tmp_path / "matrix.jsonl",
        [
            {"run_id": "active", "output_dir": str(active)},
            {"run_id": "stale", "output_dir": str(stale)},
        ],
    )
    launched: list[int] = []
    monkeypatch.setattr(
        run_experiments,
        "run_index",
        lambda _matrix, index, _gpu, _data_root: launched.append(index) or 0,
    )
    monkeypatch.setattr(run_experiments, "quarantine_invalid_output", lambda _row: None)

    assert run_experiments.main(
        [str(matrix), "--data-root", str(tmp_path), "--rerun-running"]
    ) == 0
    assert launched == [1]


def test_runner_recovery_rejects_a_reused_pid_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    output = tmp_path / "reused"
    output.mkdir()
    actual_ticks = run_experiments.process_start_ticks(os.getpid())
    (output / "run.running").write_text(
        json.dumps({"pid": os.getpid(), "pid_start_ticks": actual_ticks + 1}), encoding="utf-8"
    )
    matrix = _write_matrix(
        tmp_path / "matrix.jsonl",
        [{"run_id": "reused", "output_dir": str(output)}],
    )
    launched: list[int] = []
    monkeypatch.setattr(
        run_experiments,
        "run_index",
        lambda _matrix, index, _gpu, _data_root: launched.append(index) or 0,
    )
    monkeypatch.setattr(run_experiments, "quarantine_invalid_output", lambda _row: None)

    assert run_experiments.main(
        [str(matrix), "--data-root", str(tmp_path), "--rerun-running"]
    ) == 0
    assert launched == [0]


def test_runner_recovery_treats_a_malformed_process_identity_as_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    output = tmp_path / "malformed"
    output.mkdir()
    (output / "run.running").write_text(
        json.dumps({"pid": os.getpid(), "pid_start_ticks": "not-an-integer"}), encoding="utf-8"
    )
    matrix = _write_matrix(
        tmp_path / "matrix.jsonl",
        [{"run_id": "malformed", "output_dir": str(output)}],
    )
    launched: list[int] = []
    monkeypatch.setattr(
        run_experiments,
        "run_index",
        lambda _matrix, index, _gpu, _data_root: launched.append(index) or 0,
    )
    monkeypatch.setattr(run_experiments, "quarantine_invalid_output", lambda _row: None)

    assert run_experiments.main(
        [str(matrix), "--data-root", str(tmp_path), "--rerun-running"]
    ) == 0
    assert launched == [0]


def test_runner_dry_run_supports_first_and_index_ranges(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    rows = [{"run_id": str(i), "output_dir": str(tmp_path / str(i))} for i in range(6)]
    matrix = _write_matrix(tmp_path / "matrix.jsonl", rows)

    rc = run_experiments.main(
        [str(matrix), "--data-root", str(tmp_path), "--gpus", "0", "--indices", "1,3-5", "--first", "2", "--dry-run"]
    )

    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert [row["index"] for row in report["selected"]] == [1, 3]


def test_collect_results_calls_aggregate_and_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    run_dir = tmp_path / "runs" / "r1"
    samples = run_dir / "samples"
    samples.mkdir(parents=True)
    (run_dir / "results_raw.jsonl").write_text("{}\n", encoding="utf-8")
    manifest = samples / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps({"sample_manifest_path": str(manifest), "sample_pdf_path": ""}),
        encoding="utf-8",
    )
    matrix = _write_matrix(
        tmp_path / "matrices" / "main_results.jsonl",
        [{"run_id": "r1", "output_dir": str(run_dir)}],
    )
    aggregate_calls: list[list[str]] = []
    export_calls: list[dict] = []
    monkeypatch.setattr(collect_results, "aggregate_main", lambda argv: aggregate_calls.append(argv))
    monkeypatch.setattr(
        collect_results,
        "collect_validated_results",
        lambda rows: SimpleNamespace(records=rows, quarantine=[], missing_run_ids=[]),
    )
    monkeypatch.setattr(collect_results, "require_single_cohort", lambda _records: "cohort")
    monkeypatch.setattr(
        collect_results,
        "export_results",
        lambda rows, **kwargs: export_calls.append({"rows": rows, **kwargs}) or {"rows": 1},
    )
    out = tmp_path / "collected"
    assert collect_results.main([str(matrix), "--output-dir", str(out)]) == 0

    assert aggregate_calls == [[str(run_dir / "results_raw.jsonl"), "--output-dir", str(out)]]
    assert export_calls[0]["output"] == out / "results.xlsx"


def test_plot_results_renders_from_matrix_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    run_dir = tmp_path / "run"
    samples = run_dir / "samples"
    samples.mkdir(parents=True)
    manifest = samples / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps({"sample_manifest_path": str(manifest)}), encoding="utf-8"
    )
    matrix = _write_matrix(
        tmp_path / "matrix.jsonl",
        [{"run_id": "r1", "output_dir": str(run_dir)}],
    )
    calls: list[tuple[Path, Path, bool, int]] = []

    def fake_render(source: Path, output_dir: Path, *, force: bool, max_samples: int) -> dict:
        calls.append((Path(source), Path(output_dir), force, max_samples))
        return {"sample_count": 1, "rendered_count": 1, "skipped_count": 0, "errors": []}

    monkeypatch.setattr(plot_results, "_render_manifest", fake_render)

    assert plot_results.main(["--matrix", str(matrix), "--max-samples", "100"]) == 0
    assert calls == [(manifest, run_dir / "samples_pdf", False, 100)]


def test_plot_results_limits_each_manifest_to_first_n_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    manifest = _write_matrix(
        tmp_path / "samples" / "manifest.jsonl",
        [
            {"sample_ordinal": index, "artifact_path": str(tmp_path / f"sample_{index}.pt")}
            for index in range(3)
        ],
    )

    def fake_sample_pdf(_artifact: Path, output: Path, **_kwargs) -> Path:
        Path(output).write_bytes(b"%PDF\n%%EOF\n")
        return Path(output)

    monkeypatch.setattr(plot_results, "render_evaluation_sample_pdf", fake_sample_pdf)
    output = tmp_path / "pdfs"
    report = plot_results._render_manifest(
        manifest,
        output,
        force=False,
        max_samples=2,
    )

    assert report["source_sample_count"] == 3
    assert report["sample_count"] == 2
    assert sorted(path.name for path in output.glob("sample_*.pdf")) == [
        "sample_000000.pdf",
        "sample_000001.pdf",
    ]


def test_collect_results_refuses_incomplete_matrix_before_writing_tables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    run_dir = tmp_path / "missing"
    matrix = _write_matrix(
        tmp_path / "matrices" / "main_results.jsonl",
        [{"run_id": "missing", "output_dir": str(run_dir)}],
    )
    aggregate_calls: list[list[str]] = []
    monkeypatch.setattr(collect_results, "aggregate_main", lambda argv: aggregate_calls.append(argv))

    assert collect_results.main([str(matrix), "--output-dir", str(tmp_path / "aggregate")]) == 2
    assert aggregate_calls == []
    assert not (tmp_path / "aggregate").exists()


def test_plot_results_reports_missing_sample_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results_raw.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps({"sample_manifest_path": str(run_dir / "missing.jsonl")}), encoding="utf-8"
    )
    matrix = _write_matrix(
        tmp_path / "matrix.jsonl",
        [{"run_id": "r1", "output_dir": str(run_dir)}],
    )
    assert plot_results.main(["--matrix", str(matrix)]) == 2
