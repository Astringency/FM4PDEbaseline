from __future__ import annotations

import json
from pathlib import Path

from scripts.run_eval import (
    DISTRIBUTION_TEST_FILES,
    build_evaluation_run,
    filter_rows,
    load_source_summary,
    split_selection,
    successful_summary,
)


def _row(**overrides):
    row = {
        "baseline": "recfno",
        "pde": "poisson",
        "task": "sparse_forward",
        "task_group": "sparse_forward_main_amortized",
        "seed": 1,
        "sensor_seed": 1,
        "run_id": "source-run",
        "run_fingerprint": "a" * 64,
        "execution_mode": "train",
        "train_size": 50_000,
        "val_size": 5_000,
        "test_size": 1_000,
        "train_shards": 5,
        "batch_size": 16,
        "epochs": 200,
        "num_sensors": 500,
        "sensor_mode": "random_per_sample",
        "sensor_budget_mode": "per_time",
        "noise_level": 0.0,
        "data_loading_mode": "eager",
        "num_workers": 4,
        "prefetch_factor": 2,
        "pin_memory": True,
        "persistent_workers": True,
        "scalar_param_mode": "metadata",
        "physics_metric_mode": "per_sample",
        "comparison_track": "unified_adapted",
        "task_protocol_version": "fm4pde-task-contract-v3",
        "sensor_protocol_version": "fm4pde-sensor-contract-v3",
        "data_files": {
            "train": ["poisson/train.mat"],
            "test": ["poisson/original.mat"],
        },
        "steps": 0,
        "refine_steps": 0,
        "particles": 0,
    }
    row.update(overrides)
    return row


def _command_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_split_selection_accepts_baseline_list_syntax():
    assert split_selection("FNO, ifno  recfno,FNO") == ["fno", "ifno", "recfno"]


def test_distribution_mapping_covers_every_main_pde():
    expected = {"poisson", "helmholtz", "darcy", "nsnonbounded", "burger"}
    assert set(DISTRIBUTION_TEST_FILES) == {"id", "smooth", "rough"}
    assert all(set(mapping) == expected for mapping in DISTRIBUTION_TEST_FILES.values())


def test_filter_rows_keeps_distinct_task_groups():
    rows = [
        _row(task_group="sparse_solution_main_amortized", task="sparse_solution"),
        _row(task_group="sparse_solution_burger_time_slices", task="sparse_solution"),
        _row(baseline="vivid", task_group="time_varying_da_main", task="sparse_solution"),
    ]
    selected = filter_rows(rows, pdes=["poisson"], baselines=["recfno"])
    assert [row["task_group"] for row in selected] == [
        "sparse_solution_main_amortized",
        "sparse_solution_burger_time_slices",
    ]


def test_checkpoint_row_builds_eval_only_command(tmp_path: Path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    row = _row()
    summary = {
        "status": "success",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": "b" * 64,
        "baseline_code_sha256": "c" * 64,
    }
    evaluation = build_evaluation_run(
        row,
        summary,
        tmp_path,
        test_file="poisson/rough.mat",
        test_size=1000,
        eval_root=tmp_path / "evaluations",
        eval_tag="rough",
        data_root=tmp_path,
        config=tmp_path / "paper.yaml",
        python_bin="python",
        device="cuda",
        save_samples=False,
        train_root=tmp_path,
    )

    assert evaluation.uses_checkpoint is True
    assert _command_value(evaluation.command, "--execution-mode") == "eval_only"
    assert _command_value(evaluation.command, "--checkpoint") == str(checkpoint)
    assert _command_value(evaluation.command, "--source-train-task") == "sparse_forward"
    assert (
        _command_value(evaluation.command, "--source-train-baseline-code-sha256")
        == "c" * 64
    )
    assert "--eval-only" in evaluation.command
    assert "--resume-eval" in evaluation.command
    assert "--no-save-sample-artifacts" in evaluation.command
    data_files = json.loads(_command_value(evaluation.command, "--data-files-json"))
    assert data_files == {
        "train": ["poisson/train.mat"],
        "test": ["poisson/rough.mat"],
    }


def test_ablation_source_path_is_translated_to_active_output_mount(tmp_path: Path):
    output_root = tmp_path / "outputs" / "FM4PDEbaseline"
    relative_run = Path(
        "ablations/ablation=sparse_solution_multicondition/"
        "pde=poisson/baseline=recfno/seed=1/run=source"
    )
    mounted_run = output_root / "runs" / relative_run
    mounted_run.mkdir(parents=True)
    (mounted_run / "summary.json").write_text(
        json.dumps({"status": "success"}), encoding="utf-8"
    )
    row = _row(
        output_dir=str(Path("/remote/storage/outputs/FM4PDEbaseline/runs") / relative_run)
    )

    run_dir, summary = load_source_summary(
        row, output_root / "runs/main_results", output_root
    )

    assert run_dir == mounted_run
    assert summary["status"] == "success"


def test_multicondition_ablation_preserves_evaluation_condition(tmp_path: Path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    probabilities = {
        "a_only": 0.3333333333,
        "u_only": 0.3333333333,
        "both": 0.3333333334,
    }
    row = _row(
        task="sparse_solution_multicondition",
        task_group="sparse_solution_multicondition_eval_u_only",
        execution_mode="eval_only",
        condition_mode="u_only",
        condition_probabilities=probabilities,
        source_train_run_id="multicondition-train",
        source_train_run_fingerprint="d" * 64,
        source_train_seed=1,
        source_train_task="sparse_solution_multicondition",
        source_train_baseline_code_sha256="e" * 64,
        checkpoint_path=str(checkpoint),
        sensor_budget_mode="total",
    )
    evaluation = build_evaluation_run(
        row,
        {
            "status": "success",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": "b" * 64,
        },
        tmp_path,
        test_file="poisson/rough.mat",
        test_size=1000,
        eval_root=tmp_path / "evaluations",
        eval_tag="rough",
        data_root=tmp_path,
        config=tmp_path / "paper.yaml",
        python_bin="python",
        device="cuda",
        save_samples=False,
        train_root=tmp_path,
        output_root=tmp_path,
    )

    assert _command_value(evaluation.command, "--condition-mode") == "u_only"
    assert json.loads(
        _command_value(evaluation.command, "--condition-probabilities-json")
    ) == probabilities
    assert _command_value(evaluation.command, "--sensor-budget-mode") == "total"


def test_per_instance_row_reruns_original_budget_without_checkpoint(tmp_path: Path):
    row = _row(
        baseline="pc_bnn",
        task="sparse_inverse",
        task_group="sparse_inverse_main",
        batch_size=1,
        epochs=1,
        steps=2000,
        particles=5,
    )
    evaluation = build_evaluation_run(
        row,
        {"status": "success", "checkpoint_path": ""},
        tmp_path,
        test_file="poisson/id.mat",
        test_size=1000,
        eval_root=tmp_path / "evaluations",
        eval_tag="id",
        data_root=tmp_path,
        config=tmp_path / "paper.yaml",
        python_bin="python",
        device="cuda",
        save_samples=True,
        train_root=tmp_path,
    )

    assert evaluation.uses_checkpoint is False
    assert _command_value(evaluation.command, "--execution-mode") == "train"
    assert "--eval-only" not in evaluation.command
    assert _command_value(evaluation.command, "--steps") == "2000"
    assert _command_value(evaluation.command, "--particles") == "5"
    assert "--save-sample-artifacts" in evaluation.command


def test_ifno_inverse_uses_forward_checkpoint_identity(tmp_path: Path):
    checkpoint = tmp_path / "ifno-forward.pt"
    checkpoint.write_bytes(b"checkpoint")
    row = _row(
        baseline="ifno",
        task="inverse",
        task_group="full_inverse_main",
        execution_mode="eval_only",
        source_train_run_id="forward-run",
        source_train_run_fingerprint="c" * 64,
        source_train_seed=1,
        source_train_task="forward",
        source_train_baseline_code_sha256="d" * 64,
        baseline_code_sha256="e" * 64,
        checkpoint_path=str(checkpoint),
        num_sensors=0,
        sensor_mode="none",
    )
    evaluation = build_evaluation_run(
        row,
        {"status": "success", "checkpoint_path": str(checkpoint)},
        tmp_path,
        test_file="poisson/id.mat",
        test_size=1000,
        eval_root=tmp_path / "evaluations",
        eval_tag="id",
        data_root=tmp_path,
        config=tmp_path / "paper.yaml",
        python_bin="python",
        device="cuda",
        save_samples=False,
        train_root=tmp_path,
    )

    assert _command_value(evaluation.command, "--source-train-run-id") == "forward-run"
    assert _command_value(evaluation.command, "--source-train-task") == "forward"
    assert (
        _command_value(evaluation.command, "--source-train-baseline-code-sha256")
        == "d" * 64
    )


def test_completed_summary_must_match_requested_evaluation(tmp_path: Path):
    row = _row(baseline="pc_bnn", task="sparse_inverse", task_group="sparse_inverse_main")
    evaluation = build_evaluation_run(
        row,
        {"status": "success", "checkpoint_path": ""},
        tmp_path,
        test_file="poisson/id.mat",
        test_size=1000,
        eval_root=tmp_path / "evaluations",
        eval_tag="id",
        data_root=tmp_path,
        config=tmp_path / "paper.yaml",
        python_bin="python",
        device="cuda",
        save_samples=False,
        train_root=tmp_path,
    )
    summary_path = evaluation.output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary = {
        "status": "success",
        "baseline": "pc_bnn",
        "pde": "poisson",
        "task": "sparse_inverse",
        "run_id": _command_value(evaluation.command, "--run-id"),
        "seed": 1,
        "test_size": 1000,
        "test_requested_size": 1000,
        "batch_size": 16,
        "metric_granularity": "per_sample",
        "execution_mode": "train",
        "eval_only": False,
        "data_files_json": _command_value(evaluation.command, "--data-files-json"),
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert successful_summary(summary_path, evaluation)

    summary["test_size"] = 999
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert not successful_summary(summary_path, evaluation)


def test_completed_multicondition_summary_must_match_condition(tmp_path: Path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    row = _row(
        task="sparse_solution_multicondition",
        task_group="sparse_solution_multicondition_eval_both",
        execution_mode="eval_only",
        condition_mode="both",
        condition_probabilities={
            "a_only": 0.3333333333,
            "u_only": 0.3333333333,
            "both": 0.3333333334,
        },
        source_train_run_id="multicondition-train",
        source_train_run_fingerprint="d" * 64,
        source_train_seed=1,
        source_train_task="sparse_solution_multicondition",
        checkpoint_path=str(checkpoint),
    )
    evaluation = build_evaluation_run(
        row,
        {"status": "success", "checkpoint_path": str(checkpoint)},
        tmp_path,
        test_file="poisson/id.mat",
        test_size=1000,
        eval_root=tmp_path / "evaluations",
        eval_tag="id",
        data_root=tmp_path,
        config=tmp_path / "paper.yaml",
        python_bin="python",
        device="cuda",
        save_samples=False,
        train_root=tmp_path,
    )
    summary_path = evaluation.output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary = {
        "status": "success",
        "baseline": "recfno",
        "pde": "poisson",
        "task": "sparse_solution_multicondition",
        "run_id": _command_value(evaluation.command, "--run-id"),
        "seed": 1,
        "test_size": 1000,
        "test_requested_size": 1000,
        "batch_size": 16,
        "metric_granularity": "per_sample",
        "execution_mode": "eval_only",
        "eval_only": True,
        "condition_mode": "both",
        "data_files_json": _command_value(evaluation.command, "--data-files-json"),
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert successful_summary(summary_path, evaluation)

    summary["condition_mode"] = "mixed"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert not successful_summary(summary_path, evaluation)
