from __future__ import annotations

import json
from pathlib import Path

from scripts.run_eval import (
    DISTRIBUTION_TEST_FILES,
    build_evaluation_run,
    filter_rows,
    split_selection,
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
    assert "--eval-only" in evaluation.command
    assert "--no-save-sample-artifacts" in evaluation.command
    data_files = json.loads(_command_value(evaluation.command, "--data-files-json"))
    assert data_files == {
        "train": ["poisson/train.mat"],
        "test": ["poisson/rough.mat"],
    }


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
