from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from openpyxl import Workbook

from scripts.audit_experiment_plan_v2 import (
    AuditInputError,
    build_audit,
    inspect_xlsx,
    write_audit_artifacts,
    write_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
MOUNTED_MATRIX = ROOT / "outputs/experiment_plan_v2/matrices/experiment_plan_v2.jsonl"
MOUNTED_WORKBOOK = ROOT / "outputs/experiment_plan_v2_summary.xlsx"
MOUNTED_SENSOR_EVIDENCE = ROOT / "outputs/sensor_generalization_retrain"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def audit_sources(tmp_path: Path) -> dict[str, Path]:
    """Create the confirmed 87-row overlap structure without mounted outputs."""

    matrix_path = tmp_path / "matrix.jsonl"
    workbook_path = tmp_path / "source.xlsx"
    sensor_root = tmp_path / "sensor_generalization"
    matrix_rows = []

    for index in range(87):
        run_id = f"run_{index:02d}"
        task_group = "full_forward_main"
        pde = "poisson"
        baseline = "fno"
        sensor_mode = "none"

        if 33 <= index <= 47:
            task_group = "sparse_forward_main_amortized"
            baseline = "recfno"
            sensor_mode = "random"
        elif 48 <= index <= 53:
            task_group = "sparse_forward_main_physics"
            baseline = "pinn_sparse" if index % 2 else "pde_opt"
            sensor_mode = "random"
        elif 54 <= index <= 57:
            task_group = "full_inverse_main" if index == 54 else "sparse_inverse_main"
            pde = "burger"
            baseline = "ifno" if index == 54 else "recfno"
            sensor_mode = "none" if index == 54 else "random"
        elif 58 <= index <= 67:
            task_group = "sparse_solution_main_physics"
            baseline = "pinn_sparse" if index % 2 else "pde_opt"
            sensor_mode = "random"
        elif 68 <= index <= 73:
            task_group = "full_inverse_main"
            baseline = "ifno"
        elif 74 <= index <= 85:
            task_group = "sparse_inverse_main"
            baseline = ("recfno", "senseiver", "voronoicnn")[index % 3]
            sensor_mode = "random"

        # Exactly two amortized Voronoi rows were created by the affected commit.
        commit_hash: str | None = "269fa323e7"
        if index in (33, 34):
            baseline = "voronoicnn"
            commit_hash = "3340de00718ea795"
        # Eight legacy rows overlap physics; six are legacy-only.
        if 58 <= index <= 65 or 68 <= index <= 73:
            commit_hash = None

        run_dir = tmp_path / "runs" / run_id
        run_dir.mkdir(parents=True)
        matrix_row = {
            "run_id": run_id,
            "output_dir": str(run_dir),
            "task_group": task_group,
            "task": "inverse" if "inverse" in task_group else "forward",
            "pde": pde,
            "baseline": baseline,
            "seed": 1,
            "sensor_mode": sensor_mode,
            "num_sensors": 500 if sensor_mode == "random" else 0,
            "train_size": 50_000,
            "val_size": 1_000,
            "test_size": 1_000,
            "skip_reason": "",
        }
        summary = {
            **matrix_row,
            "train_time": 0.0 if index <= 32 else (None if commit_hash is None else 1.0),
            "commit_hash": commit_hash,
            "config_hash": "config-hash" if commit_hash else "",
            "checkpoint_path": "/old/checkpoint.pt" if index <= 32 else "",
            "file_paths_summary": "{}",
            "requested_sensor_mode": sensor_mode,
            "effective_sensor_mode": sensor_mode,
            "effective_train_size": 49_000,
            "relative_l2_solution_mean": 0.1,
            "mse_mean": 0.01,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary))
        matrix_rows.append(matrix_row)

    matrix_path.write_text(
        "".join(json.dumps(row) + "\n" for row in matrix_rows), encoding="utf-8"
    )
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "runs"
    sheet.append(["run_id", "task_group"])
    for row in matrix_rows:
        sheet.append([row["run_id"], row["task_group"]])
    workbook.save(workbook_path)

    sensor_values = {
        "recfno": (0.0150162501, 0.2887927516, 0.2756855929),
        "senseiver": (0.0115569804, 0.0656374964, 0.0671960488),
    }
    for baseline, values in sensor_values.items():
        for sensor_seed, value in enumerate(values, start=1):
            summary_dir = (
                sensor_root
                / f"{baseline}_sparse_forward_poisson"
                / f"eval_testseed{sensor_seed}"
            )
            summary_dir.mkdir(parents=True)
            (summary_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "baseline": baseline,
                        "relative_l2_solution_mean": value,
                        "mse_mean": value**2,
                        "mask_id": f"mask-{sensor_seed}",
                        "checkpoint_path": f"/{baseline}.pt",
                    }
                )
            )

    return {
        "matrix": matrix_path,
        "workbook": workbook_path,
        "sensor_evidence": sensor_root,
    }


def _build_from_fixture(paths: dict[str, Path]) -> dict:
    return build_audit(
        matrix_path=paths["matrix"],
        source_workbook_path=paths["workbook"],
        sensor_generalization_root=paths["sensor_evidence"],
    )


def test_confirmed_cohort_is_fail_closed_with_expected_reason_counts(
    audit_sources: dict[str, Path],
) -> None:
    workbook_hash_before = _sha256(audit_sources["workbook"])
    audit = _build_from_fixture(audit_sources)

    assert audit["summary"]["row_counts"] == {
        "total": 87,
        "hard_excluded_unique": 74,
        "conditional_only": 13,
        "publishable": 0,
    }
    assert audit["summary"]["confirmed_reason_counts"] == {
        "stale_eval_only_artifact": 33,
        "sparse_forward_protocol_invalid": 21,
        "burger_inverse_target_leakage": 4,
        "sparse_solution_physics_hidden_truth": 10,
        "legacy_summary_schema": 14,
    }
    assert audit["summary"]["confirmed_detail_reason_counts"] == {
        "sparse_forward_wrong_input_normalization": 15,
        "sparse_forward_physics_hidden_target": 6,
        "sparse_forward_voronoicnn_hidden_target": 2,
    }
    assert audit["summary"]["legacy_schema_overlap_with_other_hard_exclusions"] == 8
    assert audit["summary"]["conditional_only_breakdown"] == {
        "fixed_sensor_layout_shared_across_splits": 12,
        "non_sensor_single_seed": 1,
    }
    assert sum(row["hard_excluded"] for row in audit["rows"]) == 74
    assert all(row["publishable"] is False for row in audit["rows"])
    assert _sha256(audit_sources["workbook"]) == workbook_hash_before

    multi_reason = next(
        row
        for row in audit["rows"]
        if row["task_group"] == "sparse_solution_main_physics"
        and row["commit_hash"] is None
    )
    assert {
        "sparse_solution_physics_hidden_truth",
        "legacy_summary_schema",
    }.issubset(multi_reason["hard_reason_tags"])
    assert multi_reason["source_workbook_sha256"] == workbook_hash_before
    assert multi_reason["summary_sha256"]
    assert "checkpoint_path" in multi_reason
    assert "config_hash" in multi_reason


def test_artifacts_are_machine_readable_and_report_sensor_generalization(
    audit_sources: dict[str, Path], tmp_path: Path
) -> None:
    audit = _build_from_fixture(audit_sources)
    workbook_hash_before = _sha256(audit_sources["workbook"])

    paths = write_audit_artifacts(audit, tmp_path / "audit")
    markdown_path = write_markdown(audit, tmp_path / "audit.md")

    summary = json.loads(Path(paths["summary_json"]).read_text())
    rows = json.loads(Path(paths["rows_json"]).read_text())
    with Path(paths["rows_csv"]).open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert paths["rows_xlsx"] is not None
    xlsx = inspect_xlsx(Path(paths["rows_xlsx"]))
    markdown = markdown_path.read_text()

    assert summary["row_counts"]["publishable"] == 0
    assert len(rows) == len(csv_rows) == xlsx["data_rows"] == 87
    assert xlsx["has_formulas"] is False
    assert all(row["publishable"] is False for row in rows)
    assert all(json.loads(row["reason_tags"]) for row in csv_rows)
    assert _sha256(audit_sources["workbook"]) == workbook_hash_before

    evidence = summary["sensor_generalization_evidence"]
    recfno_seed2 = next(
        row
        for row in evidence
        if row["baseline"] == "recfno" and row["test_sensor_seed"] == 2
    )
    senseiver_seed2 = next(
        row
        for row in evidence
        if row["baseline"] == "senseiver" and row["test_sensor_seed"] == 2
    )
    assert recfno_seed2["degradation_vs_seed1"] > 19
    assert senseiver_seed2["degradation_vs_seed1"] > 5
    assert recfno_seed2["summary_sha256"]
    assert "0 行可直接发表" in markdown
    assert "0.288793" in markdown
    assert "official_component_reuse" in markdown


def test_audit_refuses_a_matrix_that_is_not_the_source_workbook_cohort(
    audit_sources: dict[str, Path], tmp_path: Path
) -> None:
    truncated_matrix = tmp_path / "truncated.jsonl"
    truncated_matrix.write_text(
        "\n".join(audit_sources["matrix"].read_text().splitlines()[:-1]) + "\n"
    )

    with pytest.raises(AuditInputError, match="workbook/matrix row mismatch"):
        build_audit(
            matrix_path=truncated_matrix,
            source_workbook_path=audit_sources["workbook"],
            sensor_generalization_root=audit_sources["sensor_evidence"],
        )


@pytest.mark.skipif(
    not (MOUNTED_MATRIX.is_file() and MOUNTED_WORKBOOK.is_file()),
    reason="historical experiment outputs are not mounted",
)
def test_mounted_historical_cohort_matches_the_audit_contract() -> None:
    audit = build_audit(
        matrix_path=MOUNTED_MATRIX,
        source_workbook_path=MOUNTED_WORKBOOK,
        sensor_generalization_root=MOUNTED_SENSOR_EVIDENCE,
    )

    assert audit["summary"]["row_counts"] == {
        "total": 87,
        "hard_excluded_unique": 74,
        "conditional_only": 13,
        "publishable": 0,
    }
