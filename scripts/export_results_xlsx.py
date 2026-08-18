#!/usr/bin/env python
"""Export every ``experiment_plan_v2`` run summary into a single xlsx workbook.

Reads ``summary.json`` for each completed run from the matrix and writes one
row per run with the most useful task/configuration/metric fields.

Usage
-----
    OUTPUT_ROOT=/path/to/outputs python scripts/export_results_xlsx.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", ROOT / "outputs"))
MATRIX = OUTPUT_ROOT / "experiment_plan_v2" / "matrices" / "experiment_plan_v2.jsonl"
OUT_XLSX = OUTPUT_ROOT / "experiment_plan_v2_summary.xlsx"

COLUMNS = [
    "task_group",
    "pde",
    "task",
    "baseline",
    "seed",
    "train_size",
    "val_size",
    "test_size",
    "num_sensors",
    "sensor_mode",
    "noise_level",
    "scalar_param_mode",
    "steps",
    "refine_steps",
    "particles",
    "best_val_loss",
    "best_epoch",
    "mse_mean",
    "mse_std",
    "mae_mean",
    "mae_std",
    "relative_l2_solution_mean",
    "relative_l2_solution_std",
    "relative_l2_input_or_coeff_mean",
    "obs_mse_clean_mean",
    "pde_residual_mean",
    "bc_residual_mean",
    "ic_residual_mean",
    "physics_loss_mean",
    "train_time",
    "num_params",
    "backend_used",
    "capability_status",
    "implementation_mode_effective",
    "fallback_used",
    "paper_table_eligible",
    "adapter_status",
    "run_id",
]

SORT_KEYS = ["task_group", "pde", "task", "baseline", "seed"]


def remap_output_dir(row: dict) -> dict:
    for key in ("output_dir", "log_dir", "status_file"):
        value = row.get(key)
        if isinstance(value, str) and value.startswith("outputs/"):
            row[key] = str(OUTPUT_ROOT / value[len("outputs/"):])
    return row


def load_rows() -> list[dict]:
    rows = []
    with MATRIX.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("skip_reason"):
                continue
            rows.append(remap_output_dir(row))
    return rows


def main() -> None:
    records = []
    missing = []
    for row in load_rows():
        summary = Path(row["output_dir"]) / "summary.json"
        if not summary.exists():
            missing.append(row.get("run_id", "?"))
            continue
        data = json.loads(summary.read_text())
        records.append({col: data.get(col) for col in COLUMNS})

    if not records:
        raise SystemExit("No summary.json files found.")

    df = pd.DataFrame.from_records(records)
    for col in SORT_KEYS:
        if col in df.columns:
            df[col] = df[col].astype(str)
    df = df.sort_values(SORT_KEYS).reset_index(drop=True)

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="runs", index=False)

    print(json.dumps({
        "output": str(OUT_XLSX),
        "rows": len(df),
        "columns": len(df.columns),
        "missing_summaries": len(missing),
        "missing_run_ids": missing,
    }, indent=2))


if __name__ == "__main__":
    main()
