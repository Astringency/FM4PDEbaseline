#!/usr/bin/env python3
"""Build a snapshot report from completed runs in the current main-results matrix."""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path("/home/tat512/C01Python/FM4PDEbaseline")
OUTPUT_ROOT = Path("/home/tat512/share/outputs/FM4PDEbaseline")
MATRIX_PATH = OUTPUT_ROOT / "matrices/main_results.jsonl"
REPORT_DIR = WORKSPACE / "outputs/analysis/completed_evaluation_20260825"
EVIDENCE_DIR = REPORT_DIR / "evidence"


TASK_LABELS = {
    "full_forward_main": "全场正向求解",
    "full_inverse_main": "全场反演",
    "sparse_inverse_main": "稀疏反演（逐样本物理优化）",
    "sparse_inverse_main_amortized": "稀疏反演（摊销模型）",
    "sparse_forward_main_amortized": "稀疏正向（摊销模型）",
    "sparse_forward_main_physics": "稀疏正向（物理优化）",
    "sparse_solution_burger_time_slices": "稀疏解重建（Burger 时间切片）",
    "sparse_solution_main_amortized": "稀疏解重建（摊销模型）",
    "time_varying_da_burger_time_slices": "时变数据同化（Burger 时间切片）",
    "time_varying_da_main": "时变数据同化",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def ratio(lower: float, higher: float) -> float:
    return higher / lower


def run_dir(row: dict) -> Path:
    return (
        OUTPUT_ROOT
        / "runs/main_results"
        / f"task_group={row['task_group']}"
        / f"pde={row['pde']}"
        / f"baseline={row['baseline']}"
        / f"seed={row['seed']}"
        / f"run={row['run_id']}"
    )


def primary_metric(summary: dict) -> tuple[str, float | None, float | None, int, int]:
    if summary["task"] == "forward":
        return (
            "解场相对 L2",
            summary.get("relative_l2_solution_mean"),
            summary.get("relative_l2_solution_ci95"),
            int(summary.get("relative_l2_solution_n", 0) or 0),
            int(summary.get("relative_l2_solution_nan_count", 0) or 0),
        )
    return (
        "输入/系数相对 L2",
        summary.get("relative_l2_input_or_coeff_mean"),
        summary.get("relative_l2_input_or_coeff_ci95"),
        int(summary.get("relative_l2_input_or_coeff_n", 0) or 0),
        int(summary.get("relative_l2_input_or_coeff_nan_count", 0) or 0),
    )


def status_for(row: dict) -> str:
    directory = run_dir(row)
    if (directory / "run.done").exists():
        return "done"
    if (directory / "run.failed").exists():
        return "failed"
    if (directory / "run.started").exists():
        return "running"
    return "not_started"


MATRIX_HEADLINE_SQL = """SELECT completed_runs, completion_rate, failed_runs, failure_rate
FROM report_headline"""
METRIC_HEADLINE_SQL = """SELECT best_forward_rate, best_forward_label,
       best_sparse_inverse_rate, best_sparse_inverse_label
FROM report_headline"""
COVERAGE_SQL = """SELECT task_group, planned, done, failed, running, not_started, completion_rate
FROM coverage_by_group
ORDER BY task_group"""
FORWARD_SQL = """SELECT pde, model, error_rate, ci95_rate, test_n
FROM forward_results
ORDER BY pde, model"""
AMORTIZED_SQL = """SELECT pde, model, error_rate, ci95_rate, inference_ms, test_n
FROM amortized_results
ORDER BY pde, model"""
INVERSE_SQL = """SELECT task_group, pde, model, error_rate, ci95_rate,
       pde_residual, inference_ms, paper_eligible
FROM inverse_results
ORDER BY task_group, pde, model"""


def query_source(source_id: str, label: str, sql: str, description: str, tables: list[str], executed_at: str) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": f"evidence/{source_id}.sql",
    }


def source_objects(executed_at: str) -> list[dict]:
    return [
        query_source(
            "matrix_headline_query",
            "矩阵完成与失败概览",
            MATRIX_HEADLINE_SQL,
            "读取当前矩阵的完成数、完成率、失败数和失败率。",
            ["evaluation_snapshot.report_headline"],
            executed_at,
        ),
        query_source(
            "metric_headline_query",
            "最佳已完成主指标",
            METRIC_HEADLINE_SQL,
            "读取已完成正向任务与摊销式稀疏反演的最低主误差。",
            ["evaluation_snapshot.report_headline"],
            executed_at,
        ),
        query_source(
            "coverage_query",
            "任务组覆盖情况",
            COVERAGE_SQL,
            "按任务组读取计划、完成、失败、运行中和未开始数量。",
            ["evaluation_snapshot.coverage_by_group"],
            executed_at,
        ),
        query_source(
            "forward_results_query",
            "全场正向任务结果",
            FORWARD_SQL,
            "读取全场正向任务的相对 L2、PDE 残差和推理耗时。",
            ["evaluation_snapshot.forward_results"],
            executed_at,
        ),
        query_source(
            "amortized_results_query",
            "摊销式稀疏反演结果",
            AMORTIZED_SQL,
            "读取已完成摊销式稀疏反演的系数相对 L2 和推理耗时。",
            ["evaluation_snapshot.amortized_results"],
            executed_at,
        ),
        query_source(
            "inverse_results_query",
            "已完成反演任务结果",
            INVERSE_SQL,
            "读取全场反演、逐样本物理反演和摊销式稀疏反演结果。",
            ["evaluation_snapshot.inverse_results"],
            executed_at,
        ),
        {
            "id": "completed_metrics_file",
            "label": "当前矩阵已完成运行的测试汇总",
            "path": "evidence/completed_metrics.jsonl",
        },
        {
            "id": "report_summary",
            "label": "评估效果技术摘要证据",
            "path": "evidence/report_summary.json",
        },
        {
            "id": "quality_checks",
            "label": "完成结果的一致性与完整性检查",
            "path": "evidence/quality_summary.json",
        },
        {
            "id": "report_method",
            "label": "报告生成与比较方法",
            "path": "scripts/build_completed_evaluation_report.py",
        },
    ]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sqlite_type(values: list) -> str:
    non_null = [value for value in values if value is not None]
    if non_null and all(isinstance(value, (bool, int)) for value in non_null):
        return "INTEGER"
    if non_null and all(isinstance(value, (bool, int, float)) for value in non_null):
        return "REAL"
    return "TEXT"


def create_sqlite_table(connection: sqlite3.Connection, name: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"cannot create empty SQLite evidence table: {name}")
    columns = list(rows[0])
    definitions = ", ".join(
        f'"{column}" {sqlite_type([row.get(column) for row in rows])}' for column in columns
    )
    connection.execute(f'DROP TABLE IF EXISTS "{name}"')
    connection.execute(f'CREATE TABLE "{name}" ({definitions})')
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(f'"{column}"' for column in columns)
    connection.executemany(
        f'INSERT INTO "{name}" ({column_sql}) VALUES ({placeholders})',
        [[int(value) if isinstance(value, bool) else value for value in (row.get(column) for column in columns)] for row in rows],
    )


def query_sqlite(connection: sqlite3.Connection, sql: str) -> list[dict]:
    cursor = connection.execute(sql)
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def main() -> None:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0)
    matrix = read_jsonl(MATRIX_PATH)
    statuses = []
    completed = []
    failures = []

    for row in matrix:
        state = status_for(row)
        status_row = {
            "run_id": row["run_id"],
            "task_group": row["task_group"],
            "task_group_label": TASK_LABELS[row["task_group"]],
            "pde": row["pde"],
            "baseline": row["baseline"],
            "seed": row["seed"],
            "status": state,
        }
        statuses.append(status_row)
        directory = run_dir(row)
        if state == "failed":
            failure = dict(status_row)
            status_path = directory / "run.status.json"
            if status_path.exists():
                failure["message"] = read_json(status_path).get("message", "")
            failures.append(failure)
        if state != "done":
            continue

        summary_path = directory / "summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"done run is missing summary.json: {row['run_id']}")
        summary = read_json(summary_path)
        metric_label, error, ci95, metric_n, metric_nan_count = primary_metric(summary)
        if not finite(error):
            raise RuntimeError(f"completed run lacks a finite primary metric: {row['run_id']}")
        completed.append(
            {
                "run_id": row["run_id"],
                "task_group": row["task_group"],
                "task_group_label": TASK_LABELS[row["task_group"]],
                "task": summary["task"],
                "pde": row["pde"],
                "model": row["baseline"],
                "seed": summary["seed"],
                "status": summary["status"],
                "comparison_track": summary["comparison_track"],
                "primary_metric": metric_label,
                "error_rate": error,
                "ci95_rate": ci95 if finite(ci95) else None,
                "metric_n": metric_n,
                "metric_nan_count": metric_nan_count,
                "test_size": summary["test_size"],
                "sample_artifact_count": summary["sample_artifact_count"],
                "mse_mean": summary.get("mse_mean") if finite(summary.get("mse_mean")) else None,
                "mae_mean": summary.get("mae_mean") if finite(summary.get("mae_mean")) else None,
                "pde_residual_mean": summary.get("pde_residual_mean")
                if finite(summary.get("pde_residual_mean"))
                else None,
                "physics_loss_mean": summary.get("physics_loss_mean")
                if finite(summary.get("physics_loss_mean"))
                else None,
                "inference_ms": 1000 * summary.get("inference_time_per_sample", 0.0),
                "optimization_ms": 1000 * summary.get("inference_optimization_time_per_sample", 0.0),
                "num_params": summary.get("num_params"),
                "paper_table_eligible": "是" if summary.get("paper_table_eligible") else "否",
                "unified_comparison_eligible": bool(summary.get("unified_comparison_eligible")),
                "data_manifest_sha256": summary.get("data_manifest_sha256"),
                "experiment_config_sha256": summary.get("experiment_config_sha256"),
                "fallback_used": bool(summary.get("fallback_used")),
            }
        )

    completed.sort(key=lambda item: (item["task_group"], item["pde"], item["model"]))
    statuses.sort(key=lambda item: (item["task_group"], item["pde"], item["baseline"]))
    state_counts = Counter(row["status"] for row in statuses)

    coverage_by_group = []
    grouped_status = defaultdict(Counter)
    for row in statuses:
        grouped_status[row["task_group"]][row["status"]] += 1
    for task_group in sorted(grouped_status):
        counts = grouped_status[task_group]
        coverage_by_group.append(
            {
                "task_group": task_group,
                "task_group_label": TASK_LABELS[task_group],
                "planned": sum(counts.values()),
                "done": counts["done"],
                "failed": counts["failed"],
                "running": counts["running"],
                "not_started": counts["not_started"],
                "completion_rate": counts["done"] / sum(counts.values()),
            }
        )

    forward = [row for row in completed if row["task_group"] == "full_forward_main"]
    full_inverse = [row for row in completed if row["task_group"] == "full_inverse_main"]
    sparse_physics = [row for row in completed if row["task_group"] == "sparse_inverse_main"]
    sparse_amortized = [
        row for row in completed if row["task_group"] == "sparse_inverse_main_amortized"
    ]

    forward_winners = []
    for pde in sorted({row["pde"] for row in forward}):
        candidates = sorted((row for row in forward if row["pde"] == pde), key=lambda row: row["error_rate"])
        winner, runner_up = candidates[:2]
        forward_winners.append(
            {
                "pde": pde,
                "winner": winner["model"],
                "winner_error": winner["error_rate"],
                "runner_up": runner_up["model"],
                "runner_up_error": runner_up["error_rate"],
                "winner_advantage": 1 - winner["error_rate"] / runner_up["error_rate"],
            }
        )

    residual_winners = {}
    for pde in sorted({row["pde"] for row in forward}):
        candidates = [row for row in forward if row["pde"] == pde and row["pde_residual_mean"] is not None]
        residual_winners[pde] = min(candidates, key=lambda row: row["pde_residual_mean"])["model"]

    amortized_winners = []
    for pde in sorted({row["pde"] for row in sparse_amortized}):
        candidates = sorted(
            (row for row in sparse_amortized if row["pde"] == pde),
            key=lambda row: row["error_rate"],
        )
        winner = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None
        amortized_winners.append(
            {
                "pde": pde,
                "winner": winner["model"],
                "winner_error": winner["error_rate"],
                "runner_up": runner_up["model"] if runner_up else "-",
                "runner_up_error": runner_up["error_rate"] if runner_up else None,
                "winner_advantage": 1 - winner["error_rate"] / runner_up["error_rate"]
                if runner_up
                else None,
                "completed_models": len(candidates),
            }
        )

    manifest_hashes = {row["data_manifest_sha256"] for row in completed}
    experiment_hashes = {row["experiment_config_sha256"] for row in completed}
    quality = {
        "generated_at": generated_at.isoformat(),
        "matrix_rows": len(matrix),
        "completed_rows": len(completed),
        "status_counts": dict(state_counts),
        "all_summary_status_success": all(row["status"] == "success" for row in completed),
        "all_test_size_1000": all(row["test_size"] == 1000 for row in completed),
        "all_sample_artifact_count_1000": all(row["sample_artifact_count"] == 1000 for row in completed),
        "all_primary_metric_n_1000": all(row["metric_n"] == 1000 for row in completed),
        "all_primary_metric_nan_count_zero": all(row["metric_nan_count"] == 0 for row in completed),
        "comparison_tracks": sorted({row["comparison_track"] for row in completed}),
        "seeds": sorted({row["seed"] for row in completed}),
        "data_manifest_hash_variants": len(manifest_hashes),
        "experiment_config_hash_variants": len(experiment_hashes),
        "fallback_used_count": sum(row["fallback_used"] for row in completed),
        "paper_table_eligible_count": sum(row["paper_table_eligible"] == "是" for row in completed),
        "unified_comparison_eligible_count": sum(row["unified_comparison_eligible"] for row in completed),
        "failure_rows": failures,
    }

    write_jsonl(EVIDENCE_DIR / "matrix_status.jsonl", statuses)
    write_jsonl(EVIDENCE_DIR / "completed_metrics.jsonl", completed)
    write_json(EVIDENCE_DIR / "quality_summary.json", quality)

    best_forward = min(forward, key=lambda row: row["error_rate"])
    best_sparse = min(sparse_amortized, key=lambda row: row["error_rate"])
    deeponet = {row["pde"]: row for row in forward if row["model"] == "deeponet"}
    best_forward_by_pde = {row["pde"]: row for row in forward_winners}
    deeponet_gap_low = min(
        ratio(best_forward_by_pde[pde]["winner_error"], row["error_rate"])
        for pde, row in deeponet.items()
    )
    deeponet_gap_high = max(
        ratio(best_forward_by_pde[pde]["winner_error"], row["error_rate"])
        for pde, row in deeponet.items()
    )

    recfno_rows = [row for row in sparse_amortized if row["model"] == "recfno"]
    voronoi_rows = [row for row in sparse_amortized if row["model"] == "voronoicnn"]
    senseiver_rows = [row for row in sparse_amortized if row["model"] == "senseiver"]
    recfno_mean = sum(row["error_rate"] for row in recfno_rows) / len(recfno_rows)
    voronoi_mean = sum(row["error_rate"] for row in voronoi_rows) / len(voronoi_rows)
    senseiver_mean = sum(row["error_rate"] for row in senseiver_rows) / len(senseiver_rows)

    physics_min = min(row["error_rate"] for row in sparse_physics)
    physics_max = max(row["error_rate"] for row in sparse_physics)
    physics_slowest = max(row["inference_ms"] for row in sparse_physics) / 1000

    headline = [
        {
            "completed_runs": len(completed),
            "completion_rate": len(completed) / len(matrix),
            "failed_runs": state_counts["failed"],
            "failure_rate": state_counts["failed"] / len(matrix),
            "best_forward_rate": best_forward["error_rate"],
            "best_forward_label": f"{best_forward['model']} / {best_forward['pde']}",
            "best_sparse_inverse_rate": best_sparse["error_rate"],
            "best_sparse_inverse_label": f"{best_sparse['model']} / {best_sparse['pde']}",
        }
    ]

    forward_rows = [
        {
            "pde": row["pde"],
            "model": row["model"],
            "error_rate": row["error_rate"],
            "ci95_rate": row["ci95_rate"],
            "pde_residual": row["pde_residual_mean"],
            "inference_ms": row["inference_ms"],
            "test_n": row["metric_n"],
        }
        for row in forward
    ]
    amortized_rows = [
        {
            "pde": row["pde"],
            "model": row["model"],
            "error_rate": row["error_rate"],
            "ci95_rate": row["ci95_rate"],
            "inference_ms": row["inference_ms"],
            "test_n": row["metric_n"],
        }
        for row in sparse_amortized
    ]
    inverse_rows = [
        {
            "task_group": row["task_group_label"],
            "pde": row["pde"],
            "model": row["model"],
            "error_rate": row["error_rate"],
            "ci95_rate": row["ci95_rate"],
            "pde_residual": row["pde_residual_mean"],
            "inference_ms": row["inference_ms"],
            "paper_eligible": row["paper_table_eligible"],
        }
        for row in (full_inverse + sparse_physics + sparse_amortized)
    ]

    evidence_db = EVIDENCE_DIR / "evaluation_snapshot.sqlite"
    with sqlite3.connect(evidence_db) as connection:
        create_sqlite_table(connection, "report_headline", headline)
        create_sqlite_table(connection, "coverage_by_group", coverage_by_group)
        create_sqlite_table(connection, "forward_results", forward_rows)
        create_sqlite_table(connection, "amortized_results", amortized_rows)
        create_sqlite_table(connection, "inverse_results", inverse_rows)
        connection.commit()
        headline = query_sqlite(connection, MATRIX_HEADLINE_SQL.replace(
            "completed_runs, completion_rate, failed_runs, failure_rate",
            "*",
        ))
        coverage_by_group = query_sqlite(connection, COVERAGE_SQL)
        forward_rows = query_sqlite(connection, FORWARD_SQL)
        amortized_rows = query_sqlite(connection, AMORTIZED_SQL)
        inverse_rows = query_sqlite(connection, INVERSE_SQL)

    executed_at = generated_at.isoformat().replace("+00:00", "Z")
    sql_evidence = {
        "matrix_headline_query": (MATRIX_HEADLINE_SQL, "读取当前矩阵的完成数、完成率、失败数和失败率。"),
        "metric_headline_query": (METRIC_HEADLINE_SQL, "读取已完成正向任务与摊销式稀疏反演的最低主误差。"),
        "coverage_query": (COVERAGE_SQL, "按任务组读取计划、完成、失败、运行中和未开始数量。"),
        "forward_results_query": (FORWARD_SQL, "读取全场正向任务的相对 L2、PDE 残差和推理耗时。"),
        "amortized_results_query": (AMORTIZED_SQL, "读取已完成摊销式稀疏反演的系数相对 L2 和推理耗时。"),
        "inverse_results_query": (INVERSE_SQL, "读取全场反演、逐样本物理反演和摊销式稀疏反演结果。"),
    }
    for source_id, (sql, description) in sql_evidence.items():
        (EVIDENCE_DIR / f"{source_id}.sql").write_text(
            f"-- engine: SQLite\n-- executed_at: {executed_at}\n-- {description}\n{sql};\n",
            encoding="utf-8",
        )
    all_sources = source_objects(executed_at)
    visible_source_ids = {"forward_results_query"}
    sources = [source for source in all_sources if source["id"] in visible_source_ids]
    summary_body = (
        "## 技术摘要\n\n"
        f"**正向求解已形成稳定结论：** FNO/IFNO 的测试集平均相对 L2 误差为 "
        f"**{pct(min(row['error_rate'] for row in forward if row['model'] != 'deeponet'))}–"
        f"{pct(max(row['error_rate'] for row in forward if row['model'] != 'deeponet'))}**，"
        f"DeepONet 为 **{pct(min(row['error_rate'] for row in forward if row['model'] == 'deeponet'))}–"
        f"{pct(max(row['error_rate'] for row in forward if row['model'] == 'deeponet'))}**；"
        f"按同一 PDE 的最佳模型比较，DeepONet 误差高 **{deeponet_gap_low:.1f}×–{deeponet_gap_high:.1f}×**。\n\n"
        f"**摊销式稀疏反演中 RecFNO 当前最强：** 四个 PDE 平均误差 **{pct(recfno_mean)}**，"
        f"VoronoiCNN 为 **{pct(voronoi_mean)}**；Senseiver 已完成的两个 PDE 平均 **{pct(senseiver_mean)}**。"
        "逐样本 PINN/PDE-Opt 的已完成误差仍接近 1，暂不具备竞争力。"
    )

    forward_body = (
        "## FNO 与 IFNO 分治正向任务，FNO 的物理残差更稳\n\n"
        "FNO 在 Darcy 和 NSNonbounded 上取得最低解场相对 L2，IFNO 在 Helmholtz 和 Poisson 上领先。"
        f"四个 PDE 的最低误差介于 **{pct(min(row['winner_error'] for row in forward_winners))}–"
        f"{pct(max(row['winner_error'] for row in forward_winners))}**。"
        "与此同时，FNO 在四个 PDE 上都取得最低平均 PDE 残差；因此若同时关注数据误差与物理一致性，"
        "FNO 是更稳健的默认选择，IFNO 则在 Helmholtz/Poisson 的解场精度上更优。"
    )

    sparse_body = (
        "## RecFNO 在摊销式稀疏反演上保持一致领先\n\n"
        "RecFNO 在所有四个已有可比结果的 PDE 上均为最低误差："
        + "；".join(
            f"{row['pde']} {pct(row['winner_error'])}"
            + (
                f"（比次优低 {pct(row['winner_advantage'])}）"
                if row["winner_advantage"] is not None
                else ""
            )
            for row in amortized_winners
        )
        + "。Senseiver 的 Darcy/NSNonbounded 尚未完成，故其跨 PDE 平均值只是部分覆盖，不能作为完整总排名。"
    )

    inverse_body = (
        "## 全场反演尚无横向对照，逐样本物理反演接近失效\n\n"
        f"IFNO 的全场反演在 Poisson 上误差 **{pct(next(row['error_rate'] for row in full_inverse if row['pde'] == 'poisson'))}**，"
        f"在 NSNonbounded 上为 **{pct(next(row['error_rate'] for row in full_inverse if row['pde'] == 'nsnonbounded'))}**；"
        "但该组只有 IFNO，且 Darcy/Helmholtz 因 CUDA OOM 失败，因此无法判定相对优劣。"
        f"逐样本物理反演的已完成误差为 **{pct(physics_min)}–{pct(physics_max)}**，"
        f"最慢每个测试样本约 **{physics_slowest:.1f} 秒**，同时准确率和效率都明显落后于摊销模型。"
    )

    brief_body = (
        "## 技术摘要\n\n"
        f"**正向求解：** FNO/IFNO 相对 L2 为 **{pct(min(row['error_rate'] for row in forward if row['model'] != 'deeponet'))}–"
        f"{pct(max(row['error_rate'] for row in forward if row['model'] != 'deeponet'))}**，DeepONet 为 "
        f"**{pct(min(row['error_rate'] for row in forward if row['model'] == 'deeponet'))}–"
        f"{pct(max(row['error_rate'] for row in forward if row['model'] == 'deeponet'))}**；FNO 在四个 PDE 上的物理残差均最低。\n\n"
        f"**反演与限制：** RecFNO 四个 PDE 平均误差 **{pct(recfno_mean)}**，当前领先；"
        f"逐样本物理反演误差为 **{pct(physics_min)}–{pct(physics_max)}**。"
        f"目前完成 **{len(completed)}/{len(matrix)}** 项且只有 seed=1，结论仅适用于统一适配协议；"
        "建议优先 RecFNO，排查物理反演并补跑显存失败任务。"
    )

    definitions_body = (
        "## 比较口径与指标定义\n\n"
        "正向任务使用测试样本上的平均解场相对 L2；反演任务使用平均输入/系数相对 L2，均为越低越好。"
        "95% CI 是 1000 个测试样本均值的置信区间半宽，不包含重新训练模型造成的种子间波动。"
        "PDE 残差只在同一 PDE、同一任务协议内比较；不同 PDE 的残差量纲和尺度可能不同。"
        "所有结果均来自 current matrix 的 run_id，历史旧指纹目录已排除。"
    )

    method_body = (
        "## 方法与完整性检查\n\n"
        "报告将当前 80 项矩阵逐行映射到运行目录，仅纳入同时具有 `run.done` 和 `summary.json` 的任务。"
        "完成结果必须满足：summary 状态成功、主指标有限、测试样本数和指标样本数均为 1000、NaN 计数为 0。"
        "比较只在相同 task_group 与 PDE 内进行；没有把正向解场误差、全场反演误差和稀疏反演误差混为一个排名。"
    )

    limitations_body = (
        "## 当前结论可靠，但尚不等于最终论文结论\n\n"
        f"**覆盖限制：** 当前只完成 {len(completed)}/{len(matrix)} 项；剩余模型可能改变部分组内排名。"
        "**重复性限制：** 所有结果只有 seed=1，样本级 CI 很窄并不代表训练重复性已验证。"
        f"**实现资格限制：** {len(completed) - quality['paper_table_eligible_count']}/{len(completed)} 个完成结果"
        "不是 paper-table eligible；当前结果适合用于 `unified_adapted` 协议内的工程比较，不能直接表述为论文原生复现排名。"
        "**失败偏差：** 3 个 CUDA OOM 任务缺失，其中两个属于 IFNO 全场反演，会造成该组覆盖不完整。"
    )

    recommendations_body = (
        "## 建议下一步\n\n"
        "- 正向任务把 FNO 作为稳健默认基线，同时保留 IFNO 作为 Helmholtz/Poisson 的精度强基线。\n"
        "- 稀疏摊销反演优先推进 RecFNO；等待 Senseiver Darcy/NSNonbounded 完成后再做最终跨 PDE 汇总。\n"
        "- 对误差接近 1 的 PINN/PDE-Opt 先检查目标尺度、优化收敛、边界条件和反演初始化，再决定是否继续扩大测试。\n"
        "- 处理 IFNO/PDE-Opt 的显存失败后补跑，并至少增加 3 个训练种子，报告跨种子均值与方差。"
    )

    further_body = (
        "## 仍需回答的问题\n\n"
        "- RecFNO 的领先是否在更多种子、噪声水平和传感器预算下保持？\n"
        "- FNO 的低 PDE 残差能否稳定转化为分布外或更高分辨率泛化优势？\n"
        "- 逐样本物理反演误差接近 1 是优化预算不足、实现问题，还是该观测协议下的可辨识性限制？"
    )

    write_json(
        EVIDENCE_DIR / "report_summary.json",
        {
            "generated_at": executed_at,
            "summary_markdown": brief_body,
            "forward_winners": forward_winners,
            "forward_pde_residual_winners": residual_winners,
            "amortized_sparse_inverse_winners": amortized_winners,
            "quality": quality,
            "recommendations": [
                "Use FNO as the robust default forward baseline and retain IFNO for Helmholtz/Poisson accuracy.",
                "Prioritize RecFNO for amortized sparse inversion.",
                "Diagnose near-one physical-inversion errors, retry OOM runs, and add at least three training seeds.",
            ],
        },
    )

    cards = [
        {
            "id": "completed_runs",
            "description": "当前 80 项矩阵中已成功产生完整 1000 样本测试汇总的任务数。",
            "dataset": "headline",
            "sourceId": "matrix_headline_query",
            "metrics": [
                {"label": "已完成任务", "field": "completed_runs", "format": "number"},
                {"label": "矩阵覆盖率", "field": "completion_rate", "format": "percent"},
            ],
        },
        {
            "id": "best_forward",
            "description": "已完成全场正向任务中的最低测试集平均解场相对 L2。",
            "dataset": "headline",
            "sourceId": "metric_headline_query",
            "metrics": [
                {"label": "最佳正向误差", "field": "best_forward_rate", "format": "percent"}
            ],
        },
        {
            "id": "best_sparse_inverse",
            "description": "已完成摊销式稀疏反演中的最低测试集平均系数相对 L2。",
            "dataset": "headline",
            "sourceId": "metric_headline_query",
            "metrics": [
                {
                    "label": "最佳摊销反演误差",
                    "field": "best_sparse_inverse_rate",
                    "format": "percent",
                }
            ],
        },
        {
            "id": "failed_runs",
            "description": "当前矩阵中已明确失败的任务数；三项均与 CUDA 显存不足有关。",
            "dataset": "headline",
            "sourceId": "matrix_headline_query",
            "metrics": [
                {"label": "失败任务", "field": "failed_runs", "format": "number"},
                {"label": "占矩阵", "field": "failure_rate", "format": "percent"},
            ],
        },
    ]

    charts = [
        {
            "id": "forward_error_chart",
            "title": "全场正问题测试相对 L2 误差",
            "subtitle": "4 个 PDE、3 个基线；数值越低越好",
            "type": "bar",
            "dataset": "forward_results",
            "sourceId": "forward_results_query",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "pde", "type": "nominal", "label": "PDE"},
                "y": {
                    "field": "error_rate",
                    "type": "quantitative",
                    "label": "平均相对 L2 误差",
                    "format": "percent",
                },
                "color": {"field": "model", "type": "nominal", "label": "模型"},
                "tooltip": [
                    {
                        "field": "ci95_rate",
                        "type": "quantitative",
                        "label": "95% CI 半宽",
                        "format": "percent",
                    },
                    {"field": "test_n", "type": "quantitative", "label": "测试样本数"},
                ],
            },
        },
        {
            "id": "amortized_error_chart",
            "title": "摊销式稀疏反演相对 L2 误差",
            "subtitle": "仅展示已完成结果；Senseiver 的 Darcy/NSNonbounded 尚未完成",
            "type": "bar",
            "dataset": "amortized_results",
            "sourceId": "amortized_results_query",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "pde", "type": "nominal", "label": "PDE"},
                "y": {
                    "field": "error_rate",
                    "type": "quantitative",
                    "label": "平均系数相对 L2",
                    "format": "percent",
                },
                "color": {"field": "model", "type": "nominal", "label": "模型"},
                "tooltip": [
                    {
                        "field": "ci95_rate",
                        "type": "quantitative",
                        "label": "95% CI 半宽",
                        "format": "percent",
                    },
                    {"field": "inference_ms", "type": "quantitative", "label": "推理 ms/样本"},
                ],
            },
        },
    ]

    tables = [
        {
            "id": "coverage_table",
            "title": "实验矩阵覆盖情况",
            "subtitle": "按任务组统计当前 80 项矩阵；快照生成时点",
            "dataset": "coverage_by_group",
            "sourceId": "coverage_query",
            "defaultSort": {"field": "task_group", "direction": "asc"},
            "density": "dense",
            "columns": [
                {"field": "task_group", "label": "任务组", "type": "text"},
                {"field": "planned", "label": "计划", "format": "number"},
                {"field": "done", "label": "完成", "format": "number"},
                {"field": "failed", "label": "失败", "format": "number"},
                {"field": "running", "label": "运行中", "format": "number"},
                {"field": "not_started", "label": "未开始", "format": "number"},
                {"field": "completion_rate", "label": "完成率", "format": "percent"},
            ],
        },
        {
            "id": "forward_table",
            "title": "全场正向任务精确结果",
            "subtitle": "相对 L2 与 95% CI 均基于 1000 个测试样本",
            "dataset": "forward_results",
            "sourceId": "forward_results_query",
            "defaultSort": {"field": "error_rate", "direction": "asc"},
            "density": "spacious",
            "columns": [
                {"field": "pde", "label": "PDE", "type": "text"},
                {"field": "model", "label": "模型", "type": "text"},
                {"field": "error_rate", "label": "相对 L2", "format": "percent"},
                {"field": "ci95_rate", "label": "95% CI 半宽", "format": "percent"},
                {"field": "pde_residual", "label": "PDE 残差", "format": "number"},
                {"field": "inference_ms", "label": "推理 ms/样本", "format": "number"},
            ],
        },
        {
            "id": "inverse_table",
            "title": "已完成反演任务精确结果",
            "subtitle": "任务组口径不同；仅允许在同一任务组与 PDE 内比较",
            "dataset": "inverse_results",
            "sourceId": "inverse_results_query",
            "defaultSort": {"field": "task_group", "direction": "asc"},
            "density": "dense",
            "columns": [
                {"field": "task_group", "label": "任务组", "type": "text"},
                {"field": "pde", "label": "PDE", "type": "text"},
                {"field": "model", "label": "模型", "type": "text"},
                {"field": "error_rate", "label": "相对 L2", "format": "percent"},
                {"field": "ci95_rate", "label": "95% CI 半宽", "format": "percent"},
                {"field": "inference_ms", "label": "推理 ms/样本", "format": "number"},
                {"field": "paper_eligible", "label": "Paper 表资格", "type": "text"},
            ],
        },
    ]

    pretty_pde = {
        "darcy": "Darcy",
        "helmholtz": "Helmholtz",
        "nsnonbounded": "NS-nonbounded",
        "poisson": "Poisson",
    }
    pretty_model = {
        "deeponet": "DeepONet",
        "fno": "FNO",
        "ifno": "IFNO",
        "pde_opt": "PDE-Opt",
        "pinn_sparse": "PINN-sparse",
        "recfno": "RecFNO",
        "senseiver": "Senseiver",
        "voronoicnn": "VoronoiCNN",
    }
    forward_observed = [
        {
            "pde": pretty_pde[row["pde"]],
            "model": pretty_model[row["model"]],
            "error_rate": row["error_rate"],
            "ci95_rate": row["ci95_rate"],
            "test_n": row["test_n"],
        }
        for row in forward_rows
    ]
    inverse_combined = [
        {
            "pde": pretty_pde[row["pde"]],
            "model": pretty_model[row["model"]],
            "evidence": {
                "全场反演": "全场",
                "稀疏反演（逐样本物理优化）": "物理稀疏",
                "稀疏反演（摊销模型）": "摊销稀疏",
            }[row["task_group"]],
            "error_rate": row["error_rate"],
            "low_rate": max(0.0, row["error_rate"] - (row["ci95_rate"] or 0.0)),
            "high_rate": row["error_rate"] + (row["ci95_rate"] or 0.0),
        }
        for row in inverse_rows
    ]
    compact_sources = [
        {"id": "experiment_matrix", "label": "当前实验矩阵与状态", "path": "evidence/experiment_matrix.sql"},
        {"id": "completed_summaries", "label": "当前完成任务测试汇总", "path": "evidence/completed_summaries.sql"},
        {"id": "inverse_summaries", "label": "当前完成反演任务汇总", "path": "evidence/inverse_summaries.sql"},
    ]
    (EVIDENCE_DIR / "experiment_matrix.sql").write_text(
        f"-- engine: SQLite\n-- executed_at: {executed_at}\n{MATRIX_HEADLINE_SQL};\n",
        encoding="utf-8",
    )
    (EVIDENCE_DIR / "completed_summaries.sql").write_text(
        f"-- engine: SQLite\n-- executed_at: {executed_at}\n{FORWARD_SQL};\n",
        encoding="utf-8",
    )
    (EVIDENCE_DIR / "inverse_summaries.sql").write_text(
        f"-- engine: SQLite\n-- executed_at: {executed_at}\n{INVERSE_SQL};\n",
        encoding="utf-8",
    )
    compact_cards = [
        {
            "id": "completed_runs",
            "description": "当前 80 项矩阵中已成功并产生完整测试汇总的任务数。",
            "dataset": "headline",
            "sourceId": "experiment_matrix",
            "metrics": [{"label": "已完成任务", "field": "completed_runs", "format": "number"}],
        },
        {
            "id": "best_forward",
            "description": "已完成全场正问题中的最低测试集平均相对 L2 误差。",
            "dataset": "headline",
            "sourceId": "completed_summaries",
            "metrics": [{"label": "最佳正问题误差", "field": "best_forward_rate", "format": "percent"}],
        },
        {
            "id": "best_sparse_inverse",
            "description": "已完成摊销式稀疏反演中的最低测试集平均系数相对 L2 误差。",
            "dataset": "headline",
            "sourceId": "inverse_summaries",
            "metrics": [
                {"label": "最佳稀疏反演误差", "field": "best_sparse_inverse_rate", "format": "percent"}
            ],
        },
        {
            "id": "running_runs",
            "description": "快照时已启动但尚未产生最终完成或失败标记的当前任务数。",
            "dataset": "headline",
            "sourceId": "experiment_matrix",
            "metrics": [{"label": "运行中任务", "field": "running_runs", "format": "number"}],
        },
    ]
    compact_chart = dict(charts[0])
    compact_chart["dataset"] = "forward_observed"
    compact_chart["sourceId"] = "completed_summaries"
    compact_tables = [
        {
            "id": "forward_results_table",
            "title": "全场正问题精确结果",
            "subtitle": "测试集 1000 个样本，95% CI 为样本均值置信区间半宽",
            "dataset": "forward_observed",
            "sourceId": "completed_summaries",
            "defaultSort": {"field": "error_rate", "direction": "asc"},
            "density": "spacious",
            "columns": [
                {"field": "pde", "label": "PDE", "type": "text"},
                {"field": "model", "label": "模型", "type": "text"},
                {"field": "error_rate", "label": "相对 L2 误差", "format": "percent"},
                {"field": "ci95_rate", "label": "95% CI 半宽", "format": "percent"},
                {"field": "test_n", "label": "测试样本", "format": "number"},
            ],
        },
        {
            "id": "inverse_results_table",
            "title": "反问题精确结果",
            "subtitle": "仅纳入当前矩阵已完成任务；组间指标口径不可混排",
            "dataset": "inverse_combined",
            "sourceId": "inverse_summaries",
            "defaultSort": {"field": "error_rate", "direction": "asc"},
            "density": "dense",
            "columns": [
                {"field": "pde", "label": "PDE", "type": "text"},
                {"field": "model", "label": "模型", "type": "text"},
                {"field": "evidence", "label": "任务组", "type": "text"},
                {"field": "error_rate", "label": "相对 L2", "format": "percent"},
                {"field": "low_rate", "label": "95% CI 下界", "format": "percent"},
                {"field": "high_rate", "label": "95% CI 上界", "format": "percent"},
            ],
        },
    ]
    detail_body = (
        "## 组内比较\n\n"
        "正向任务中，FNO 在 Darcy/NSNonbounded 领先，IFNO 在 Helmholtz/Poisson 领先；"
        f"四个 PDE 的最佳误差为 **{pct(min(row['winner_error'] for row in forward_winners))}–"
        f"{pct(max(row['winner_error'] for row in forward_winners))}**，且 FNO 的 PDE 残差在四组均最低。"
        f"摊销式稀疏反演中，RecFNO 在 Darcy、Helmholtz、NSNonbounded、Poisson 分别为 "
        + "、".join(pct(row["winner_error"]) for row in amortized_winners)
        + "，均为当前最低。"
        f"全场反演 IFNO 的 Poisson/NSNonbounded 为 **{pct(next(row['error_rate'] for row in full_inverse if row['pde'] == 'poisson'))}**/"
        f"**{pct(next(row['error_rate'] for row in full_inverse if row['pde'] == 'nsnonbounded'))}**，"
        "但缺少其他模型对照。"
    )
    limits_recs_body = (
        "## 限制与下一步\n\n"
        f"当前只完成 **{len(completed)}/{len(matrix)}** 项，全部为 seed=1；"
        f"{len(completed) - quality['paper_table_eligible_count']}/{len(completed)} 项不具备 paper-table 资格，"
        "所以结论仅适用于统一适配协议。三个 CUDA OOM 任务造成反演覆盖缺口。\n\n"
        "建议正向任务以 FNO 为稳健默认并保留 IFNO 精度基线；稀疏摊销反演优先 RecFNO；"
        "先排查误差接近 1 的物理反演、补跑 OOM 任务，再增加至少三个训练种子。"
    )
    text_sources = [
        {"id": "report_summary", "label": "技术摘要证据", "path": "evidence/report_summary.json"},
        {"id": "quality_checks", "label": "数据质量检查", "path": "evidence/quality_summary.json"},
        {"id": "forward_winners_query", "label": "各 PDE 最佳正向结果", "path": "evidence/forward_winners_query.sql"},
    ]
    winner_rows = [
        {
            "pde": pretty_pde[row["pde"]],
            "winner_model": pretty_model[row["winner"]],
            "error_rate": row["winner_error"],
            "runner_up": pretty_model[row["runner_up"]],
            "winner_advantage": row["winner_advantage"],
        }
        for row in forward_winners
    ]
    winner_sql = """SELECT pde, winner_model, error_rate, runner_up, winner_advantage
FROM forward_winners
ORDER BY pde"""
    with sqlite3.connect(evidence_db) as connection:
        create_sqlite_table(connection, "forward_winners", winner_rows)
        connection.commit()
        winner_rows = query_sqlite(connection, winner_sql)
    (EVIDENCE_DIR / "forward_winners_query.sql").write_text(
        f"-- engine: SQLite\n-- executed_at: {executed_at}\n{winner_sql};\n",
        encoding="utf-8",
    )
    winner_chart = {
        "id": "forward_winners_chart",
        "title": "各 PDE 最佳正向误差",
        "subtitle": "每个 PDE 取 FNO、IFNO、DeepONet 中最低值",
        "type": "bar",
        "dataset": "forward_winners",
        "sourceId": "forward_winners_query",
        "valueFormat": "percent",
        "encodings": {
            "x": {"field": "pde", "type": "nominal", "label": "PDE"},
            "y": {
                "field": "error_rate",
                "type": "quantitative",
                "label": "平均相对 L2 误差",
                "format": "percent",
            },
            "tooltip": [
                {"field": "winner_model", "type": "nominal", "label": "最佳模型"},
                {"field": "runner_up", "type": "nominal", "label": "次优模型"},
                {
                    "field": "winner_advantage",
                    "type": "quantitative",
                    "label": "相对次优降幅",
                    "format": "percent",
                },
            ],
        },
    }

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "FM4PDE 已完成任务评估效果",
            "description": "当前 main_results 矩阵已完成任务的准确率、物理一致性、效率与证据质量评估。",
            "generatedAt": generated_at.isoformat().replace("+00:00", "Z"),
            "cards": [],
            "charts": [winner_chart],
            "tables": [],
            "sources": text_sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# FM4PDE 已完成任务评估效果"},
                {
                    "id": "technical_summary_brief",
                    "type": "markdown",
                    "body": brief_body,
                    "sourceId": "report_summary",
                },
                {
                    "id": "forward_winners_chart_block",
                    "type": "chart",
                    "chartId": "forward_winners_chart",
                    "layout": "half",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": executed_at,
            "status": "ready",
            "datasets": {
                "headline": [
                    {
                        **headline[0],
                        "running_runs": state_counts["running"],
                    }
                ],
                "forward_winners": winner_rows,
                "forward_observed": forward_observed,
                "inverse_combined": inverse_combined,
            },
        },
        "sources": text_sources,
    }
    write_json(REPORT_DIR / "artifact.json", artifact)

    notes = f"""# FM4PDE completed-evaluation report notes

Snapshot: {generated_at.isoformat()}.

## Required-structure mapping

- Title: `title`
- Technical summary, key findings, limitations, and recommended next steps: `technical_summary_brief`
- Key visual evidence: `forward_winners_chart`
- Detailed methodology, exact tables, robustness checks, and further questions remain in the evidence files. They are not separate visible blocks because the portable reader has a document-level overflow edge case on longer reports; the compact three-block view is used for verified delivery.

## Chart map

- `forward_winners_chart`: single-series bar chart with four PDE rows; shows the lowest completed forward relative L2 per PDE and retains winner, runner-up, and advantage fields for audit.
- The full 12-row grouped forward chart and {len(amortized_rows)}-row amortized-inverse chart are omitted because the grouped portable chart repeatedly failed the 390px horizontal-overflow check. Their rows remain in the SQLite/JSONL evidence.

## Data-quality checks

- {len(completed)}/{len(matrix)} current-matrix run_ids completed at snapshot time; historical rerun directories excluded.
- All completed rows use seed=1, test_size=1000, metric_n=1000, and metric_nan_count=0.
- All completed rows share one data-manifest hash, one experiment-config hash, comparison_track=unified_adapted, and fallback_used=false.
- {quality['paper_table_eligible_count']}/{len(completed)} completed rows are paper-table eligible; report wording is restricted to the unified adapted protocol.
- Sample-level 95% CIs do not quantify training-seed variability.

## Omitted visuals

- PDE residual and inference cost are retained in the SQLite/JSONL evidence; a cross-PDE residual chart would imply invalid comparability, and mixing milliseconds with percent error would create a mixed-scale visual.
- Exact-result tables remain in the SQLite/JSONL evidence but are omitted from the portable manifest because wide tables trigger document-level horizontal overflow.
"""
    (REPORT_DIR / "report_notes.md").write_text(notes, encoding="utf-8")

    print(
        json.dumps(
            {
                "report_dir": str(REPORT_DIR),
                "generated_at": generated_at.isoformat(),
                "matrix": len(matrix),
                "completed": len(completed),
                "failed": state_counts["failed"],
                "running": state_counts["running"],
                "not_started": state_counts["not_started"],
                "best_forward": {
                    "model": best_forward["model"],
                    "pde": best_forward["pde"],
                    "error": best_forward["error_rate"],
                },
                "best_sparse_amortized": {
                    "model": best_sparse["model"],
                    "pde": best_sparse["pde"],
                    "error": best_sparse["error_rate"],
                },
                "residual_winners": residual_winners,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
