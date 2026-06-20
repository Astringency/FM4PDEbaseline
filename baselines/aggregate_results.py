from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from baselines.capabilities import write_capability_matrix


BASE_GROUP_KEYS = [
    "experiment_kind",
    "ablation_factor",
    "task_group",
    "pde",
    "task",
    "baseline",
    "train_size",
    "scalar_param_mode",
    "normalize",
    "num_sensors",
    "sensor_budget_mode",
    "sensor_mode",
    "noise_level",
    "backend_used",
    "capability_status",
    "implementation_mode_effective",
    "paper_table_eligible",
]

BUDGET_GROUP_KEYS = [
    "steps",
    "refine_steps",
    "particles",
    "method_budget_label",
]

GROUP_KEYS = BASE_GROUP_KEYS + BUDGET_GROUP_KEYS

METRICS = [
    "relative_l2_solution",
    "relative_l2_input_or_coeff",
    "mse",
    "mae",
    "obs_mse",
    "obs_mse_clean",
    "obs_mse_noisy",
    "pde_residual",
    "bc_residual",
    "ic_residual",
    "physics_loss",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Aggregate FM4PDE baseline result JSONL files.")
    parser.add_argument("inputs", nargs="+", help="results_raw.jsonl or results_summary.jsonl files/directories")
    parser.add_argument("--output-dir", default="outputs/baselines/aggregate")
    parser.add_argument("--latex-tex", action="store_true", help="Also write latex_table.tex")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = _expand_inputs(args.inputs)
    rows = []
    for path in paths:
        rows.extend(_read_jsonl(path))
    summary_rows = aggregate_rows(rows)
    main_rows, supplement_rows = partition_rows_for_tables(rows)
    summary_main = aggregate_rows(main_rows)
    summary_supplement = aggregate_rows(supplement_rows)
    skipped_rows = _read_skipped_for_inputs(args.inputs)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "summary.csv", summary_rows)
    (out / "summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(out / "summary_main.csv", summary_main)
    (out / "summary_main.json").write_text(json.dumps(summary_main, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(out / "summary_supplement.csv", summary_supplement)
    (out / "summary_supplement.json").write_text(json.dumps(summary_supplement, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(out / "skipped_combinations.csv", skipped_rows)
    (out / "skipped_combinations.json").write_text(json.dumps(skipped_rows, indent=2, sort_keys=True), encoding="utf-8")
    tuning_rows = tuning_summary(rows)
    _write_csv(out / "tuning_summary.csv", tuning_rows)
    (out / "tuning_summary.json").write_text(json.dumps(tuning_rows, indent=2, sort_keys=True), encoding="utf-8")
    write_capability_matrix(out)
    latex_rows = _latex_rows(summary_main)
    _write_csv(out / "latex_table.csv", latex_rows)
    if args.latex_tex:
        (out / "latex_table.tex").write_text(_latex_tex(latex_rows), encoding="utf-8")
    print(
        json.dumps(
            {
                "inputs": [str(p) for p in paths],
                "groups": len(summary_rows),
                "main_groups": len(summary_main),
                "supplement_groups": len(summary_supplement),
                "skipped": len(skipped_rows),
                "tuning_groups": len(tuning_rows),
                "output_dir": str(out),
            },
            indent=2,
        )
    )


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[tuple[str, Any], ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key_fields = _group_keys_for_item(row)
        key = tuple((field, _norm(row.get(field, ""))) for field in key_fields)
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: _sort_key(kv[0])):
        result = {field: value for field, value in key}
        runtime_budget_group = result.get("ablation_factor") == "runtime_budget"
        result["grouping_budget_mode"] = "budget_grouped" if runtime_budget_group else "budget_recorded_only"
        result["budget_variation_warning"] = _budget_variation_warning(items, runtime_budget_group)
        for field in BUDGET_GROUP_KEYS:
            if field not in result:
                result[field] = _representative_value(items, field)
        residual_counts: Counter[str] = Counter()
        assimilation_counts: Counter[str] = Counter()
        for item in items:
            residual_counts.update(_parse_counts(item, "residual_mode_counts", "residual_mode"))
            assimilation_counts.update(_parse_counts(item, "assimilation_mode_counts", "assimilation_mode"))
        result["residual_mode_counts"] = json.dumps(dict(residual_counts), sort_keys=True)
        result["assimilation_mode_counts"] = json.dumps(dict(assimilation_counts), sort_keys=True)
        result["run_count"] = len(items)
        metric_modes: list[str] = []
        for metric in METRICS:
            stats, mode = _metric_stats_from_items(items, metric)
            metric_modes.append(mode)
            for suffix, value in stats.items():
                result[f"{metric}_{suffix}"] = value
        result["aggregation_mode"] = _aggregation_mode(metric_modes)
        row_warnings = sorted({str(item.get("aggregation_warning", "")) for item in items if item.get("aggregation_warning")})
        metric_warning = (
            "summary rows missing metric std/n; aggregated unweighted run-level means"
            if result["aggregation_mode"] == "run_level_summary_only_unweighted"
            else ""
        )
        warnings = [w for w in [metric_warning, *row_warnings] if w]
        result["aggregation_warning"] = "; ".join(warnings)
        out.append(result)
    return out


def partition_rows_for_tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main: list[dict[str, Any]] = []
    supplement: list[dict[str, Any]] = []
    for row in rows:
        issue = _main_eligibility_issue(row)
        if not issue:
            main.append(row)
            continue
        downgraded = dict(row)
        if _truthy(row.get("paper_table_eligible", False)):
            downgraded["paper_table_eligible"] = False
            downgraded["aggregation_warning"] = f"downgraded_to_supplement: {issue}"
        supplement.append(downgraded)
    return main, supplement


def tuning_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [row for row in rows if _has_finite(row.get("best_val_loss"))]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        key = (
            row.get("pde", ""),
            row.get("task", ""),
            row.get("baseline", ""),
            row.get("sensor_mode", ""),
            row.get("sensor_budget_mode", ""),
            row.get("num_sensors", ""),
            row.get("noise_level", ""),
            row.get("implementation_mode_effective", ""),
        )
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        best = min(items, key=lambda row: float(row.get("best_val_loss")))
        out.append(
            {
                "pde": key[0],
                "task": key[1],
                "baseline": key[2],
                "sensor_mode": key[3],
                "sensor_budget_mode": key[4],
                "num_sensors": key[5],
                "noise_level": key[6],
                "implementation_mode_effective": key[7],
                "best_val_loss": float(best.get("best_val_loss")),
                "best_epoch": best.get("best_epoch", ""),
                "config_hash": best.get("config_hash", ""),
                "selected_config_path": best.get("config_path", ""),
                "run_id": best.get("run_id", ""),
                "seed": best.get("seed", ""),
                "candidate_count": len(items),
            }
        )
    return out


def _main_eligibility_issue(row: dict[str, Any]) -> str:
    if not _truthy(row.get("paper_table_eligible", False)):
        return "paper_table_eligible=false"
    if _truthy(row.get("fallback_used", False)):
        return "fallback_used=true"
    status = str(row.get("capability_status", row.get("support_status", ""))).lower()
    if status not in {"native", "official_adapter"}:
        return f"capability_status={status or 'missing'}"
    adapter_status = str(row.get("adapter_status", "") or "").lower()
    if any(token in adapter_status for token in ("fallback", "local", "surrogate", "style", "adapted", "toy", "debug", "target_change")):
        return f"adapter_status={adapter_status}"
    mode = str(row.get("implementation_mode_effective", "") or "").lower()
    allowed = _eligible_modes_from_row(row)
    if mode not in allowed:
        return f"implementation_mode_effective={mode or 'missing'} not in {sorted(allowed)}"
    required = str(row.get("implementation_required", "") or "").lower()
    if required == "official" and mode == "official" and not _truthy(row.get("official_import_success", False)):
        return "official_import_success=false"
    if required == "official" and mode in {"official_architecture", "official_aligned"} and not _truthy(
        row.get("official_reimplementation_success", False)
    ):
        return "official_reimplementation_success=false"
    if required == "canonical_math" and mode != "canonical_math":
        return f"canonical_math required, got {mode or 'missing'}"
    if required == "adapted_allowed":
        return "adapted_allowed is supplement-only"
    return ""


def _eligible_modes_from_row(row: dict[str, Any]) -> set[str]:
    raw = row.get("eligible_implementation_modes", "")
    if isinstance(raw, (list, tuple, set)):
        modes = {str(x).lower() for x in raw}
        if modes:
            return modes
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                modes = {str(x).lower() for x in parsed}
                if modes:
                    return modes
        except Exception:
            modes = {part.strip().lower() for part in raw.replace(";", ",").split(",") if part.strip()}
            if modes:
                return modes
    required = str(row.get("implementation_required", "") or "").lower()
    if required == "canonical_math":
        return {"canonical_math"}
    if required == "official":
        modes = {"official"}
        if _truthy(row.get("official_architecture_allowed", False)):
            modes.add("official_architecture")
        if _truthy(row.get("official_aligned_allowed", False)):
            modes.add("official_aligned")
        return modes
    return set()


def _group_keys_for_item(row: dict[str, Any]) -> list[str]:
    if row.get("ablation_factor") == "runtime_budget":
        return BASE_GROUP_KEYS + BUDGET_GROUP_KEYS
    return BASE_GROUP_KEYS


def _budget_variation_warning(items: list[dict[str, Any]], runtime_budget_group: bool) -> str:
    if runtime_budget_group:
        return ""
    for field in BUDGET_GROUP_KEYS:
        values = {_norm(item.get(field, "")) for item in items}
        if len(values) > 1:
            return "budget fields vary within this non-runtime group"
    return ""


def _representative_value(items: list[dict[str, Any]], field: str) -> Any:
    if not items:
        return ""
    return items[0].get(field, "")


def _metric_stats_from_items(items: list[dict[str, Any]], metric: str) -> tuple[dict[str, float | int], str]:
    values: list[float] = []
    summaries: list[tuple[int, float, float]] = []
    summary_nan_count = 0
    degraded_means: list[float] = []
    summary_mean = f"{metric}_mean"
    raw_values = f"{metric}_values"
    for item in items:
        if raw_values in item:
            try:
                values.extend(float(v) for v in json.loads(item[raw_values]))
                continue
            except Exception:
                pass
        if metric in item:
            values.append(float(item[metric]))
            continue
        if summary_mean in item:
            try:
                mean = float(item[summary_mean])
                n_raw = item.get(f"{metric}_n")
                std_raw = item.get(f"{metric}_std")
                summary_nan_count += int(float(item.get(f"{metric}_nan_count", 0) or 0))
                if n_raw is None or std_raw is None:
                    degraded_means.append(mean)
                    continue
                n = int(float(n_raw))
                std = float(std_raw)
                if n <= 0 or math.isnan(mean) or math.isnan(std):
                    degraded_means.append(mean)
                    continue
                summaries.append((n, mean, std))
            except Exception:
                continue
    if values:
        return _stats(values), "raw"
    if summaries and not degraded_means:
        return _pooled_summary_stats(summaries, summary_nan_count), "pooled_summary"
    if summaries:
        degraded_means.extend(mean for _n, mean, _std in summaries)
    if degraded_means:
        stats = _stats(degraded_means)
        stats["nan_count"] = int(stats["nan_count"]) + summary_nan_count
        return stats, "run_level_summary_only_unweighted"
    return _stats([]), "no_data"


def _stats(values: list[float]) -> dict[str, float | int]:
    finite = [v for v in values if not math.isnan(v)]
    nan_count = len(values) - len(finite)
    n = len(finite)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "sem": float("nan"), "ci95": float("nan"), "n": 0, "nan_count": nan_count}
    mean = sum(finite) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in finite) / (n - 1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n else float("nan")
    return {"mean": mean, "std": std, "sem": sem, "ci95": 1.96 * sem, "n": n, "nan_count": nan_count}


def _pooled_summary_stats(summaries: list[tuple[int, float, float]], nan_count: int) -> dict[str, float | int]:
    total_n = sum(n for n, _mean, _std in summaries)
    if total_n <= 0:
        return {"mean": float("nan"), "std": float("nan"), "sem": float("nan"), "ci95": float("nan"), "n": 0, "nan_count": nan_count}
    pooled_mean = sum(n * mean for n, mean, _std in summaries) / total_n
    if total_n > 1:
        ss = sum((n - 1) * (std**2) + n * ((mean - pooled_mean) ** 2) for n, mean, std in summaries)
        pooled_std = math.sqrt(max(ss / (total_n - 1), 0.0))
    else:
        pooled_std = 0.0
    sem = pooled_std / math.sqrt(total_n)
    return {"mean": pooled_mean, "std": pooled_std, "sem": sem, "ci95": 1.96 * sem, "n": total_n, "nan_count": nan_count}


def _aggregation_mode(modes: list[str]) -> str:
    relevant = {m for m in modes if m != "no_data"}
    if not relevant:
        return "no_data"
    if "raw" in relevant:
        return "raw"
    if "run_level_summary_only_unweighted" in relevant:
        return "run_level_summary_only_unweighted"
    if relevant == {"pooled_summary"}:
        return "pooled_summary"
    return "+".join(sorted(relevant))


def _parse_counts(item: dict[str, Any], counts_key: str, mode_key: str) -> Counter[str]:
    if counts_key in item:
        value = item[counts_key]
        if value is None or value == "":
            pass
        elif isinstance(value, str):
            try:
                parsed = json.loads(value)
                return Counter({str(k): int(v) for k, v in parsed.items()})
            except Exception:
                return Counter({value: 1})
        elif isinstance(value, dict):
            return Counter({str(k): int(v) for k, v in value.items()})
    if mode_key in item:
        mode = str(item[mode_key])
        if mode:
            return Counter({mode: int(item.get("sample_count", 1) or 1)})
    return Counter()


def _expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            raw = sorted(path.rglob("results_raw.jsonl"))
            raw_dirs = {p.parent for p in raw}
            summaries = [p for p in sorted(path.rglob("results_summary.jsonl")) if p.parent not in raw_dirs]
            paths.extend(raw)
            paths.extend(summaries)
        else:
            paths.append(path)
    return [p for p in paths if p.exists()]


def _read_skipped_for_inputs(inputs: list[str]) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.rglob("skipped_combinations.jsonl")))
        elif path.name == "skipped_combinations.jsonl":
            paths.append(path)
        else:
            sibling = path.parent / "skipped_combinations.jsonl"
            if sibling.exists():
                paths.append(sibling)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in _read_jsonl(path):
            key = json.dumps(row, sort_keys=True)
            if key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _latex_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "pde": row.get("pde", ""),
                "task": row.get("task", ""),
                "baseline": row.get("baseline", ""),
                "train_size": row.get("train_size", ""),
                "method_budget_label": row.get("method_budget_label", ""),
                "scalar_param_mode": row.get("scalar_param_mode", ""),
                "relative_l2_solution": _pm(row, "relative_l2_solution"),
                "mse": _pm(row, "mse"),
                "pde_residual": _pm(row, "pde_residual"),
                "physics_loss": _pm(row, "physics_loss"),
                "n": row.get("relative_l2_solution_n", 0),
                "nan_count": row.get("relative_l2_solution_nan_count", 0),
                "residual_mode_counts": row.get("residual_mode_counts", "{}"),
                "assimilation_mode_counts": row.get("assimilation_mode_counts", "{}"),
            }
        )
    return out


def _pm(row: dict[str, Any], metric: str) -> str:
    mean = row.get(f"{metric}_mean", float("nan"))
    ci = row.get(f"{metric}_ci95", float("nan"))
    try:
        return f"{float(mean):.4g} +/- {float(ci):.2g}"
    except Exception:
        return "nan"


def _latex_tex(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    header = list(rows[0].keys())
    lines = ["\\begin{tabular}{" + "l" * len(header) + "}", " \\hline", " & ".join(header) + " \\\\", " \\hline"]
    for row in rows:
        lines.append(" & ".join(str(row.get(k, "")).replace("_", "\\_") for k in header) + " \\\\")
    lines.extend([" \\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def _norm(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    return value


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).lower() in {"1", "true", "yes", "y"}


def _has_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _sort_key(key: tuple[tuple[str, Any], ...]) -> tuple[tuple[str, str], ...]:
    return tuple((field, str(value)) for field, value in key)


if __name__ == "__main__":
    main()
