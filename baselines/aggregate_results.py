from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


GROUP_KEYS = [
    "pde",
    "task",
    "baseline",
    "train_size",
    "scalar_param_mode",
    "num_sensors",
    "sensor_mode",
    "noise_level",
    "backend_used",
]

METRICS = [
    "relative_l2_solution",
    "relative_l2_input_or_coeff",
    "mse",
    "mae",
    "obs_mse",
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
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "summary.csv", summary_rows)
    (out / "summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding="utf-8")
    latex_rows = _latex_rows(summary_rows)
    _write_csv(out / "latex_table.csv", latex_rows)
    if args.latex_tex:
        (out / "latex_table.tex").write_text(_latex_tex(latex_rows), encoding="utf-8")
    print(json.dumps({"inputs": [str(p) for p in paths], "groups": len(summary_rows), "output_dir": str(out)}, indent=2))


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(_norm(row.get(k, "")) for k in GROUP_KEYS)
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: kv[0]):
        result = {k: v for k, v in zip(GROUP_KEYS, key)}
        residual_counts: Counter[str] = Counter()
        for item in items:
            residual_counts.update(_parse_residual_counts(item))
        result["residual_mode_counts"] = json.dumps(dict(residual_counts), sort_keys=True)
        result["run_count"] = len(items)
        for metric in METRICS:
            values = _metric_values(items, metric)
            stats = _stats(values)
            for suffix, value in stats.items():
                result[f"{metric}_{suffix}"] = value
        out.append(result)
    return out


def _metric_values(items: list[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    summary_mean = f"{metric}_mean"
    raw_values = f"{metric}_values"
    for item in items:
        if raw_values in item:
            try:
                values.extend(float(v) for v in json.loads(item[raw_values]))
                continue
            except Exception:
                pass
        if summary_mean in item:
            n = int(float(item.get(f"{metric}_n", 1) or 1))
            values.extend([float(item[summary_mean])] * max(n, 1))
        elif metric in item:
            values.append(float(item[metric]))
    return values


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


def _parse_residual_counts(item: dict[str, Any]) -> Counter[str]:
    if "residual_mode_counts" in item:
        value = item["residual_mode_counts"]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return Counter({str(k): int(v) for k, v in parsed.items()})
            except Exception:
                return Counter({value: 1})
        if isinstance(value, dict):
            return Counter({str(k): int(v) for k, v in value.items()})
    if "residual_mode" in item:
        return Counter({str(item["residual_mode"]): int(item.get("sample_count", 1) or 1)})
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
                "scalar_param_mode": row.get("scalar_param_mode", ""),
                "relative_l2_solution": _pm(row, "relative_l2_solution"),
                "mse": _pm(row, "mse"),
                "pde_residual": _pm(row, "pde_residual"),
                "physics_loss": _pm(row, "physics_loss"),
                "n": row.get("relative_l2_solution_n", 0),
                "nan_count": row.get("relative_l2_solution_nan_count", 0),
                "residual_mode_counts": row.get("residual_mode_counts", "{}"),
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


if __name__ == "__main__":
    main()
