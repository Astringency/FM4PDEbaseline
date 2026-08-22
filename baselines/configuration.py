from __future__ import annotations

from typing import Any, Mapping


def resolve_method_config(
    config: Mapping[str, Any],
    *,
    baseline: str,
    pde: str,
    epochs: int | None = None,
    lr: float | None = None,
    steps: int | None = None,
    refine_steps: int | None = None,
    particles: int | None = None,
    implementation_mode: str | None = None,
    official_backend: str | None = None,
    method_overrides: Mapping[str, Any] | None = None,
    device: str = "cpu",
    seed: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Resolve only the configuration that can affect one baseline execution.

    Keeping this resolver independent from the CLI lets the matrix builder and
    the runner hash exactly the same effective method configuration. Unrelated
    ``method_by_baseline`` sections are deliberately excluded.
    """

    merged: dict[str, Any] = dict(config.get("method", {}) or {})
    merged.update(
        (config.get("method_by_baseline", {}) or {}).get(baseline, {}) or {}
    )
    merged.update((config.get("method_by_pde", {}) or {}).get(pde, {}) or {})
    merged.update(
        (
            (config.get("method_by_baseline_and_pde", {}) or {}).get(
                baseline, {}
            )
            or {}
        ).get(pde, {})
        or {}
    )

    if epochs is not None:
        merged["epochs"] = int(epochs)
    else:
        merged.setdefault("epochs", int(config.get("epochs", 1)))
    if lr is not None:
        merged["lr"] = float(lr)
    else:
        merged.setdefault("lr", float(config.get("learning_rate", 1e-3)))
    if steps is not None:
        merged["steps"] = int(steps)
        if baseline == "pinn_sparse" and bool(merged.get("deepxde_native", False)):
            merged["adam_iterations"] = int(steps)
    if refine_steps is not None:
        merged["refine_steps"] = int(refine_steps)
    if particles is not None:
        merged["particles"] = int(particles)
    if implementation_mode is not None:
        merged["implementation_mode"] = str(implementation_mode)
    if official_backend is not None:
        merged["official_backend"] = str(official_backend)
    if method_overrides:
        merged.update(dict(method_overrides))

    merged["device"] = str(device)
    merged.setdefault("seed", int(seed))
    if dry_run:
        merged["max_steps"] = min(int(merged.get("max_steps", 1) or 1), 1)
        merged["max_val_steps"] = min(int(merged.get("max_val_steps", 1) or 1), 1)
        merged.setdefault("steps", 1)
        merged.setdefault("refine_steps", 1)
        merged["steps"] = min(int(merged.get("steps", 1)), 1)
        merged["refine_steps"] = min(int(merged.get("refine_steps", 1)), 1)
        if baseline == "pinn_sparse" and bool(merged.get("deepxde_native", False)):
            merged["adam_iterations"] = min(
                int(merged.get("adam_iterations", merged["steps"]) or 0), 1
            )
            merged["lbfgs_steps"] = min(int(merged.get("lbfgs_steps", 0) or 0), 1)
        merged["epochs"] = min(int(merged.get("epochs", 1)), 1)
    return merged
